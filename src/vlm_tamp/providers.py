"""Official SDK adapters and a deterministic instruction-only offline baseline."""

import base64
import json
import re
from dataclasses import dataclass

from .grounding import (
    Constraint,
    Dependency,
    Entity,
    GroundingPrediction,
    CURRENT_SCHEMA_VERSION,
    Parameter,
    Predicate,
    Stage,
    strict_output_schema,
)


@dataclass
class ProviderResponse:
    text: str
    raw: dict
    input_tokens: int
    output_tokens: int


class MockProvider:
    """No access to reference compartment; does not inspect RGB pixels."""

    def __init__(self, config):
        self.config = config

    def infer(self, request):
        instruction = request.payload["instruction"].lower()
        entities, goals, stages, dependencies, constraints = [], [], [], [], []
        candidates = request.payload.get("candidates", [])
        version = self.config.get("schema_version", CURRENT_SCHEMA_VERSION)

        def entity(description, kind="object"):
            category = next(
                (
                    c
                    for c in (
                        "bowl",
                        "plate",
                        "bottle",
                        "cabinet",
                        "drawer",
                        "stove",
                        "mug",
                        "basket",
                        "microwave",
                    )
                    if c in description
                ),
                description.replace(" ", "_"),
            )
            matches = [c for c in candidates if c["kind"] == "object" and category in c["category"]]
            # A region explicitly naming a drawer is resolvable only through the supplied inventory.
            if "drawer" in description:
                level = next((x for x in ("middle", "top", "bottom") if x in description), "")
                matches = [
                    c
                    for c in candidates
                    if c["kind"] == "region"
                    and "cabinet" in c["identifier"]
                    and c["identifier"].endswith(f"_{level}_region")
                ]
                kind = "part"
            resolved = matches[0]["identifier"] if len(matches) == 1 else None
            if resolved and resolved in [e.reference_id for e in entities]:
                return next(e.id for e in entities if e.reference_id == resolved)
            eid = f"entity_{len(entities)}"
            parent = None
            if version == "1.2" and resolved:
                candidate = matches[0]
                kind = candidate["kind"]
                if candidate["parent"]:
                    parents = [e for e in entities if e.reference_id == candidate["parent"]]
                    if not parents:
                        parent_candidate = next(
                            (c for c in candidates if c["identifier"] == candidate["parent"]), None
                        )
                        if parent_candidate is not None:
                            parent = eid
                            entities.append(
                                Entity(
                                    id=parent,
                                    category=parent_candidate["category"],
                                    description="parent object",
                                    reference_id=parent_candidate["identifier"],
                                    evidence="oracle_inventory",
                                )
                            )
                            eid = f"entity_{len(entities)}"
                    else:
                        parent = parents[0].id
            if version == "1.2" and kind in {"part", "region"} and parent is None:
                parent = eid
                entities.append(
                    Entity(id=parent, category="cabinet", description="unresolved parent cabinet")
                )
                eid = f"entity_{len(entities)}"
            entities.append(
                Entity(
                    id=eid,
                    category=category,
                    description=description,
                    kind=kind,
                    parent=parent,
                    reference_id=resolved,
                    ambiguous=len(matches) > 1,
                    evidence="oracle_inventory" if resolved else "instruction",
                )
            )
            return eid

        # Deliberately small rules: imperfections remain visible in offline reports.
        opening = re.search(r"(?:open|close) (?:the )?((?:middle|top|bottom) drawer|microwave)", instruction)
        if opening:
            target = entity(opening.group(1))
            goals.append(
                Predicate(name="Open" if "open" in opening.group(0) else "Close", arguments=[target])
            )
        relation = re.search(
            r"(?:put|place) (?:the |it |them )?(.+?) (on top of|on|inside|in) (?:the )?(.+)$", instruction
        )
        if relation:
            source = entity(relation.group(1))
            target = entity(relation.group(3))
            goals.append(
                Predicate(
                    name="On" if relation.group(2).startswith("on") else "In", arguments=[source, target]
                )
            )
        elif "put the bowl inside" in instruction:
            source = entity("bowl")
            target = entity("top drawer")
            goals.append(Predicate(name="In", arguments=[source, target]))
        if "turn on" in instruction or "turn off" in instruction:
            goals.append(
                Predicate(
                    name="Turnon" if "turn on" in instruction else "Turnoff", arguments=[entity("stove")]
                )
            )
        if "partial-order" in request.prompt and goals:
            for i, goal in enumerate(goals):
                sid = f"stage_{i}"
                stages.append(
                    Stage(
                        id=sid,
                        action="achieve_" + goal.name.lower(),
                        entities=goal.arguments,
                        subgoals=[goal],
                        requirement="required",
                        evidence="instruction",
                    )
                )
                constraints.append(
                    Constraint(
                        id=f"constraint_{i}",
                        category="goal",
                        type=goal.name.lower(),
                        entities=goal.arguments,
                        stages=[sid],
                        scope="goal",
                        requirement="required",
                        evidence="instruction",
                    )
                )
            if relation and len(goals[-1].arguments) == 2:
                moved = goals[-1].arguments[0]
                stages.append(Stage(id="grasp", action="grasp", entities=[moved], requirement="optional"))
                dependencies.append(
                    Dependency(
                        before="grasp",
                        after=stages[len(goals) - 1].id,
                        requirement="conditional",
                        condition="if transporting with a grasp",
                    )
                )
                constraints.append(
                    Constraint(
                        id="grasp_transport",
                        category="path",
                        type="maintain_grasp",
                        entities=[moved],
                        stages=[stages[len(goals) - 1].id],
                        scope="path",
                        requirement="conditional",
                        condition="if transporting with a grasp",
                        parameters=[Parameter(name="grasp_transform")],
                        evidence="planning_hypothesis",
                    )
                )
        goal_data = [g.model_dump(mode="json") for g in goals]
        if version == "1.2":
            for goal in goal_data:
                goal.update(
                    intent="explicit_terminal"
                    if goal["name"] in {"Open", "Close", "Turnon", "Turnoff"}
                    else "task_outcome",
                    evidence_text=request.payload["instruction"],
                )
        prediction = GroundingPrediction(
            schema_version=version,
            entities=entities,
            goals=goal_data,
            stages=stages,
            dependencies=dependencies,
            constraints=constraints,
            abstentions=["Mock ignores RGB and never estimates image or joint geometry."],
        )
        text = json.dumps(prediction.inference_dict())
        return ProviderResponse(text, {"backend": "deterministic_mock", "output": json.loads(text)}, 0, 0)


class OpenAIProvider:
    def __init__(self, config, client=None):
        from openai import OpenAI

        self.config = config
        # SDK retries disabled: every actual attempt is accounted for by the runner.
        self.client = client or OpenAI(timeout=config["timeout_seconds"], max_retries=0)

    def infer(self, request):
        content = [{"type": "input_text", "text": request.text}]
        if request.image is not None:
            content.append(
                {
                    "type": "input_image",
                    "detail": self.config.get("image_detail", "high"),
                    "image_url": "data:image/png;base64," + base64.b64encode(request.image).decode(),
                }
            )
        options = {}
        if self.config.get("temperature") is not None:
            options["temperature"] = self.config["temperature"]
        if self.config.get("reasoning_effort") is not None:
            options["reasoning"] = {"effort": self.config["reasoning_effort"]}
        if self.config.get("service_tier") is not None:
            options["service_tier"] = self.config["service_tier"]
        response = self.client.responses.create(
            model=self.config["model"],
            instructions=request.prompt,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "GroundingPrediction",
                    "schema": strict_output_schema(self.config.get("schema_version", CURRENT_SCHEMA_VERSION)),
                    "strict": True,
                }
            },
            max_output_tokens=self.config["max_output_tokens"],
            store=False,
            **options,
        )
        usage = response.usage
        return ProviderResponse(
            response.output_text,
            response.model_dump(mode="json"),
            usage.input_tokens if usage else 0,
            usage.output_tokens if usage else 0,
        )


class GeminiProvider:
    def __init__(self, config, client=None):
        from google import genai
        from google.genai import types

        self.config = config
        self.client = client or genai.Client(
            http_options=types.HttpOptions(
                timeout=int(config["timeout_seconds"] * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            )
        )

    def infer(self, request):
        from google.genai import types

        contents = [request.text]
        if request.image is not None:
            contents.append(types.Part.from_bytes(data=request.image, mime_type="image/png"))
        response = self.client.models.generate_content(
            model=self.config["model"],
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=request.prompt,
                response_mime_type="application/json",
                response_schema=GroundingPrediction,
                max_output_tokens=self.config["max_output_tokens"],
                temperature=0,
            ),
        )
        usage = response.usage_metadata
        return ProviderResponse(
            response.text or "",
            response.model_dump(mode="json"),
            (usage.prompt_token_count or 0) if usage else 0,
            ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        )


def make_provider(config):
    return {"mock": MockProvider, "openai": OpenAIProvider, "gemini": GeminiProvider}[config["provider"]](
        config
    )


def retryable(error):
    # Never retry malformed requests, authentication failures or schema-invalid output.
    status = getattr(error, "status_code", getattr(error, "code", None))
    return status in {408, 429, 500, 502, 503, 504} or type(error).__name__ in {
        "APITimeoutError",
        "APIConnectionError",
        "TimeoutException",
        "ConnectError",
    }
