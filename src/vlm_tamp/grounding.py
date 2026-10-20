"""Provider-independent semantic declarations, never executable VLM code."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from .bddl import normalize_predicate


Unit = Annotated[float, Field(ge=0, le=1)]
Evidence = Literal["instruction", "observation", "oracle_inventory", "oracle_geometry", "planning_hypothesis"]
Requirement = Literal["required", "conditional", "optional"]
CURRENT_SCHEMA_VERSION = "1.2"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Entity(Model):
    id: str = Field(description="Local semantic ID used by goal/stage arguments, not a simulator identity.")
    category: str
    description: str
    kind: Literal["object", "part", "region"] = "object"
    parent: str | None = Field(
        default=None,
        description="Local semantic parent entity ID, declared in entities, or null. This is not an inventory identifier.",
    )
    reference_id: str | None = Field(
        default=None,
        description="Exact supplied candidate identifier if resolved; null without candidate inventory or if ambiguous.",
    )
    point: list[Unit] | None = Field(
        default=None, min_length=2, max_length=2, description="Normalized [x,y] point, top-left image origin."
    )
    bbox: list[Unit] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Normalized [x_min,y_min,x_max,y_max] corner coordinates; x_min<x_max and y_min<y_max. Never center/width/height.",
    )
    camera: str | None = None
    confidence: Unit | None = None
    ambiguous: bool = False
    evidence: Evidence = "instruction"

    @model_validator(mode="after")
    def valid_localization(self):
        if self.bbox and (self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]):
            raise ValueError("Bounding box must have positive width and height")
        if (self.point or self.bbox) and self.camera is None:
            raise ValueError("Image localization requires a camera reference")
        if self.ambiguous and self.reference_id is not None:
            raise ValueError("Ambiguous entities must abstain from instance identity")
        return self


class Predicate(Model):
    name: str
    arguments: list[str]


class FinalGoal(Predicate):
    intent: Literal["explicit_terminal", "task_outcome", "uncertain"] | None = Field(
        default=None,
        description="In schema 1.2: explicit_terminal for a requested terminal state; task_outcome for the intended result; uncertain if persistence as a final goal is ambiguous. Access actions belong in stages.",
    )
    evidence_text: str | None = Field(
        default=None,
        description="In schema 1.2: a verbatim phrase from the supplied instruction supporting this goal. It is not a benchmark answer.",
    )


class Stage(Model):
    id: str
    action: str
    entities: list[str] = Field(default_factory=list)
    subgoals: list[Predicate] = Field(default_factory=list)
    requirement: Requirement = "optional"
    condition: str | None = None
    evidence: Evidence = "planning_hypothesis"


class Dependency(Model):
    before: str
    after: str
    requirement: Requirement = "conditional"
    condition: str | None = None
    evidence: Evidence = "planning_hypothesis"


class Parameter(Model):
    name: str
    value: float | str | list[float] | None = None
    status: Literal["known", "unresolved"] = "unresolved"
    frame: str | None = None
    units: str | None = None
    evidence: Evidence = "planning_hypothesis"

    @model_validator(mode="after")
    def consistent_resolution(self):
        if (self.status == "known") != (self.value is not None):
            raise ValueError("Known parameters need values; unresolved parameters must be null")
        return self


class GeometryTarget(Model):
    id: str
    entity: str = Field(
        description="Declared local entity ID whose geometry is described. Never a goal argument or invented scene identity."
    )
    role: Literal["support_surface", "interior", "handle", "articulated_body", "other"]
    description: str
    parameters: list[Parameter] = Field(default_factory=list)
    evidence: Evidence = "planning_hypothesis"


class Constraint(Model):
    id: str
    category: Literal["goal", "path", "articulation", "contact", "temporal"]
    type: str
    joint_type: Literal["prismatic", "revolute", "unknown"] | None = Field(
        default=None,
        description="Required for articulation: explicit joint mechanism or unknown. Null for other categories.",
    )
    articulated_part: str | None = Field(
        default=None,
        description="For articulation, the local ID of a declared part/region entity in this constraint's entities. Null otherwise.",
    )
    entities: list[str]
    stages: list[str] = Field(default_factory=list)
    scope: Literal["goal", "path", "transition"]
    requirement: Requirement
    condition: str | None = None
    parameters: list[Parameter] = Field(default_factory=list)
    frame_assumption: str | None = None
    evidence: Evidence
    verified: bool = False


class GroundingPrediction(Model):
    # Unversioned historical payloads remain readable. New inference supplies an explicit version.
    schema_version: Literal["1.0", "1.1", "1.2"] = "1.1"
    entities: list[Entity] = Field(default_factory=list)
    goal_logic: Literal["and", "unresolved"] = "and"
    goals: list[FinalGoal] = Field(default_factory=list)
    geometry_targets: list[GeometryTarget] = Field(default_factory=list)
    stages: list[Stage] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    abstentions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_references(self, info: ValidationInfo):
        ids = [e.id for e in self.entities]
        stage_ids = [s.id for s in self.stages]
        constraint_ids = [c.id for c in self.constraints]
        if any(len(x) != len(set(x)) for x in (ids, stage_ids, constraint_ids)):
            raise ValueError("Entity, stage, and constraint IDs must be unique")
        by_id = {e.id: e for e in self.entities}
        if self.schema_version in {"1.1", "1.2"}:
            for entity in self.entities:
                if entity.parent is not None and (entity.parent not in by_id or entity.parent == entity.id):
                    raise ValueError("Parent must be another declared semantic entity ID")
        for goal in self.goals:
            if not set(goal.arguments) <= set(ids):
                raise ValueError("Goal arguments must refer to declared semantic entity IDs")
        if self.schema_version == "1.2":
            geometry_ids = [target.id for target in self.geometry_targets]
            if len(set(geometry_ids)) != len(geometry_ids) or set(geometry_ids) & set(ids):
                raise ValueError("Geometry target IDs must be unique and distinct from entity IDs")
            for target in self.geometry_targets:
                if target.entity not in by_id:
                    raise ValueError("Geometry target must belong to a declared local entity")
            for entity in self.entities:
                if entity.kind in {"part", "region"} and entity.parent is None:
                    raise ValueError("Parts and regions require a declared local parent entity")
                ancestry, current = set(), entity.id
                while current is not None:
                    if current in ancestry:
                        raise ValueError("Entity parent relationships must be acyclic")
                    ancestry.add(current)
                    current = by_id[current].parent
            arities = {
                "on": 2,
                "in": 2,
                "open": 1,
                "close": 1,
                "turnon": 1,
                "turnoff": 1,
                "at": 2,
                "stack": 2,
            }
            for goal in self.goals:
                if goal.intent is None or not goal.evidence_text or not goal.evidence_text.strip():
                    raise ValueError("Final goals require intent and an instruction evidence phrase")
                if (
                    normalize_predicate(goal.name) not in arities
                    or len(goal.arguments) != arities[normalize_predicate(goal.name)]
                ):
                    raise ValueError("Final goal predicate has an unsupported name or wrong arity")
                if goal.intent == "uncertain" and not self.abstentions:
                    raise ValueError("Uncertain terminal intent requires an explanatory abstention")
            if info.context is not None:
                self.check_input_context(info.context, by_id)
        elif self.geometry_targets or any(
            g.intent is not None or g.evidence_text is not None for g in self.goals
        ):
            raise ValueError("Geometry targets and goal intent require schema 1.2")
        for stage in self.stages:
            if not set(stage.entities) <= set(ids):
                raise ValueError("Unknown stage entity")
            for subgoal in stage.subgoals:
                if not set(subgoal.arguments) <= set(ids):
                    raise ValueError("Unknown subgoal entity")
        for edge in self.dependencies:
            if edge.before not in stage_ids or edge.after not in stage_ids:
                raise ValueError("Dependency must refer to declared stages")
        for constraint in self.constraints:
            if constraint.verified:
                raise ValueError("VLM predictions cannot verify their own constraints")
            if not set(constraint.entities) <= set(ids) or not set(constraint.stages) <= set(stage_ids):
                raise ValueError("Unknown constraint entity or stage")
            if self.schema_version in {"1.1", "1.2"}:
                if constraint.category == "articulation":
                    if constraint.joint_type is None or constraint.articulated_part is None:
                        raise ValueError(
                            "Articulation requires explicit joint_type and articulated_part; use unknown for unresolved mechanism"
                        )
                    if constraint.articulated_part not in constraint.entities:
                        raise ValueError("Articulated part must be included in the constraint's entities")
                    if by_id[constraint.articulated_part].kind not in {"part", "region"}:
                        raise ValueError("Articulated entity must be declared as a part or region")
                elif constraint.joint_type is not None or constraint.articulated_part is not None:
                    raise ValueError("Articulation-specific fields must be null for other categories")
        return self

    def check_input_context(self, context, by_id):
        """Validate against the allowlisted request, never the reference compartment."""
        instruction = " ".join(context["instruction"].lower().split())
        candidates = context.get("candidates")
        supplied = {e["identifier"]: e for e in candidates} if candidates is not None else {}
        for goal in self.goals:
            if " ".join(goal.evidence_text.lower().split()) not in instruction:
                raise ValueError("Goal evidence phrase must appear in the supplied instruction")
        goal_entities = {a for goal in self.goals for a in goal.arguments}
        for entity in self.entities:
            if not context["image_available"] and (entity.point is not None or entity.bbox is not None):
                raise ValueError("Image localization requires an image in the supplied request")
            if entity.reference_id is not None:
                if entity.reference_id not in supplied:
                    raise ValueError("Entity reference_id was not supplied in the candidate inventory")
                candidate = supplied[entity.reference_id]
                if entity.kind != candidate["kind"]:
                    raise ValueError("Entity kind must match its supplied candidate kind")
                expected_parent = candidate["parent"]
                parent_reference = by_id[entity.parent].reference_id if entity.parent else None
                if parent_reference != expected_parent:
                    raise ValueError(
                        "Resolved candidate requires its supplied parent identity through a local entity link"
                    )
            if (
                entity.id in goal_entities
                and entity.kind == "region"
                and entity.reference_id is None
                and candidates is not None
            ):
                parent_ref = by_id[entity.parent].reference_id
                if parent_ref and not any(
                    c["kind"] == "region" and c["parent"] == parent_ref for c in candidates
                ):
                    raise ValueError(
                        "Unsupported derived region in final goal; describe its parent geometry separately"
                    )

    def inference_dict(self):
        data = self.model_dump(mode="json")
        if self.schema_version != "1.2":
            data.pop("geometry_targets")
            for goal in data["goals"]:
                goal.pop("intent")
                goal.pop("evidence_text")
        return data


class GeometricDeclaration(Model):
    """Future compiler boundary; declares unresolved geometry, no planner API guess."""

    semantic_constraint: Constraint
    geometric_parameters: list[Parameter]
    verification_source: str | None = None
    ready_for_compilation: bool = False

    @model_validator(mode="after")
    def readiness(self):
        if self.ready_for_compilation and (
            not self.verification_source or any(p.status != "known" for p in self.geometric_parameters)
        ):
            raise ValueError("Compilation readiness requires verified and resolved geometry")
        return self


def strict_output_schema(version=CURRENT_SCHEMA_VERSION):
    """OpenAI requires all fields and closed objects, including nullable fields."""
    if version == "1.1":
        return json.loads(
            (Path(__file__).resolve().parents[2] / "configs/schemas/grounding_v1_1.json").read_text()
        )
    if version != CURRENT_SCHEMA_VERSION:
        raise ValueError("Unsupported inference schema; legacy 1.0 is evaluation-only")
    schema = GroundingPrediction.model_json_schema()

    def close(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(schema)
    final_properties = schema["$defs"]["FinalGoal"]["properties"]
    final_properties["intent"] = {
        "type": "string",
        "enum": ["explicit_terminal", "task_outcome", "uncertain"],
        "description": final_properties["intent"]["description"],
    }
    final_properties["evidence_text"] = {
        "type": "string",
        "minLength": 1,
        "description": final_properties["evidence_text"]["description"],
    }
    schema["properties"]["schema_version"] = {"type": "string", "const": version}
    return schema
