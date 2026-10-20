"""Compact RGB task meaning, separate from simulator identity and geometric compilation."""

from typing import Literal

from pydantic import Field, model_validator

from .grounding import Model, Unit


Relation = Literal["On", "In", "InFrontOf", "Open", "Close", "Turnon", "Turnoff", "At", "Stack"]
Selector = Literal[
    "whole",
    "interior",
    "top_drawer",
    "middle_drawer",
    "bottom_drawer",
    "door",
    "front_area",
    "support_surface",
    "handle",
    "other",
]
ARITIES = {
    "On": 2,
    "In": 2,
    "InFrontOf": 2,
    "Open": 1,
    "Close": 1,
    "Turnon": 1,
    "Turnoff": 1,
    "At": 2,
    "Stack": 2,
}


class RGBEntity(Model):
    id: str = Field(description="A short local name for this image referent, not a simulator identifier.")
    category: str
    description: str
    kind: Literal["object", "part", "region"] = "object"
    parent: str | None = Field(default=None, description="Local owner object ID for a part or region.")
    selector: Selector = "whole"
    point: list[Unit] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Point on a visible pixel of this referent, normalized [x,y], top-left origin. For an invisible interior, use null and localize the parent.",
    )
    bbox: list[Unit] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional tight visible bounding box [xmin,ymin,xmax,ymax].",
    )

    @model_validator(mode="after")
    def localization(self):
        if self.bbox and (self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]):
            raise ValueError("Image box must have positive width and height")
        if self.kind == "object" and (self.parent is not None or self.selector != "whole"):
            raise ValueError("Whole objects have no ownership parent and selector whole")
        if self.kind != "object" and (self.parent is None or self.selector == "whole"):
            raise ValueError("Parts/regions need an owner and a specific semantic selector")
        return self


class RGBGoal(Model):
    relation: Relation
    arguments: list[str]

    @model_validator(mode="after")
    def arity(self):
        if len(self.arguments) != ARITIES[self.relation]:
            raise ValueError("Goal relation has the wrong argument count")
        return self


class RGBStage(Model):
    id: str
    action: str
    entities: list[str] = Field(default_factory=list)
    condition: str | None = None


class RGBConstraint(Model):
    description: str
    entities: list[str]
    scope: Literal["goal", "path", "transition"]
    requirement: Literal["required", "conditional", "optional"]
    condition: str | None = None
    mechanism: Literal["prismatic", "revolute", "unknown"] | None = None
    verified: Literal[False] = False


class RGBPrediction(Model):
    schema_version: Literal["rgb-1.0"] = "rgb-1.0"
    entities: list[RGBEntity]
    goals: list[RGBGoal]
    stages: list[RGBStage] = Field(default_factory=list)
    constraints: list[RGBConstraint] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self):
        by_id = {e.id: e for e in self.entities}
        if len(by_id) != len(self.entities):
            raise ValueError("Local entity IDs must be unique")
        if len({s.id for s in self.stages}) != len(self.stages):
            raise ValueError("Local stage IDs must be unique")
        for entity in self.entities:
            seen = {entity.id}
            parent = entity.parent
            while parent is not None:
                if parent not in by_id or parent in seen:
                    raise ValueError("Ownership needs declared acyclic parent IDs")
                seen.add(parent)
                parent = by_id[parent].parent
        for goal in self.goals:
            if not set(goal.arguments) <= set(by_id):
                raise ValueError("Final goals must reference declared local entities")
        for item in [*self.stages, *self.constraints]:
            if not set(item.entities) <= set(by_id):
                raise ValueError("Stages and constraints must reference declared local entities")
        return self


def rgb_output_schema():
    schema = RGBPrediction.model_json_schema()

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
    schema["$defs"]["RGBGoal"] = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "relation": {"type": "string", "const": relation},
                    "arguments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": arity,
                        "maxItems": arity,
                    },
                },
                "required": ["relation", "arguments"],
                "additionalProperties": False,
            }
            for relation, arity in ARITIES.items()
        ]
    }
    return schema


def from_legacy(payload):
    """Explicit representation adapter for saved predictions, with no reference access."""
    if not payload:
        raise ValueError("No saved model declaration")
    entities = []
    for source in payload.get("entities", []):
        kind = source.get("kind", "object")
        description = source.get("description", "")
        words = (
            (source.get("id", "") + " " + description + " " + source.get("category", ""))
            .lower()
            .replace("_", " ")
        )
        selector = "whole"
        if kind != "object":
            selector = next(
                (
                    level + "_drawer"
                    for level in ("top", "middle", "bottom")
                    if level in words and "drawer" in words
                ),
                None,
            )
            selector = selector or (
                "door"
                if "door" in words
                else "interior"
                if any(w in words for w in ("interior", "contain", "heating"))
                else "front_area"
                if "front" in words
                else "support_surface"
                if "surface" in words
                else "other"
            )
        entities.append(
            RGBEntity(
                id=source["id"],
                category=source.get("category", "object"),
                description=description,
                kind=kind,
                parent=source.get("parent") if kind != "object" else None,
                selector=selector,
                point=source.get("point"),
                bbox=source.get("bbox"),
            )
        )
    aliases = {r.lower(): r for r in ARITIES}
    goals = [
        RGBGoal(relation=aliases.get(g["name"].lower(), g["name"]), arguments=g["arguments"])
        for g in payload.get("goals", [])
    ]
    # Saved stage/geometry alias errors are left out of this goal-only diagnostic.
    return RGBPrediction(
        entities=entities,
        goals=goals,
        uncertainties=["Adapted saved semantic goals; original validation status is retained separately."],
    )
