"""Reviewed, coverage-aware Track B annotations, including alternative plans."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .grounding import Constraint, Dependency, Model, Stage


class Provenance(Model):
    source: Literal["official_benchmark", "simulator", "derived_rule", "human"]
    evidence: str
    reviewer: str | None = None


class ReferencePlan(Model):
    id: str
    stages: list[Stage] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_plan(self):
        ids = [s.id for s in self.stages]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate reference stage IDs")
        for d in self.dependencies:
            if d.before not in ids or d.after not in ids:
                raise ValueError("Unknown reference stage dependency")
        for c in self.constraints:
            if not set(c.stages) <= set(ids):
                raise ValueError("Unknown reference constraint stage")
        return self


class Coverage(Model):
    stages: bool = False
    dependencies: bool = False
    constraints: bool = False
    stages_complete: bool = False
    dependencies_complete: bool = False
    constraints_complete: bool = False
    constraint_categories: list[str] = Field(default_factory=list)


class ReferenceAnnotation(Model):
    schema_version: Literal["1.0"] = "1.0"
    example_id: str
    status: Literal["draft", "reviewed"] = "draft"
    provenance: list[Provenance] = Field(default_factory=list)
    relevant_entities: list[str] | None = None
    manipulable_parts: list[str] | None = None
    alternatives: list[ReferencePlan] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    axis_sign_invariant: bool = True
    gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reviewed_evidence(self):
        if self.status == "reviewed" and (
            not self.provenance or not all(p.reviewer and p.evidence for p in self.provenance)
        ):
            raise ValueError("Reviewed annotations require named reviewers and evidence")
        return self


def load_annotations(path):
    if path is None:
        return {}
    root = Path(path)
    files = sorted(root.glob("*.json")) if root.is_dir() else [root]
    result = {}
    for file in files:
        annotation = ReferenceAnnotation.model_validate_json(file.read_text())
        if annotation.example_id in result:
            raise ValueError("Duplicate annotation example ID")
        result[annotation.example_id] = annotation
    return result


def annotation_templates(records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = output / f"{record['example_id']}.json"
        if path.exists():
            continue
        annotation = ReferenceAnnotation(
            example_id=record["example_id"],
            gaps=[
                "Expert review needed for relevant entities and manipulable parts.",
                "BDDL final goals do not establish intermediate stages or path requirements.",
                "Joint metadata supplies kinematics, not required task constraints.",
            ],
        )
        path.write_text(annotation.model_dump_json(indent=2))
    (output / "schema.json.schema").write_text(json.dumps(ReferenceAnnotation.model_json_schema(), indent=2))
