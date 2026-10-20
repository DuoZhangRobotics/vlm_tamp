"""Evidence-backed draft constraint references and a local human review interface."""

import hashlib
import html
import json
from pathlib import Path

from .annotations import Coverage, Provenance, ReferenceAnnotation, ReferencePlan
from .bddl import normalize_predicate
from .dataset import read_records, safe_path
from .grounding import Constraint, Dependency, Parameter, Stage
from .references import attach_reference, load_simulator_references
from .visualization import overlay


def _proposal(record, simulator_reference):
    """Construct review proposals; these are not expert labels or model predictions."""
    goals = record["reference"]["goal_predicates"]
    instruction = record["model_inputs"]["instruction"]
    stages, dependencies, constraints = [], [], []
    opening = next((g["arguments"][0] for g in goals if normalize_predicate(g["name"]) == "open"), None)
    placement = next((g for g in goals if normalize_predicate(g["name"]) in {"on", "in"}), None)
    if opening is None and placement and "open" in instruction.lower():
        opening = placement["arguments"][1]
    if opening:
        stages += [
            Stage(
                id="engage",
                action="establish_contact",
                entities=[opening],
                requirement="conditional",
                condition="if opening is performed through robot contact",
                evidence="planning_hypothesis",
            ),
            Stage(
                id="open",
                action="open",
                entities=[opening],
                requirement="conditional",
                condition="if the requested opening/access state is not already satisfied",
                evidence="instruction",
            ),
        ]
        dependencies.append(
            Dependency(
                before="engage",
                after="open",
                requirement="conditional",
                condition="if contact-based actuation is selected",
                evidence="planning_hypothesis",
            )
        )
    if placement:
        moved, destination = placement["arguments"]
        stages += [
            Stage(
                id="grasp",
                action="grasp",
                entities=[moved],
                requirement="conditional",
                condition="if grasp-based transport is selected and the object is not already held",
                evidence="planning_hypothesis",
            ),
            Stage(
                id="transport",
                action="transport",
                entities=[moved],
                requirement="conditional",
                condition="if relocation is required",
                evidence="planning_hypothesis",
            ),
            Stage(
                id="place",
                action="place",
                entities=[moved, destination],
                requirement="conditional",
                condition="if the final relation is not already satisfied",
                evidence="instruction",
            ),
            Stage(
                id="release",
                action="release",
                entities=[moved],
                requirement="conditional",
                condition="if a held object must be left at the destination",
                evidence="planning_hypothesis",
            ),
        ]
        for before, after in (("grasp", "transport"), ("transport", "place"), ("place", "release")):
            dependencies.append(
                Dependency(
                    before=before,
                    after=after,
                    requirement="conditional",
                    condition="for the proposed grasp-based placement sequence",
                    evidence="planning_hypothesis",
                )
            )
        if opening:
            dependencies.append(
                Dependency(
                    before="open",
                    after="place",
                    requirement="conditional",
                    condition="if closed access blocks placement",
                    evidence="planning_hypothesis",
                )
            )
        constraints.append(
            Constraint(
                id="grasp_transport",
                category="contact",
                type="maintain_grasp",
                entities=[moved],
                stages=["transport"],
                scope="path",
                requirement="conditional",
                condition="if transport uses a maintained grasp",
                parameters=[Parameter(name="grasp_transform")],
                evidence="planning_hypothesis",
                verified=False,
            )
        )
    for index, goal in enumerate(goals):
        constraints.append(
            Constraint(
                id=f"published_goal_{index}",
                category="goal",
                type=normalize_predicate(goal["name"]),
                entities=goal["arguments"],
                scope="goal",
                requirement="required",
                evidence="instruction",
                verified=False,
            )
        )
    if simulator_reference and opening:
        fact = next((p for p in simulator_reference.parts if p.entity_id == opening), None)
        if fact:
            constraints.append(
                Constraint(
                    id="joint_motion",
                    category="articulation",
                    type="follow_joint",
                    joint_type=fact.joint_type,
                    articulated_part=opening,
                    entities=[opening],
                    stages=["open"],
                    scope="path",
                    requirement="conditional",
                    condition="if this verified articulated body is actuated during opening",
                    parameters=[
                        Parameter(
                            name="axis",
                            value=fact.axis_world,
                            status="known",
                            frame="world",
                            units="unit_vector",
                            evidence="oracle_geometry",
                        ),
                        Parameter(
                            name="axis_point",
                            value=fact.joint_position_world,
                            status="known",
                            frame="world",
                            units="m",
                            evidence="oracle_geometry",
                        ),
                    ],
                    evidence="planning_hypothesis",
                    verified=False,
                )
            )
    provenance = [
        Provenance(
            source="official_benchmark",
            evidence=f"Published final predicates and instruction; BDDL SHA256 {record['metadata']['bddl_sha256']}",
        ),
        Provenance(
            source="derived_rule",
            evidence="Unreviewed conditional stage/contact proposals. Neither published goals nor joint kinematics prove required manipulation strategies.",
        ),
    ]
    if simulator_reference:
        provenance.append(
            Provenance(
                source="simulator",
                evidence="Separately verified site/joint/body/mask/state facts. Known axis parameters come from that evidence; task necessity remains unreviewed.",
            )
        )
    return ReferenceAnnotation(
        example_id=record["example_id"],
        status="draft",
        provenance=provenance,
        relevant_entities=sorted({a for g in goals for a in g["arguments"]}),
        manipulable_parts=None,
        alternatives=[
            ReferencePlan(
                id="conditional_proposal_for_review",
                stages=stages,
                dependencies=dependencies,
                constraints=constraints,
            )
        ],
        coverage=Coverage(),
        gaps=[
            "Human expert must confirm relevant entities, manipulable parts and appropriate stage granularity.",
            "Review conditional assumptions against the observed initial state; remove unsupported proposals or add valid alternatives.",
            "No task-specific grasp/contact/orientation/path requirement is established by final BDDL goals alone.",
            "Kinematic facts do not establish a required task plan. All proposed constraints remain unverified.",
            "Coverage is disabled until human review; completeness must be assessed independently.",
        ],
    )


REVIEW_SCRIPT = """
function exportReview(button) {
  const card = button.closest('article');
  const message = card.querySelector('.message');
  try {
    const reviewer = card.querySelector('.reviewer').value.trim();
    if (!reviewer) throw Error('Enter the name of the human reviewer.');
    const annotation = JSON.parse(card.querySelector('textarea').value);
    if (annotation.example_id !== card.dataset.example) throw Error('The example ID must match this observation.');
    const tracks = ['stages','dependencies','constraints'];
    let checked = 0;
    for (const track of tracks) {
      const reviewed = card.querySelector('.review-' + track).checked;
      const complete = card.querySelector('.complete-' + track).checked;
      if (complete && !reviewed) throw Error('A complete track must also be reviewed.');
      annotation.coverage[track] = reviewed;
      annotation.coverage[track + '_complete'] = complete;
      checked += Number(reviewed);
    }
    if (!checked) throw Error('Review at least one track before exporting.');
    if (!card.querySelector('.attest').checked) throw Error('Confirm that you checked the proposals and their evidence.');
    annotation.status = 'reviewed';
    for (const source of annotation.provenance) source.reviewer = reviewer;
    annotation.provenance.push({source:'human', reviewer:reviewer, evidence:'Reviewed the edited reference against the observation and supplied evidence; track coverage is explicitly selected.'});
    const blob = new Blob([JSON.stringify(annotation, null, 2) + '\\n'], {type:'application/json'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = annotation.example_id + '.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 5000);
    message.textContent = 'Reviewed reference downloaded. Validate it before using it for scoring. Repository drafts remain unchanged.';
  } catch (error) { message.textContent = error.message; }
}
"""


def generate_review_set(dataset, output, simulator_references=None, example_ids=None):
    records, root = read_records(dataset)
    selected = set(example_ids) if example_ids else {r["example_id"] for r in records}
    if not selected <= {r["example_id"] for r in records}:
        raise ValueError("Unknown review example IDs")
    records = [r for r in records if r["example_id"] in selected]
    output = Path(output).resolve()
    if (output / "review.html").exists() or (output / "manifest.json").exists():
        raise FileExistsError("Review set already exists; use a new --output")
    (output / "annotations").mkdir(parents=True, exist_ok=True)
    (output / "evidence").mkdir(exist_ok=True)
    references = load_simulator_references(simulator_references)
    cards, annotations = [], []
    for index, record in enumerate(records):
        eid = record["example_id"]
        ref = references.get(eid)
        annotated_record = attach_reference(record, root, ref) if ref else record
        annotation = _proposal(record, ref)
        annotations.append(annotation)
        safe_path(output, f"annotations/{eid}.json").write_text(annotation.model_dump_json(indent=2) + "\n")
        rgb = safe_path(root, record["model_inputs"]["image"])
        safe_path(output, f"evidence/{eid}-rgb.png").write_bytes(rgb.read_bytes())
        overlay(
            annotated_record, {"prediction": None}, root, safe_path(output, f"evidence/{eid}-references.png")
        )
        evidence = {
            "example_id": eid,
            "instruction": record["model_inputs"]["instruction"],
            "published_goal_expression": record["reference"]["goal_expression"],
            "published_goal_predicates": record["reference"]["goal_predicates"],
            "bddl_sha256": record["metadata"]["bddl_sha256"],
            "initial_state_sha256": record["metadata"]["initial_state_sha256"],
            "image_sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
            "libero_commit": record["metadata"]["versions"]["libero_commit"],
            "simulator_facts": ref.model_dump(mode="json") if ref else None,
            "proposal_status": "unreviewed; excluded from Track B scoring",
        }
        safe_path(output, f"evidence/{eid}.json").write_text(json.dumps(evidence, indent=2) + "\n")
        controls = "".join(
            f'<label><input type="checkbox" class="review-{track}">Reviewed {track}</label> '
            f'<label><input type="checkbox" class="complete-{track}">Complete {track} coverage, including alternatives</label><br>'
            for track in ("stages", "dependencies", "constraints")
        )
        cards.append(
            f'<article data-example="{html.escape(eid, quote=True)}"><h2>{html.escape(record["model_inputs"]["instruction"])}</h2>'
            f"<p>{html.escape(eid)} · <strong>DRAFT — human review pending</strong></p>"
            f'<div class="images"><img src="evidence/{eid}-rgb.png" alt="Original observation"><img src="evidence/{eid}-references.png" alt="Measured object and part references"></div>'
            "<p>Green: measured whole objects. Gold: verified visible drawer bodies, not interior volumes or handle affordances.</p>"
            f"<details><summary>Published goals and independent simulator evidence</summary><pre>{html.escape(json.dumps(evidence, indent=2))}</pre></details>"
            "<p>Check which actions are needed in this initial state, valid alternatives, contact strategy, part identity and coordinate frames. "
            "Edit or remove unsupported proposals. The JSON is editable. Leave completeness unchecked unless precision can be judged.</p>"
            f'<textarea aria-label="Editable draft reference">{html.escape(annotation.model_dump_json(indent=2))}</textarea>'
            '<p><label>Human reviewer name <input class="reviewer" type="text"></label></p>'
            f'<p>{controls}</p><p><label><input type="checkbox" class="attest">I checked the edited reference and its evidence.</label></p>'
            '<button onclick="exportReview(this)">Export reviewed reference</button><p class="message" role="status"></p></article>'
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Constraint reference review</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;background:#f3f5fa;color:#18263a}"
        "article{padding:24px;background:white;margin:24px 0;border-radius:12px}.images{display:flex;gap:20px;flex-wrap:wrap}"
        "img{width:min(100%,460px)}pre{white-space:pre-wrap;overflow-wrap:anywhere}textarea{width:100%;min-height:420px;box-sizing:border-box;font:13px monospace}"
        "input,button{font:inherit}button{padding:10px 16px}.message{color:#315776}</style><body>"
        "<h1>Constraint reference review</h1><p>Four-task pilot proposals with independent evidence. Drafts do not count as reviewed labels. "
        "Your name, explicit track coverage and confirmation are required to export a reviewed reference. This page works locally and sends no data.</p>"
        + "\n".join(cards)
        + "<script>"
        + REVIEW_SCRIPT
        + "</script></body></html>"
    )
    (output / "review.html").write_text(page)
    manifest = {
        "status": "draft_human_review_pending",
        "examples": [r["example_id"] for r in records],
        "dataset_sha256": hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reviewed_annotations": 0,
        "scope": "published final goals and separately verified kinematics are evidence; stage/contact/path content is conditional, unreviewed proposal",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Constraint review set\n\nOpen `review.html` to inspect each original image, measured references, evidence and editable proposal. "
        "Enter your name, revise the JSON, select reviewed tracks and confirm review before exporting. "
        "Export does not modify repository drafts. Save reviewed JSON files in a separate directory, then run "
        "`vlm-tamp validate-annotations --annotations PATH` before evaluating with them.\n\n"
        "All saved annotations are drafts with disabled coverage. Proposed constraints are unverified; published final goals and simulator kinematics "
        "do not prove required manipulation stages or contact strategies. The exact verified axes remain separate evidence.\n"
    )
    return annotations
