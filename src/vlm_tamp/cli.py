"""Small command-line surface for collection, inference, annotation and reports."""

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

from .annotations import annotation_templates, load_annotations
from .dataset import read_records, validate_dataset
from .evaluation import evaluate_saved
from .experiment import run_experiment
from .comparison import compare_conditions, compare_experiments
from .references import generate_simulator_references
from .markers import generate_marker_bundle
from .review import generate_review_set
from .linking import link_saved_outputs
from .simulator_probe import collect_physical_facts
from .verifier_trial import run_verifier_trial
from .libero import SUITES, collect
from .visualization import contact_sheet, gallery


def load_credentials(path):
    """Read literal .env values without executing shell code or interpolating secrets."""
    for line in Path(path).read_text().splitlines():
        match = re.match(
            r"^\s*(?:export\s+)?(OPENAI_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.*)$", line
        )
        if not match:
            continue
        values = shlex.split(match.group(2), comments=True)
        if len(values) != 1 or not values[0]:
            raise ValueError("Invalid credential assignment; expected a single literal value")
        os.environ[match.group(1)] = values[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 1 LIBERO grounding, future ST-pRRTC TAMP")
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("collect")
    c.add_argument("--output", required=True)
    c.add_argument("--suites", nargs="+", choices=SUITES, default=["libero_goal", "libero_spatial"])
    c.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    c.add_argument("--states", type=int, nargs="+", default=[0, 1])
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--size", type=int, default=256)
    c.add_argument("--gl", choices=["egl", "glfw", "osmesa"], default="egl")
    for name in ("validate", "contact-sheet", "annotate"):
        p = commands.add_parser(name)
        p.add_argument("--dataset", required=True)
        if name != "validate":
            p.add_argument("--output", required=True)
    p = commands.add_parser("validate-annotations")
    p.add_argument("--annotations", required=True)
    p = commands.add_parser("run")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="configs/offline.json")
    p.add_argument("--output", required=True)
    p.add_argument("--cache")
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-paid", action="store_true")
    p.add_argument("--env-file")
    p = commands.add_parser("simulator-references")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--examples", nargs="+")
    p.add_argument("--gl", choices=["egl", "glfw", "osmesa"], default="egl")
    p = commands.add_parser("object-markers")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--examples", nargs="+")
    p = commands.add_parser("compare")
    p.add_argument("--baseline", required=True)
    p.add_argument("--updated", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--controlled-backbone", action="store_true")
    p = commands.add_parser("compare-conditions")
    p.add_argument("--experiment", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("review-set")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--examples", nargs="+")
    p.add_argument("--simulator-references")
    p = commands.add_parser("link-outputs")
    p.add_argument("--dataset", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--protocol", required=True)
    p = commands.add_parser("physical-probes")
    p.add_argument("--dataset", required=True)
    p.add_argument("--example", required=True)
    p.add_argument("--moving-object", required=True)
    p.add_argument("--target-region", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--gl", choices=["egl", "glfw", "osmesa"], default="egl")
    p = commands.add_parser("verification-trial")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-paid", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--env-file")
    for name in ("evaluate", "report"):
        p = commands.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--experiment", required=True)
        p.add_argument("--output")
        p.add_argument("--annotations")
        p.add_argument("--simulator-references")
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            rows = collect(args.output, args.suites, args.tasks, args.states, args.seed, args.size, args.gl)
            print(f"Saved {len(rows)} official LIBERO observations")
        elif args.command == "validate":
            print(json.dumps(validate_dataset(args.dataset), indent=2))
        elif args.command == "contact-sheet":
            print(contact_sheet(args.dataset, args.output))
        elif args.command == "annotate":
            records, _ = read_records(args.dataset)
            annotation_templates(records, args.output)
            print(f"Created draft annotation templates in {args.output}")
        elif args.command == "validate-annotations":
            annotations = load_annotations(args.annotations)
            print(
                json.dumps(
                    {
                        "annotations": len(annotations),
                        "reviewed": sum(a.status == "reviewed" for a in annotations.values()),
                    }
                )
            )
        elif args.command == "simulator-references":
            refs = generate_simulator_references(args.dataset, args.output, args.examples, args.gl)
            print(f"Verified simulator facts for {len(refs)} observations")
        elif args.command == "object-markers":
            bundle = generate_marker_bundle(args.dataset, args.output, args.examples)
            print(f"Prepared oracle-assisted object markers for {len(bundle['examples'])} observations")
        elif args.command == "compare":
            report = compare_experiments(args.baseline, args.updated, args.output, args.controlled_backbone)
            print(f"Compared {report['paired_predictions']} matching predictions")
        elif args.command == "compare-conditions":
            report = compare_conditions(args.experiment, args.output)
            print(f"Compared object-correspondence assistance on {report['pairs']} observations")
        elif args.command == "review-set":
            annotations = generate_review_set(
                args.dataset, args.output, args.simulator_references, args.examples
            )
            print(f"Prepared {len(annotations)} evidence-backed drafts; human review is pending")
        elif args.command == "link-outputs":
            manifest = link_saved_outputs(args.dataset, args.experiment, args.output, args.protocol)
            print(f"Linked {manifest['completed_predictions']} saved outputs without new model calls")
        elif args.command == "physical-probes":
            facts = collect_physical_facts(
                args.dataset,
                args.example,
                args.output,
                args.moving_object,
                args.target_region,
                args.grid_size,
                args.gl,
            )
            print(f"Measured {len(facts['joints'])} articulated regions; original scene restored")
        elif args.command == "verification-trial":
            if args.env_file:
                load_credentials(args.env_file)
            manifest = run_verifier_trial(
                args.dataset, args.config, args.output, args.allow_paid, args.dry_run
            )
            print(
                json.dumps(
                    {"completed_phases": manifest.get("completed_phases"), "ledger": manifest.get("ledger")},
                    indent=2,
                )
            )
        elif args.command == "run":
            if args.env_file:
                load_credentials(args.env_file)
            manifest = run_experiment(
                args.dataset, args.config, args.output, args.allow_paid, args.dry_run, args.limit, args.cache
            )
            print(
                json.dumps(
                    {
                        "provider": manifest["provider"],
                        "offline_mock": manifest["offline_mock"],
                        "ledger": manifest.get("ledger"),
                    },
                    indent=2,
                )
            )
        elif args.command in {"evaluate", "report"}:
            rows, _ = evaluate_saved(
                args.dataset, args.experiment, args.output, args.annotations, args.simulator_references
            )
            if args.command == "report":
                print(
                    gallery(
                        args.dataset,
                        args.experiment,
                        args.output,
                        args.annotations,
                        args.simulator_references,
                    )
                )
            print(f"Evaluated {len(rows)} predictions")
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
