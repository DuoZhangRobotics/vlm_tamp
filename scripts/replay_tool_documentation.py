"""Compare fresh Luna decisions at recorded points where previously read help was absent."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

from vlm_tamp.actor_structured import OPERATIONS, VISUAL_TOOLS, decision_schema, run_decision
from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.subscription_agent import clean_environment

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STUDY = ROOT / "results/luna-visual-memory-native-images-20261104"
SOURCE_RUNS = (
    "01-close_microwave-s017-cloud_anchors",
    "01-close_microwave-s017-rgb_recent",
    "02-close_top_drawer-s000-rgb_recent",
    "04-place_bowl-s000-cloud_anchors",
)


def select_context(run: Path) -> tuple[Path, dict, dict]:
    """Select the first post-motion repeated help request missing from the immediate tail."""
    contracts: dict[str, dict] = {}
    for source in sorted((run / "decisions").glob("*/input.json")):
        folder = source.parent
        if not (folder / "decision.json").exists() or not (folder / "result.json").exists():
            continue
        context = read_json(source)
        decision = read_json(folder / "decision.json")
        result = read_json(folder / "result.json")
        if decision["operation"] != "help":
            continue
        name = json.loads(decision["arguments_json"])["operation"]
        in_tail = any(
            row["operation"] == "help" and json.loads(row["arguments_json"]).get("operation") == name
            for row in context["memory"]["recent_nonmotion_feedback"]
        )
        if name in contracts and not in_tail and context["current_observation"]["control_steps"] > 60:
            return folder, context, contracts
        if result["receipt"].get("ok"):
            contracts[name] = result["receipt"]["result"]
    raise ValueError("No eligible recorded decision: " + str(run))


def main() -> None:
    """Issue model requests only; never create or connect to a robot service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    selected = []
    for name in SOURCE_RUNS:
        run = SOURCE_STUDY / "trials" / name
        folder, context, contracts = select_context(run)
        images = [run / "live-images" / item["path"] for item in context["images"]]
        for item, image in zip(context["images"], images):
            if hashlib.sha256(image.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Recorded public image hash mismatch: " + str(image))
        selected.append((folder, context, contracts, images))
    protocol = {
        "source_study": str(SOURCE_STUDY),
        "selection": "First repeated help absent from immediate history after physical motion, per declared run",
        "model": "gpt-6-luna",
        "reasoning": "high",
        "conditions": ["transient_docs", "persistent_docs"],
        "robot_motion_executed": False,
        "task_success_measured": False,
        "additional_information": "Only exact tool schemas previously returned by help in that episode",
        "contexts": [
            {
                "source": str(folder),
                "control_steps": context["current_observation"]["control_steps"],
                "contracts": sorted(contracts),
                "added_documentation_bytes": len(json.dumps(contracts, separators=(",", ":")).encode()),
            }
            for folder, context, contracts, _ in selected
        ],
    }
    write_json(output / "protocol.json", protocol)
    if args.prepare_only:
        return
    rows = []
    for index, (folder, context, contracts, images) in enumerate(selected):
        prefix = (folder / "prompt.txt").read_text().rsplit("\n", 1)[0]
        operations = [name for name in OPERATIONS if name in context["tools"]]
        operations += [
            "help",
            "inspect_image",
            *sorted(VISUAL_TOOLS if context["visual_tools_available"] else []),
        ]
        conditions = protocol["conditions"] if index % 2 == 0 else list(reversed(protocol["conditions"]))
        for condition in conditions:
            payload = dict(context)
            if condition == "persistent_docs":
                payload["previously_consulted_tool_contracts"] = contracts
            destination = output / f"{index + 1:02d}-{condition}"
            logging.info("Recorded context %d, %s; no robot connection", index + 1, condition)
            decision = run_decision(
                prefix + "\n" + json.dumps(payload, separators=(",", ":")),
                images,
                destination,
                clean_environment(),
                decision_schema(operations),
                "gpt-6-luna",
                "high",
                time.monotonic() + 300,
            )
            write_json(destination / "input.json", payload)
            if decision is None:
                raise TimeoutError("Recorded decision exceeded its 300-second allowance")
            requested_help = (
                json.loads(decision["arguments_json"]).get("operation")
                if decision["operation"] == "help"
                else None
            )
            rows.append(
                {
                    "context": index + 1,
                    "condition": condition,
                    "source": str(folder),
                    "decision": decision,
                    "requested_previously_consulted_help": requested_help in contracts,
                    "timing": read_json(destination / "timing.json"),
                    "robot_motion_executed": False,
                }
            )
            write_json(output / "summary.json", {"complete": False, "comparisons": rows})
    write_json(output / "summary.json", {"complete": True, "comparisons": rows})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
