"""Port the public ViLaIn interpreter to a bounded official API, with RGB-only detection."""

import ast
import base64
import json
import sys
import types

from vlm_tamp.baseline_api import BoundedChat
from vlm_tamp.cli import load_credentials
from vlm_tamp.libero import ROOT


def load_native():
    repo = ROOT / "third_party/ViLaIn-TAMP/vilain"
    package = types.ModuleType("vilain")
    package.__path__ = [str(repo)]
    sys.modules["vilain"] = package
    tree = ast.parse((repo / "vilain.py").read_text())
    # rospy is imported but unused in this interpreter; controllers are not run.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name == "rospy" for alias in node.names))]
    module = types.ModuleType("vilain.vilain")
    module.__file__ = str(repo / "vilain.py")
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    return module.ViLaIn


def main():
    load_credentials("/home/duo/.codex/keys/vlm-tamp.env")
    api = BoundedChat(ROOT / "results/external-component-api")
    native = load_native()

    class DirectViLaIn(native):
        def generate(self, content):
            response = api.create(model=self.model, messages=[{"role": "user", "content": content}])
            return response.choices[0].message.content

    detector_format = {"type": "json_schema", "json_schema": {
        "name": "NativeObjectBoxes", "strict": True, "schema": {
            "type": "object", "properties": {"detections": {"type": "array", "items": {
                "type": "object", "properties": {
                    "label": {"type": "string", "enum": ["cucumber", "carrot", "apple", "plate", "bowl"]},
                    "bbox_2d": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 640},
                                "minItems": 4, "maxItems": 4},
                }, "required": ["label", "bbox_2d"], "additionalProperties": False,
            }}}, "required": ["detections"], "additionalProperties": False,
        }}}

    def detect_create(**kwargs):
        # Reuse the first actual detector reply from the preserved setup trial.
        # The failure was the native empty-fixed-box guard, not model inference.
        from openai.types.chat import ChatCompletion
        cached_request = json.loads((ROOT / "results/external-component-api/request-00.json").read_text())
        assert cached_request["response_format"] == detector_format
        assert cached_request["messages"] == kwargs["messages"] and cached_request["model"] == kwargs["model"]
        cached = json.loads((ROOT / "results/external-component-api/response-00.json").read_text())
        return ChatCompletion.model_validate(cached)

    interpreter = DirectViLaIn("gpt-4o-2024-11-20")
    interpreter.detection_model = "gpt-4o-2024-11-20"
    interpreter.client_for_detection = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=detect_create)))
    utilities = sys.modules["vilain.vilain_utils"]
    original_rescale = utilities.rescale_bboxes
    utilities.rescale_bboxes = lambda boxes, size: original_rescale(boxes, size) if boxes else []
    folder = ROOT / "third_party/ViLaIn-TAMP/data/vilain_tamp_data/cooking"
    scene = folder / "moving"
    image_bytes = (scene / "observations/problem2.png").read_bytes()
    image = base64.b64encode(image_bytes).decode()
    domain = (folder / "domain.pddl").read_text()
    instruction = (scene / "instructions/problem2.txt").read_text().strip()
    output = ROOT / "results/vilain-component-smoke-fixed"
    if output.exists():
        raise FileExistsError("Preserve previous ViLaIn smoke result")
    output.mkdir(parents=True)
    result = {"instruction": instruction, "provided_scene_inventory": False,
              "provided_fixed_object_boxes": False,
              "adaptations": ["Removed unused ROS import", "Bypassed hard-coded private proxy with official API",
                              "Detector uses GPT-4o instead of local Qwen; native class ontology retained",
                              "Structured box-output format; native prompts and parsers retained",
                              "Empty provided-fixture list handled without injecting fixtures",
                              "First real detector reply cached from the preserved setup trial"],
              "scope": "Interpreter component smoke; not complete TAMP replication or LIBERO accuracy"}
    try:
        detected = interpreter.detect_objects(image, [], "cooking", (640, 640))
        result["detected_objects"] = detected
        initial = interpreter.generate_initial_state(domain, detected["result"], detected["bboxes"], image)
        result["initial_state"] = initial
        goals = interpreter.generate_goal_conditions(domain, detected["result"], initial["result"], instruction)
        result["goals"] = goals
        result["status"] = "component_calls_completed"
    except Exception as error:
        result.update(status="component_failure", error_type=type(error).__name__, error=str(error))
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "rgb.png").write_bytes(image_bytes)
    api.save_ledger()
    print(json.dumps({"status": result["status"], "calls": api.calls, "cost_usd": api.cost,
                      "generated_goals": (result.get("goals") or {}).get("result")}, indent=2))


if __name__ == "__main__":
    main()
