"""Restricted interpreter for the released ReKep stage-function protocol.

Model text is parsed as data. It is never passed to ``exec`` or ``eval``.
Only straight-line numerical functions with a finite scalar result are accepted.
"""

import ast
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


class ProgramError(ValueError):
    """The generated program cannot be safely evaluated or violates task policy."""


_CALLS = {
    "abs": abs, "max": max, "min": min, "float": float, "int": int, "len": len,
    "sum": sum,
    **{f"np.{name}": getattr(np, name) for name in (
        "array", "asarray", "abs", "maximum", "minimum", "clip", "mean", "sum", "dot", "cross",
        "sqrt", "sin", "cos", "arccos", "arcsin", "arctan2", "arctan", "exp", "log", "sign",
    )},
    "np.linalg.norm": np.linalg.norm,
}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
           ast.MatMult: operator.matmul}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_NAME = re.compile(r"stage([1-9]\d*)_(subgoal|path)_constraint([1-9]\d*)\Z")


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    raise ProgramError("Only named numerical functions may be called")


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ProgramError("Metadata must be literal data") from error


def _indices(node, count):
    """Check the keypoint dimension at parse time, including advanced indexing."""
    first = node.elts[0] if isinstance(node, ast.Tuple) else node
    if isinstance(first, ast.Slice):
        for part in (first.lower, first.upper, first.step):
            if part is not None and not isinstance(_literal(part), int):
                raise ProgramError("Keypoint slice limits must be integers")
        return
    value = _literal(first)
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values or any(type(index) is not int or not 0 <= index < count for index in values):
        raise ProgramError("Keypoint index outside the numbered image")


def _check_expr(node, names, count):
    if isinstance(node, ast.Constant):
        if node.value is not None and (type(node.value) not in (int, float, bool)
                                      or not np.isfinite(node.value)):
            raise ProgramError("Only finite numerical constants are allowed")
    elif isinstance(node, ast.Name):
        if node.id not in names:
            raise ProgramError("Undefined numerical variable: " + node.id)
    elif isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > 1000:
            raise ProgramError("Numerical literal is too large")
        for item in node.elts:
            _check_expr(item, names, count)
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        _check_expr(node.left, names, count)
        _check_expr(node.right, names, count)
        if isinstance(node.op, ast.Pow):
            exponent = _literal(node.right)
            if not isinstance(exponent, (int, float)) or abs(exponent) > 16:
                raise ProgramError("Power must have a bounded literal exponent")
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        _check_expr(node.operand, names, count)
    elif isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name not in _CALLS and name != "get_grasping_cost_by_keypoint_idx":
            raise ProgramError("Unsupported numerical function: " + name)
        if name == "get_grasping_cost_by_keypoint_idx":
            if len(node.args) != 1 or node.keywords:
                raise ProgramError("Grasp helper needs one literal keypoint index")
            _indices(node.args[0], count)
        for argument in node.args:
            _check_expr(argument, names, count)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg.startswith("_") or keyword.arg in ("out", "dtype", "like"):
                raise ProgramError("Mutating or dynamic numerical keyword")
            _check_expr(keyword.value, names, count)
    elif isinstance(node, ast.Attribute) and _dotted(node) in ("np.pi", "np.e"):
        pass
    elif isinstance(node, ast.Subscript):
        _check_expr(node.value, names, count)
        if isinstance(node.value, ast.Name) and node.value.id == "keypoints":
            _indices(node.slice, count)
        _check_slice(node.slice, names, count)
    else:
        raise ProgramError("Unsupported expression: " + type(node).__name__)


def _check_slice(node, names, count):
    if isinstance(node, ast.Slice):
        for part in (node.lower, node.upper, node.step):
            if part is not None:
                _check_expr(part, names, count)
    elif isinstance(node, ast.Tuple):
        for part in node.elts:
            _check_slice(part, names, count)
    else:
        _check_expr(node, names, count)


def _interpret(node, variables, attachment_cost):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return variables[node.id]
    if isinstance(node, ast.List):
        return [_interpret(item, variables, attachment_cost) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_interpret(item, variables, attachment_cost) for item in node.elts)
    if isinstance(node, ast.Slice):
        return slice(*[None if part is None else _interpret(part, variables, attachment_cost)
                       for part in (node.lower, node.upper, node.step)])
    if isinstance(node, ast.Attribute):
        return {"np.pi": np.pi, "np.e": np.e}[_dotted(node)]
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_interpret(node.left, variables, attachment_cost),
                                      _interpret(node.right, variables, attachment_cost))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_interpret(node.operand, variables, attachment_cost))
    if isinstance(node, ast.Subscript):
        return _interpret(node.value, variables, attachment_cost)[_interpret(node.slice, variables, attachment_cost)]
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        function = attachment_cost if name == "get_grasping_cost_by_keypoint_idx" else _CALLS[name]
        return function(*[_interpret(argument, variables, attachment_cost) for argument in node.args],
                        **{keyword.arg: _interpret(keyword.value, variables, attachment_cost)
                           for keyword in node.keywords})
    raise ProgramError("Unexpected node during interpretation")


class Constraint:
    def __init__(self, definition, count, attachment_cost):
        self.name = definition.name
        self.definition = definition
        self.count = count
        self.attachment_cost = attachment_cost
        self.source = ast.unparse(definition)
        self.keypoint_indices = sorted({int(_literal(node.args[0])) for node in ast.walk(definition)
                                       if isinstance(node, ast.Call)
                                       and _dotted(node.func) == "get_grasping_cost_by_keypoint_idx"})
        if (definition.decorator_list or definition.returns or definition.args.defaults
                or definition.args.kw_defaults or definition.args.vararg or definition.args.kwarg
                or definition.args.kwonlyargs or definition.args.posonlyargs
                or [argument.arg for argument in definition.args.args] != ["end_effector", "keypoints"]
                or any(argument.annotation for argument in definition.args.args)):
            raise ProgramError("Constraint signature must be (end_effector, keypoints)")
        names = {"end_effector", "keypoints"}
        returned = False
        for statement in definition.body:
            if returned:
                raise ProgramError("Statements after return are unsupported")
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                target = statement.targets[0].id
                if target.startswith("_") or target in names & {"end_effector", "keypoints"} or target in _CALLS or target == "np":
                    raise ProgramError("Protected numerical variable")
                _check_expr(statement.value, names, count)
                names.add(target)
            elif isinstance(statement, ast.Return) and statement.value is not None:
                _check_expr(statement.value, names, count)
                returned = True
            else:
                raise ProgramError("Only local numerical assignments and return are permitted")
        if not returned:
            raise ProgramError("Constraint has no scalar return")

    @property
    def attachment_only(self):
        meaningful = [item for item in self.definition.body if not (
            isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str))]
        return (len(meaningful) == 1 and isinstance(meaningful[0], ast.Return)
                and isinstance(meaningful[0].value, ast.Call)
                and _dotted(meaningful[0].value.func) == "get_grasping_cost_by_keypoint_idx")

    def __call__(self, end_effector, keypoints):
        ee, points = np.asarray(end_effector, dtype=float), np.asarray(keypoints, dtype=float)
        if ee.shape != (3,) or points.shape != (self.count, 3) or not np.isfinite(ee).all() or not np.isfinite(points).all():
            raise ProgramError("Constraint inputs must be finite sensor-backed XYZ arrays")
        variables = {"end_effector": ee, "keypoints": points}
        try:
            with np.errstate(all="raise"):
                for statement in self.definition.body:
                    if isinstance(statement, ast.Assign):
                        variables[statement.targets[0].id] = _interpret(statement.value, variables, self.attachment_cost)
                    elif isinstance(statement, ast.Return):
                        result = np.asarray(_interpret(statement.value, variables, self.attachment_cost))
                        if result.shape != () or result.dtype.kind not in "biuf" or not np.isfinite(result):
                            raise ProgramError("Constraint must return one finite scalar")
                        return float(result)
        except (ArithmeticError, IndexError, TypeError, ValueError, KeyError) as error:
            if isinstance(error, ProgramError):
                raise
            raise ProgramError("Numerical constraint evaluation failed: " + self.name) from error
        raise ProgramError("Constraint did not return")


@dataclass
class Stage:
    index: int
    grasp_keypoint: int
    release_keypoint: int
    subgoal_constraints: list[Constraint]
    path_constraints: list[Constraint]


@dataclass
class Program:
    num_stages: int
    grasp_keypoints: list[int]
    release_keypoints: list[int]
    stages: list[Stage]
    source: str
    num_keypoints: int
    task_type: str

    def set_attachment_cost(self, function: Callable[[int], float]):
        for stage in self.stages:
            for constraint in stage.subgoal_constraints + stage.path_constraints:
                constraint.attachment_cost = function

    def validate_values(self, end_effector, keypoints):
        for stage in self.stages:
            for constraint in stage.subgoal_constraints + stage.path_constraints:
                constraint(end_effector, keypoints)

    def metadata(self):
        return {"num_stages": self.num_stages, "grasp_keypoints": self.grasp_keypoints,
                "release_keypoints": self.release_keypoints, "num_keypoints": self.num_keypoints,
                "task_type": self.task_type, "execution": "restricted_ast_interpreter",
                "functions": [constraint.name for stage in self.stages
                              for constraint in stage.subgoal_constraints + stage.path_constraints]}


def parse_program(text, num_keypoints, task_type, attachment_cost=None):
    if type(num_keypoints) is not int or not 1 <= num_keypoints <= 1000:
        raise ProgramError("A nonempty numbered keypoint image is required")
    if task_type not in ("pick_place", "open_middle_drawer"):
        raise ProgramError("Unknown experiment task policy")
    if len(text) > 64000:
        raise ProgramError("Program exceeds the numerical protocol size bound")
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.DOTALL)
    if len(blocks) != 1:
        raise ProgramError("Expected exactly one Python code block")
    source = blocks[0]
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as error:
        raise ProgramError("Malformed Python stage program") from error
    if sum(1 for _ in ast.walk(tree)) > 4000:
        raise ProgramError("Program has too many numerical expressions")
    metadata, definitions = {}, []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "numpy" and node.names[0].asname == "np":
            continue  # Declaration only: the interpreter supplies whitelisted NumPy calls.
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name not in ("num_stages", "grasp_keypoints", "release_keypoints") or name in metadata:
                raise ProgramError("Unexpected or repeated top-level metadata")
            metadata[name] = _literal(node.value)
        elif isinstance(node, ast.FunctionDef) and _NAME.fullmatch(node.name):
            definitions.append(node)
        else:
            raise ProgramError("Unsupported top-level code")
    count = metadata.get("num_stages")
    if type(count) is not int or not 1 <= count <= 12 or len(definitions) > 64:
        raise ProgramError("Invalid stage count")
    for name in ("grasp_keypoints", "release_keypoints"):
        values = metadata.get(name)
        if (not isinstance(values, list) or len(values) != count
                or any(type(value) is not int or not -1 <= value < num_keypoints for value in values)):
            raise ProgramError("Invalid " + name)
    attachment_cost = attachment_cost or (lambda index: 1.0)
    stages = [Stage(index + 1, metadata["grasp_keypoints"][index], metadata["release_keypoints"][index], [], [])
              for index in range(count)]
    seen = set()
    for definition in definitions:
        stage, kind, _ = _NAME.fullmatch(definition.name).groups()
        stage = int(stage)
        if stage > count or definition.name in seen:
            raise ProgramError("Repeated function or nonexistent stage")
        seen.add(definition.name)
        constraint = Constraint(definition, num_keypoints, attachment_cost)
        getattr(stages[stage - 1], kind + "_constraints").append(constraint)
    held = -1
    for stage in stages:
        if stage.grasp_keypoint >= 0:
            if held >= 0 or len(stage.subgoal_constraints) != 1 or stage.path_constraints or stage.release_keypoint >= 0:
                raise ProgramError("Grasp must be an independent stage for one free gripper")
            held = stage.grasp_keypoint
        for constraint in stage.path_constraints:
            if held < 0 or any(index != held for index in constraint.keypoint_indices):
                raise ProgramError("Path constraint requires an earlier matching grasp")
            if task_type == "pick_place" and not constraint.attachment_only:
                raise ProgramError("Ordinary pick/place excludes additional geometric path constraints")
        if stage.release_keypoint >= 0:
            if held != stage.release_keypoint:
                raise ProgramError("Release keypoint was not grasped in an earlier stage")
            held = -1
    if not any(stage.grasp_keypoint >= 0 for stage in stages):
        raise ProgramError("Manipulation program needs a grasp stage")
    return Program(count, metadata["grasp_keypoints"], metadata["release_keypoints"], stages,
                   source, num_keypoints, task_type)


def build_prompt(instruction, task_type, template_path):
    native = Path(template_path).read_text().format(instruction=instruction)
    policy = (
        "Ordinary pick-and-place: do not add geometric path constraints (upright, orientation, axis, line, or plane). "
        "Only the supplied grasp-retention cost is permitted as a path constraint."
        if task_type == "pick_place" else
        "Open the middle drawer: keep the grasp stage separate. Apply contact retention and a drawer pull-line "
        "constraint only in the pulling stage, before release. Infer the line direction from visible landmarks; "
        "do not assume a provided joint axis or simulator identity."
    )
    return native + "\n\n## Adaptation policy\n" + policy + (
        "\nUse only straight-line numerical assignments and returns. Use np.array, np.asarray, np.linalg.norm, "
        "np.dot, np.cross, np.mean, np.sum, np.abs, np.maximum, np.minimum, np.clip, np.sqrt, np.sin, np.cos, "
        "np.arccos, np.arcsin, np.arctan2, np.arctan, np.sign, np.pi, or abs/min/max/float. "
        "Use literal keypoint indices and bounded numerical constants. Do not import modules, read files, "
        "use loops, branches, comprehensions, dynamic attributes, or external resources. "
        "Return a single finite scalar cost from each function. The image and instruction are your scene evidence."
    )


def generate_program(numbered_path, instruction, task_type, num_keypoints, outdir, config, attachment_cost=None):
    """Generate once or replay a durable cached answer; never repair its semantics."""
    import json
    from .api import request
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    template = config.get("prompt_template") or Path(__file__).resolve().parents[3] / "third_party/ReKep/vlm_query/prompt_template.txt"
    prompt = build_prompt(instruction, task_type, template)
    (outdir / "prompt.txt").write_text(prompt)
    raw = request(numbered_path, prompt, outdir, config)
    text = "".join(item["text"] for block in raw.get("output", []) if block.get("type") == "message"
                   for item in block.get("content", []) if item.get("type") == "output_text")
    (outdir / "output_raw.txt").write_text(text)
    if raw.get("status") != "completed":
        raise ProgramError("API response is incomplete; no retry or substitute program")
    program = parse_program(text, num_keypoints, task_type, attachment_cost)
    (outdir / "metadata.json").write_text(json.dumps(program.metadata(), indent=2) + "\n")
    for stage in program.stages:
        for kind in ("subgoal", "path"):
            constraints = getattr(stage, kind + "_constraints")
            (outdir / f"stage{stage.index}_{kind}_constraints.txt").write_text(
                "\n\n".join(constraint.source for constraint in constraints) + "\n")
    return program


generate = generate_program
