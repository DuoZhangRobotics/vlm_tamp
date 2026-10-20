"""Lossless S-expression parsing with explicit limits on goal-set evaluation."""

import re
from pathlib import Path


PREDICATE_ALIASES = {
    "On": "on",
    "on": "on",
    "In": "in",
    "in": "in",
    "Open": "open",
    "open": "open",
    "Close": "close",
    "close": "close",
    "Closed": "close",
    "closed": "close",
    "Turnon": "turnon",
    "TurnOn": "turnon",
    "turnon": "turnon",
    "Turnoff": "turnoff",
    "TurnOff": "turnoff",
    "turnoff": "turnoff",
    "At": "at",
    "at": "at",
    "Stack": "stack",
    "stack": "stack",
}
LOGICAL = {"and", "or", "not", "imply", "exists", "forall", "when"}


def normalize_predicate(name):
    return PREDICATE_ALIASES.get(name, name)


def parse_sexpr(text):
    text = re.sub(r";[^\n]*", "", text)
    tokens = re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text)
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unmatched closing parenthesis in BDDL")
            stack.pop()
        else:
            if not stack:
                raise ValueError("Unexpected atom outside BDDL expression")
            stack[-1].append(token)
    if stack or len(roots) != 1:
        raise ValueError("Unbalanced or multiple top-level BDDL expressions")
    return roots[0]


def typed_entities(tokens):
    result, pending = {}, []
    i = 0
    while i < len(tokens):
        if tokens[i] == "-":
            if not pending or i + 1 >= len(tokens):
                raise ValueError("Malformed typed entity declaration")
            for entity in pending:
                result[entity] = tokens[i + 1]
            pending = []
            i += 2
        else:
            pending.append(tokens[i])
            i += 1
    for entity in pending:
        result[entity] = "object"
    return result


def goal_atoms(expression):
    """Only positive conjunctions have unordered goal-set semantics."""
    if not isinstance(expression, list) or not expression:
        raise ValueError("Malformed goal expression")
    name = expression[0]
    if name.lower() == "and":
        atoms, unsupported = [], []
        for child in expression[1:]:
            a, u = goal_atoms(child)
            atoms.extend(a)
            unsupported.extend(u)
        return atoms, unsupported
    if name.lower() in LOGICAL or any(isinstance(x, list) for x in expression[1:]):
        return [], [expression]
    return [{"name": name, "arguments": expression[1:], "normalized_name": normalize_predicate(name)}], []


def parse_bddl(path):
    path = Path(path)
    text = path.read_text()
    tree = parse_sexpr(text)
    if not tree or tree[0].lower() != "define":
        raise ValueError("Expected BDDL define")
    sections = {
        node[0].lower(): node[1:]
        for node in tree[1:]
        if isinstance(node, list) and node and node[0].startswith(":")
    }
    if ":goal" not in sections or len(sections[":goal"]) != 1:
        raise ValueError("Expected exactly one goal expression")
    goal = sections[":goal"][0]
    atoms, unsupported = goal_atoms(goal)
    entities = []
    for section in (":fixtures", ":objects"):
        for identifier, category in typed_entities(sections.get(section, [])).items():
            entities.append(
                {"identifier": identifier, "category": category, "kind": "object", "parent": None}
            )
    for region in sections.get(":regions", []):
        target = next((x[1] for x in region[1:] if x[0] == ":target"), None)
        if target:
            entities.append(
                {
                    "identifier": f"{target}_{region[0]}",
                    "category": "region",
                    "kind": "region",
                    "parent": target,
                }
            )
    return {
        "instruction": " ".join(sections.get(":language", [])).strip('"'),
        "entities": entities,
        "goal_expression": goal,
        "goal_predicates": atoms,
        "unsupported_goal_expressions": unsupported,
        "raw_goal": goal,
        "source": str(path),
    }
