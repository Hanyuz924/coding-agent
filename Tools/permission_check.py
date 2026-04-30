from __future__ import annotations
import json
import re
from pathlib import Path


BASH_TOOL_NAME = "Bash"
GLOB_TOOL_NAME = "Glob"
GREP_TOOL_NAME = "Grep"
FILE_READ_TOOL_NAME = "Read"
FILE_EDIT_TOOL_NAME = "Edit"
FILE_WRITE_TOOL_NAME = "Write"
AGENT_TOOL_NAME = "Agent"
TODO_WRITE_TOOL_NAME = "TodoWrite"


READ_TOOLS = [FILE_READ_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME]
EDIT_TOOLS = [FILE_WRITE_TOOL_NAME, FILE_EDIT_TOOL_NAME]


DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bshutdown\b"),
]

def is_dangerous(cmd: str):
    return any(pattern.search(cmd) for pattern in DANGEROUS_PATTERNS)


def _parse_rule(rule: str) -> dict:
    m = re.match(r"^([a-zA-Z_]+)\((.+)\)$", rule)
    if m:
        return {"tool": m.group(1), "pattern": m.group(2)}
    return {"tool": rule, "pattern": None}



def _load_settings(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return None


_cached_rules: dict | None = None


def load_permission_rules() -> dict:
    global _cached_rules
    if _cached_rules is not None:
        return _cached_rules

    allow: list[dict] = []
    deny: list[dict] = []

    user_settings = _load_settings(Path.home() / ".claude" / "settings.json")
    project_settings = _load_settings(Path.cwd() / ".claude" / "settings.json")

    for settings in [user_settings, project_settings]:
        if not settings or "permissions" not in settings:
            continue
        perms = settings["permissions"]
        for r in perms.get("allow", []):
            allow.append(_parse_rule(r))
        for r in perms.get("deny", []):
            deny.append(_parse_rule(r))

    _cached_rules = {"allow": allow, "deny": deny}
    return _cached_rules

def _match_rule(rule:dict, tool_name:str, input: dict) -> bool:
    if rule["tool"] != tool_name:
        return False
    #for this tool, all the pattern are ok or prohibited !
    if rule["pattern"] is None:
        return True
    
    value = ""
    if tool_name == BASH_TOOL_NAME:
        value = input.get("command", "")
    elif "file_path" in input:
        value = input.get("file_path")
    else:
        return True
    pattern = rule["pattern"]
    if pattern.endswith("*"):
        return value == pattern[:-1]
    return value == pattern

def _check_permission_rules(tool_name: str, inp: dict) -> str | None:
    rules = load_permission_rules()
    for rule in rules["deny"]:
        if _match_rule(rule, tool_name, inp):
            return "deny"
    for rule in rules["allow"]:
        if _match_rule(rule, tool_name, inp):
            return "allow"
    return None

def check_permission(
        tool_name: str, 
        input: dict, 
        mode:str ="default"
    )-> dict:
    """Returns {"action": "allow"|"deny"|"confirm", "message": ...}"""
    if mode == "bypassPermissions":
        return {"action": "allow"}

    rule_result = _check_permission_rules(tool_name, input)
    if rule_result == "deny":
        return {"action": "deny", "message": f"Denied by permission rule for {tool_name}"}
    if rule_result == "allow":
        return {"action": "allow"}

    if tool_name in READ_TOOLS:
        return {"action": "allow"}
    
    needs_confirm = False
    if tool_name == BASH_TOOL_NAME and is_dangerous(input.get("command", "")):
        needs_confirm = True
        confirm_message = input.get("command", "")
    elif tool_name == FILE_WRITE_TOOL_NAME and not Path(input.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"write new file: {input.get('file_path', '')}"
    elif tool_name == FILE_EDIT_TOOL_NAME and not Path(input.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"edit non-existent file: {input.get('file_path', '')}"

    if needs_confirm:
        # if mode == "dontAsk":
        #     return {"action": "deny", "message": f"Auto-denied (dontAsk mode): {confirm_message}"}
        return {"action": "confirm", "message": confirm_message}

    return {"action": "allow"}
    



