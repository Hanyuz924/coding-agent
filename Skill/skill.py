from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class SkillDefinition:
    name: str
    description: str
    trigger: list[str]          # ["/commit", "commit changes"] user type the cmd in the terminal
    allowed_tool: list[str]     # ["Bash", "Read"]  (allowed-tools)
    prompt_template: str         # prompt how to use this skill
    context: str = "inline"     # skill use context , inline or fork
    user_invocable: bool =True  # user can trigger this skill 
    arguments: list[str] = field(default_factory=list)
    when_to_use: str = ""        # when Claude should auto-invoke this skill
    source:str = "user"         # User defined or builtin skill

@dataclass
class FrontmatterResult:
    meta: dict[str, str] = field(default_factory=dict)
    body: str = ""

def _parse_skill_file_string(raw_string: str) -> FrontmatterResult | None:
    #invalid skill file
    if not raw_string.startswith("---"):
        return None
    parts = raw_string.split("---", 2)
    if len(parts) <3 :
        return None
    frontmatter_raw = parts[1].strip()
    meta: dict[str, str] ={}
    for line in frontmatter_raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        colon_index = line.find(":")
        key = line[:colon_index].strip().lower()
        value = line[colon_index +1:].strip()
        if key:
            meta[key] = value
    name = meta.get("name","")
    if not name:
        return None
    prompt = parts[2].strip()
    
    return FrontmatterResult(
        meta=meta,
        body=prompt
    )

def _parse_skill_tools(raw_string: str) -> list[str]:
    result = []
    raw_string = raw_string.strip()
    if raw_string == "":
        return result
    if raw_string.startswith("[") and raw_string.endswith("]"):
        raw_string = raw_string[1:-1]
    parts = raw_string.split(",")
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result
    

def _parse_skill_args(skill_prompt_template:str, args:str ="", named_args: list[str] | None = None) -> str:
    named_args = named_args or []
    if not args and not named_args:
        return skill_prompt_template
    ret = skill_prompt_template.replace("$ARGUMENTS", args)
    agrs_split = args.split()
    for i, arg_name in enumerate(named_args):
        placeholder = f"${arg_name.upper()}"
        value = agrs_split[i] if i < len(named_args) else ""
        ret = ret.replace(placeholder, value)
    return ret


def _parse_skill_file(
    file_path: Path, source:str
) -> SkillDefinition | None:
    try:
        raw_string = file_path.read_text()
        result = _parse_skill_file_string(raw_string)
        if not result:
            return None
        name = result.meta.get("name")
        user_invocable_raw = result.meta.get("user-invocable", "true")
        user_invocable = user_invocable_raw.lower() not in ("false", "0", "no")
        context = result.meta.get("context", "inline")

        allowed_tools_string = result.meta.get("allowed_tool", "")
        allowed_tool = _parse_skill_tools(allowed_tools_string)

        argument_string = result.meta.get("argument","")
        argument_list =_parse_skill_tools(argument_string)

        trigger_string = result.meta.get("triggers", "")
        tiggers = _parse_skill_tools(trigger_string) if trigger_string else [f"/{name}"]
        
        return SkillDefinition(
            name=name, # type: ignore
            description=result.meta.get("description", ""),
            trigger=tiggers,
            allowed_tool=allowed_tool,
            prompt_template=result.body,
            context=context,
            user_invocable=user_invocable,
            arguments=argument_list,
            when_to_use=result.meta.get("when_to_use",""),
            source=source
        )
    except Exception as e:
        raise e



        
