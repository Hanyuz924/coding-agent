from Skill.skill import SkillDefinition, _parse_skill_file, _parse_skill_args
from Agent.agent import AgentState, BaseAgent
from typing import Generator
from pathlib import Path

_SKILL_REGISTRY: dict[str, SkillDefinition] = {}
_TRIGGER_MAP: dict[str, SkillDefinition] = {}  # "/commit" -> skill
BUILTIN_SKILL_DIR = Path(__file__).parent / "builtin"
USER_SKILL_DIRS = [
    Path.cwd() / ".claude" / "skills",
    Path.home() / ".claude" / "skills",
]


def _register(skill: SkillDefinition) -> None:
    _SKILL_REGISTRY[skill.name] = skill
    for trigger in skill.trigger:
        _TRIGGER_MAP[trigger] = skill


def load_builtin_skills() -> None:
    for path in BUILTIN_SKILL_DIR.glob("*.md"):
        skill = _parse_skill_file(path, source="builtin")
        if skill:
            _register(skill)


def load_user_skills() -> None:
    for skill_dir in USER_SKILL_DIRS:
        if skill_dir.exists():
            for path in skill_dir.glob("*.md"):
                skill = _parse_skill_file(path, source="user")
                if skill:
                    _register(skill)


def load_all_skills() -> None:
    load_builtin_skills()
    load_user_skills()


def skill_dispatch(user_input: str) -> SkillDefinition | None:
    trigger = user_input.split()[0]  # e.g. "/commit"
    return _TRIGGER_MAP.get(trigger)


def execute_skill(
    skill: SkillDefinition,
    state: AgentState,
    args: str,
    agent_instance: BaseAgent,
    system_prompt: str,
    named_args: list[str] | None = None,
) -> Generator:
    skill_content = _parse_skill_args(skill.prompt_template, args, named_args)
    message = f"[Skill: {skill.name}]\n\n{skill_content}"

    if skill.context == "fork":
        new_state = AgentState()
        yield from agent_instance.queryLoop(
            user_message=message,
            system_prompt=system_prompt,
            state=new_state
        )
    else:
        yield from agent_instance.queryLoop(
            user_message=message,
            system_prompt=system_prompt,
            state=state
        )
