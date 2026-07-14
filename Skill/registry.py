# stdlib
import logging
from pathlib import Path
from typing import AsyncGenerator

# local
from Agent.agent import query_loop
from Agent.types import AgentDefinition, AgentState, Message, AgentRunContext
from Skill.skill import SkillDefinition, _parse_skill_file, _parse_skill_args
from Agent.compact import get_messages_after_compact_boundary
from ForkAgent.fork_agent import create_fork_runcontext, build_cache_safe_params, run_fork_agent, ForkedAgentResult
from Model.anthropic_base import TextChunk, ThinkingChunk
logger = logging.getLogger("myagent.skill")

_SKILL_REGISTRY: dict[str, SkillDefinition] = {}
_TRIGGER_MAP: dict[str, SkillDefinition] = {}
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
    trigger = user_input.split()[0]
    return _TRIGGER_MAP.get(trigger)


def get_skill(name: str) -> SkillDefinition | None:
    """Look up a skill by its canonical name (used by the model-facing SkillTool)."""
    return _SKILL_REGISTRY.get(name)


def get_model_invocable_skills() -> list[SkillDefinition]:
    """Skills the model may invoke via the SkillTool.

    Mirrors CC's getSkillToolCommands filter: user-invocable skills are also the
    ones the model is allowed to pick. (CC additionally honours a
    disableModelInvocation flag; add one to SkillDefinition if you need it.)
    """
    return [s for s in _SKILL_REGISTRY.values() if s.user_invocable]


async def execute_skill(
    skill: SkillDefinition,
    state: AgentState,
    args: str,
    agent_def: AgentDefinition,
    system_prompt: list[Message],
    run_ctx:AgentRunContext,
    named_args: list[str] | None = None,
) -> AsyncGenerator:
    logger.info(f"Skill triggered: {skill.name}, arguments: {args if args else '(none)'}")
    skill_content = _parse_skill_args(skill.prompt_template, args, named_args)
    message = Message(role = "user", content = f"[Skill: {skill.name}]\n\n{skill_content}")
    if skill.context == "fork" :
        cache_safe_params = build_cache_safe_params(
            model=run_ctx.options.model,
            system_prompt=run_ctx.rendered_system_prompt, # type: ignore
            tools=run_ctx.options.tools,
            fork_context_messages=get_messages_after_compact_boundary(state.messages),
        )


    if skill.context == "fork":
        async for event in run_fork_agent(
            parent_ctx=run_ctx,
            agent_def=agent_def,
            cachesafeparams=cache_safe_params,
            user_message=message,
            agent_type=f"forked-skill-{skill.name}",
            disable_compact=True,
            max_turns=12
        ):
            if isinstance(event, (TextChunk, ThinkingChunk)):
                yield event
            elif isinstance(event, ForkedAgentResult):
                state.usage.accumulate(event.usage)
    else:
        async for event in query_loop(
            agent_def=agent_def,
            state=state,
            user_message=message,
            system_prompt=system_prompt,
            run_ctx=run_ctx # type: ignore
        ):
            yield event
