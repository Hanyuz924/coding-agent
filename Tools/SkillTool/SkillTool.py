"""Model-invocable skill tool — the second skill entrypoint (Claude Code parity).

Claude Code exposes skills two ways (see SkillTool/SkillTool.ts + getSkillToolCommands):
  1. The user types `/name`               -> handled in the REPL slash dispatch.
  2. The MODEL picks a skill mid-query    -> that is THIS tool.

Since the REPL now sends all free text straight to the model, this tool is how a
skill actually gets selected from natural-language intent. It does NOT spawn a
sub-agent: for an `inline` skill it simply returns the skill's expanded prompt as
the tool result, and the model carries out those instructions on the next turn.
"""

# stdlib
from typing import Any, Dict

# local
from Tools.BaseTool import AgentRunContext, BaseTool, ToolResult, logger
from Skill.registry import get_model_invocable_skills, get_skill
from Skill.skill import _parse_skill_args

SKILL_TOOL_NAME = "Skill"

_BASE_DESCRIPTION = """Invoke a specialized skill by name.

A skill is a reusable, curated procedure. When the user's request matches a
skill's purpose, call this tool with the skill's exact name; the skill's
instructions are returned to you, and you then carry them out. Prefer invoking a
matching skill over improvising the same procedure yourself. Only invoke a skill
that appears in the list below."""


class SkillTool(BaseTool):
    """Lets the model invoke a registered skill. Schema is built dynamically, so
    the listed skills always reflect whatever ``load_all_skills()`` registered."""

    @property
    def name(self) -> str:
        return SKILL_TOOL_NAME

    @property
    def description(self) -> str:
        skills = get_model_invocable_skills()
        if not skills:
            return f"{_BASE_DESCRIPTION}\n\n(No skills are currently available.)"
        lines = [f"{_BASE_DESCRIPTION}\n\nAvailable skills:"]
        for s in skills:
            guidance = s.when_to_use or s.description or "(no description)"
            lines.append(f"- {s.name}: {guidance}")
        return "\n".join(lines)

    @property
    def input_schema(self) -> Dict[str, Any]:
        names = [s.name for s in get_model_invocable_skills()]
        skill_name_schema: Dict[str, Any] = {
            "type": "string",
            "description": "The exact name of the skill to invoke.",
        }
        # Constrain to known skills so the model can't hallucinate a skill name.
        if names:
            skill_name_schema["enum"] = names
        return {
            "type": "object",
            "properties": {
                "skill_name": skill_name_schema,
                "arguments": {
                    "type": "string",
                    "description": "Optional free-text arguments/context passed to "
                    "the skill (substituted for $ARGUMENTS in its template).",
                },
            },
            "required": ["skill_name"],
        }

    @property
    def read_only(self) -> bool:
        # Expanding a skill has no side effects of its own; the model acts on the
        # returned instructions afterward (and those actions are permission-gated).
        return True

    def execute(self, ctx: AgentRunContext, **kwargs) -> ToolResult:
        skill_name = kwargs.get("skill_name", "")
        arguments = kwargs.get("arguments", "") or ""

        skill = get_skill(skill_name)
        if skill is None:
            return ToolResult(success=False, error=f"Unknown skill: {skill_name!r}")
        if not skill.user_invocable:
            return ToolResult(
                success=False, error=f"Skill {skill_name!r} is not invocable."
            )

        content = _parse_skill_args(skill.prompt_template, arguments)
        logger.info(f"[Skill] model invoked skill={skill_name!r} args={arguments!r}")

        # Inline expansion (CC 'inline' context): hand the skill's instructions
        # back as the tool result; the model follows them on the next turn.
        # NOTE: skills with context='fork' are meant to run as an isolated
        # sub-agent with their own token budget. That needs the fork path, not a
        # synchronous tool call, so they still expand inline here. Wire fork
        # skills through run_forked_agent if/when you need true isolation.
        return ToolResult(
            success=True,
            data=content,
            metadata={"skill": skill_name, "context": skill.context},
        )
