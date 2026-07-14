# stdlib
import argparse
import asyncio
import os
import readline
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure readline for better line editing with history
_HISTORY_FILE = Path.home() / ".mini_claude_history"
readline.set_completer_delims(" \t\n")
if _HISTORY_FILE.exists():
    readline.read_history_file(_HISTORY_FILE)
readline.set_history_length(1000)

# third-party
from dotenv import load_dotenv

# local
from Agent import system_prompt
from Agent.agent import TextChunk, ThinkingChunk, create_agent_definition, create_agent_runcontext, estimate_tokens_by_api, query_loop
from Agent.events import MaxTurnsReached, PermissionRequest, ToolEnd, ToolStart, TurnDone
from Agent.types import AgentDefinition, AgentState, Message
from Agent.compact import CompactResult, get_messages_after_compact_boundary
from ForkAgent.fork_agent import ForkedAgentResult
from Skill.registry import execute_skill, load_all_skills, skill_dispatch
from Tools.registry import register_skill_tool
from utils import utils

_BUILTIN_CMDS = {"/clear", "/cost", "/context_tokens", "/print_message"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mini-claude", add_help=False)
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--help", "-h", action="store_true")
    return parser.parse_args()


async def _handle_builtin(
    cmd: str,
    state: AgentState,
    agent_def: AgentDefinition,
    system_prompt_str: list[Message],
) -> AgentState:
    if cmd == "/clear":
        print("[conversation cleared]")
        return AgentState()
    if cmd == "/cost":
        print(f"Input tokens: {state.usage.input_tokens}  Output tokens: {state.usage.output_tokens}")
    elif cmd == "/context_tokens":
        api_tokens = await estimate_tokens_by_api(agent_def, system_prompt_str, get_messages_after_compact_boundary(state.messages))
        print(f"Current message list token count: {api_tokens}")
    elif cmd == "/print_message":
        print(state.messages)
    return state

async def async_input(prompt: str = "") -> str:
    loop = asyncio.get_running_loop()
    # input() runs in a thread; the await yields to the event loop,
    # so background tasks (extraction) run while waiting for the user
    return await loop.run_in_executor(None, input, prompt)

async def _handle_events(gen) -> None:
    async for event in gen:
        if isinstance(event, TextChunk):
            print(event.text, end="", flush=True)
        elif isinstance(event, ThinkingChunk):
            print(f"\033[2m[thinking] {event.text}\033[0m", end="", flush=True)
        elif isinstance(event, ToolStart):
            print(f"\n\033[33m[tool] {event.name} {event.inputs}\033[0m")
        elif isinstance(event, ToolEnd):
            status = "" if event.permitted else " (denied)"
            print(f"\033[33m[done] {event.name}{status}\033[0m")
        elif isinstance(event, TurnDone):
            cache_hit = event.cache_read_tokens > 0
            cache_info = f" cache={'hit' if cache_hit else 'miss'} read={event.cache_read_tokens} create={event.cache_creation_tokens}"
            print(f"\n\033[2m[tokens: in={event.input_tokens} out={event.output_tokens} stop={event.turn_done_reason}{cache_info}]\033[0m")
        elif isinstance(event, PermissionRequest):
            print(f"\033[33m[permission] :\n{event.description}\033[0m")
            answer = input("Allow? [y/n]: ").strip().lower()
            if answer == "y":
                event.granted = True
        elif isinstance(event, MaxTurnsReached):
            answer = input(f"\n[max turns ({event.max_turns}) reached] Continue? [y/n]: ").strip().lower()
            if answer == "y":
                event.continue_turn = True
        elif isinstance(event, CompactResult):
            status = "ok" if event.success else "failed"
            cache_hit = event.cache_read_tokens > 0
            cache_info = f" cache={'hit' if cache_hit else 'miss'} read={event.cache_read_tokens} create={event.cache_creation_tokens}"
            print(f"\n\033[31m[compact] {status} path={event.path}{cache_info}\033[0m")
        elif isinstance(event, ForkedAgentResult):
            print(f"\n\033[2m[fork] done in={event.usage.input_tokens} out={event.usage.output_tokens}\033[0m")


async def repl(agent_def: AgentDefinition, system_prompt_str: list[Message]) -> None:
    state = AgentState()
    print("Mini Claude Code — type 'exit' or Ctrl-C to quit, /clear to reset.\n")

    while True:
        run_ctx = create_agent_runcontext(
            model_name=agent_def.model.model_name,
            filecachestate=state.filecachestate,
            tool_schemas=agent_def.tool_schemas,
            parent_system_prompt=system_prompt_str,
        )
        try:
            user_input  = (await async_input("\033[1;32m>\033[0m ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break

        cmd = user_input.split()[0]

        if cmd in _BUILTIN_CMDS:
            state = await _handle_builtin(cmd, state, agent_def, system_prompt_str)
            continue

        print()
        try:
            # Dispatch mirrors Claude Code (REPL.tsx:3161): a leading '/' is the
            # ONLY thing that triggers a command/skill lookup. Free text always
            # goes to the model — it selects skills itself via the SkillTool.
            # We never match free-text intent to a skill in the REPL.
            if user_input.startswith("/"):
                skill = skill_dispatch(user_input)
                if skill is None:
                    print(f"\033[31mUnknown command: {cmd}\033[0m")
                    continue
                args = " ".join(user_input.split()[1:])
                gen = execute_skill(skill, state, args, agent_def, system_prompt_str,run_ctx)
            else:
                user_message = Message(role="user", content=user_input)
                gen = query_loop(agent_def, state, user_message, system_prompt_str, run_ctx)
            await _handle_events(gen)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as e:
            print(f"\n\033[31mError: {e}\033[0m")

        print()


async def _async_main() -> None:
    load_dotenv()
    args = parse_args()
    log_level = os.environ.get("LOG_LEVEL", "warning")
    utils.setup_logger(console_log=False, log_level=log_level)
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = Path(__file__).parent.parent / "logs" / f"agent_{log_time}.log"
    utils.add_file_handler(log_file_name)
    if args.help:
        print("Usage: mini-claude [prompt] [--model MODEL]")
        sys.exit(0)

    model_name = args.model or os.environ.get("MINI_CLAUDE_MODEL", "claude-sonnet-4-6")
    langfuse_use = bool(os.environ.get("LANGFUSE_USE", "false"))
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    load_all_skills()
    register_skill_tool()  # after skills load, so the Skill tool schema lists them
    agent_def = create_agent_definition(
        model_name=model_name,
        api_key=api_key,
        langfuse_use=langfuse_use,
    )
    system_prompt_str = [Message(role="system", content=system_prompt.build_system_prompt())]

    # if args.prompt:
    #     state = AgentState()
    #     user_input = " ".join(args.prompt)
    #     async for event in query_loop(agent_def, state, user_input, system_prompt_str):
    #         if isinstance(event, TextChunk):
    #             print(event.text, end="", flush=True)
    #     print()
    # else:
    try:
        await repl(agent_def, system_prompt_str)
    finally:
        readline.write_history_file(_HISTORY_FILE)
        if agent_def.langfuse is not None:
            agent_def.langfuse.shutdown()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
