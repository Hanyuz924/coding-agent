import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from Agent import agent, system_prompt
from Agent.agent import AgentState, TextChunk, ThinkingChunk, ToolStart, ToolEnd, TurnDone,PermissionRequest
from Model import anthropic_base
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from Tools.registry import get_tool_schemas
from Skill.registry import load_all_skills, skill_dispatch, execute_skill

_BUILTIN_CMDS = {"/clear", "/cost", "/context_tokens", "/print_message"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mini-claude", add_help=False)
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--help", "-h", action="store_true")
    return parser.parse_args()


def create_agent_model(model_name: str, api_key: str) -> anthropic_base.AnthropicModelClass:
    return anthropic_base.AnthropicModelClass(model_name=model_name, api_key=api_key)


def _handle_builtin(cmd: str, state: AgentState, agent_instance: agent.BaseAgent, system_prompt_str: str) -> AgentState:
    if cmd == "/clear":
        print("[conversation cleared]")
        return AgentState()
    if cmd == "/cost":
        print(f"Input tokens: {state.total_input_tokens}  Output tokens: {state.total_output_tokens}")
    elif cmd == "/context_tokens":
        api_tokens = agent_instance._estimate_tokens_by_api(system_prompt_str, state.messages, get_tool_schemas())
        print(f"Current message list token count: {api_tokens}")
    elif cmd == "/print_message":
        print(state.messages)
    return state


def _handle_events(gen) -> None:
    for event in gen:
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
            print(f"\n\033[2m[tokens: in={event.input_tokens} out={event.output_tokens} stop={event.turn_done_reason}]\033[0m")
        elif isinstance(event, PermissionRequest):
            print(f"\033[33m[permission] :\n{event.description}\033[0m")
            answer = input("Allow? [y/n]: ").strip().lower()
            if answer == "y":
                event.granted = True


def repl(agent_instance: agent.BaseAgent, system_prompt_str: str) -> None:
    state = AgentState()
    print("Mini Claude Code — type 'exit' or Ctrl-C to quit, /clear to reset.\n")

    while True:
        try:
            user_input = input("\033[1;32m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break

        cmd = user_input.split()[0] if user_input.split() else ""

        if cmd in _BUILTIN_CMDS:
            state = _handle_builtin(cmd, state, agent_instance, system_prompt_str)
            continue

        print()
        try:
            skill = skill_dispatch(user_input)
            if skill:
                args = " ".join(user_input.split()[1:])
                gen = execute_skill(skill, state, args, agent_instance, system_prompt_str)
            else:
                gen = agent_instance.queryLoop(user_input, system_prompt_str, state)
            _handle_events(gen)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as e:
            print(f"\n\033[31mError: {e}\033[0m")

        print()


def main() -> None:
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    load_all_skills()
    model = create_agent_model(model_name, api_key)
    agent_instance = agent.BaseAgent(model=model)
    system_prompt_str = system_prompt.build_system_prompt()

    if args.prompt:
        # single-shot mode
        state = AgentState()
        user_input = " ".join(args.prompt)
        for event in agent_instance.queryLoop(
            user_message=user_input,
            system_prompt=system_prompt_str,
            state=state,
        ):
            if isinstance(event, TextChunk):
                print(event.text, end="", flush=True)
        print()
    else:
        repl(agent_instance, system_prompt_str)


if __name__ == "__main__":
    main()
