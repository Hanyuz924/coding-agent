from typing import Union

BASH_TOOL_NAME = "Bash"
GLOB_TOOL_NAME = "Glob"
GREP_TOOL_NAME = "Grep"
FILE_READ_TOOL_NAME = "Read"
FILE_EDIT_TOOL_NAME = "Edit"
FILE_WRITE_TOOL_NAME = "Write"
AGENT_TOOL_NAME = "Agent"
TODO_WRITE_TOOL_NAME = "TodoWrite"


DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000



def prepend_bullets(items: list[Union[str, list[str]]]) -> list[str]:
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(f"  - {subitem}" for subitem in item)
        else:
            result.append(f" - {item}")
    return result

def get_simple_prompt() -> str:
    tool_preference_items = [
        f"File search: Use {GLOB_TOOL_NAME} (NOT find or ls)",
        f"Content search: Use {GREP_TOOL_NAME} (NOT grep or rg)",
        f"Read files: Use {FILE_READ_TOOL_NAME} (NOT cat/head/tail)",
        f"Edit files: Use {FILE_EDIT_TOOL_NAME} (NOT sed/awk)",
        f"Write files: Use {FILE_WRITE_TOOL_NAME} (NOT echo >/cat <<EOF)",
        "Communication: Output text directly (NOT echo/printf)",
    ]
    avoid_commands = "`find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo`"

    multiple_commands_subitems = [
        f"If the commands are independent and can run in parallel, make multiple {BASH_TOOL_NAME} "
        f"tool calls in a single message. Example: if you need to run \"git status\" and \"git diff\", "
        f"send a single message with two {BASH_TOOL_NAME} tool calls in parallel.",
        f"If the commands depend on each other and must run sequentially, use a single {BASH_TOOL_NAME} "
        "call with '&&' to chain them together.",
        "Use ';' only when you need to run commands sequentially but don't care if earlier commands fail.",
        "DO NOT use newlines to separate commands (newlines are ok in quoted strings).",
    ]

    instruction_items: list[Union[str, list[str]]] = [
        "If your command will create new directories or files, first use this tool to run `ls` "
        "to verify the parent directory exists and is the correct location.",
        'Always quote file paths that contain spaces with double quotes in your command '
        '(e.g., cd "path with spaces/file.txt")',
        "Try to maintain your current working directory throughout the session by using absolute "
        "paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.",
        f"You may specify an optional timeout in milliseconds (up to {MAX_TIMEOUT_MS}ms / "
        f"{MAX_TIMEOUT_MS // 60_000} minutes). By default, your command will timeout after "
        f"{DEFAULT_TIMEOUT_MS}ms ({DEFAULT_TIMEOUT_MS // 60_000} minutes).",
        "When issuing multiple commands:",
        multiple_commands_subitems,
        "When using `find -regex` with alternation, put the longest alternative first. "
        "Example: use `'.*\\.\\(tsx\\|ts\\)'` not `'.*\\.\\(ts\\|tsx\\)'` — the second "
        "form silently skips `.tsx` files."
    ]

    lines = [
        "Executes a given bash command and returns its output.",
        "",
        "The working directory persists between commands, but shell state does not. "
        "The shell environment is initialized from the user's profile (bash or zsh).",
        "",
        f"IMPORTANT: Avoid using this tool to run {avoid_commands} commands, unless explicitly "
        "instructed or after you have verified that a dedicated tool cannot accomplish your task. "
        "Instead, use the appropriate dedicated tool as this will provide a much better experience "
        "for the user:",
        "",
        *prepend_bullets(tool_preference_items), # type: ignore
        f"While the {BASH_TOOL_NAME} tool can do similar things, it's better to use the built-in "
        "tools as they provide a better user experience and make it easier to review tool calls "
        "and give permission.",
        "",
        "# Instructions",
        *prepend_bullets(instruction_items),
    ]

    return "\n".join(lines)

if __name__ == "__main__":
    print(get_simple_prompt())
