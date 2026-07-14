# stdlib
import os
import shutil
import subprocess
from typing import Any, Dict, Optional

# third-party
from pydantic import BaseModel, Field

# local
from Tools.BaseTool import BaseTool, ToolResult, AgentRunContext, logger
from Tools.BashTool.bash_prompt import DEFAULT_TIMEOUT_MS


MAX_OUTPUT_CHARS   = 20_000

"""
For current implementation the ShellSession will just live in one round 
But later on consider to add multi-rounds ShellSession.
With this bash cmd become statefull,
So we can avoid LLM to use long cmd that chain with && 
"""

# class ShellSession:
#     def __init__(self, session_id: str) -> None:
#         self.session_id = session_id
#         self.cwd = os.getcwd()
#         self.env = os.environ
#         #used to track the cmd history and cmd change history 
#         self.dir_history = []
#         self.cmd_history = []


class BashToolInput(BaseModel):
    command: str = Field(
        description="The command to execute"
    )
    timeout: int = Field(
        default=DEFAULT_TIMEOUT_MS,
        description="Seconds before timeout"
    )
    
    description: Optional[str] = Field(
        default=None,
        description= """\
            Clear, concise description of what this command does in active voice
        """
    )

class BashTool(BaseTool):
    @property 
    def name(self) -> str:
        return "Bash"
    
    @property
    def description(self) -> str:
        return "Execute a shell command. Returns stdout+stderr. Stateless (no cd persistence)."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type":"object",
            "required": ["command"],
            "properties":{
                "command":{
                    "type":"string",
                    "description":"The command to execute"
                },
                "timeout":{
                    "type": "integer",
                    "description":"Optional timeout in milliseconds before timeout. Use 120-300 for package installs (npm, pip, npx), builds, and long-running commands."
                },
                "description":{
                    "type": "string",
                    "description":"""\
                                Clear, concise description of what this command does in active voice. Never use words like "complex" or "risk" in the description - just describe what it does.

                                For simple commands (git, npm, standard CLI tools), keep it brief (5-10 words):
                                - ls → "List files in current directory"
                                - git status → "Show working tree status"
                                - npm install → "Install package dependencies"

                                For commands that are harder to parse at a glance (piped commands, obscure flags, etc.), add enough context to clarify what it does:
                                - find . -name "*.tmp" -exec rm {} \\; → "Find and delete all .tmp files recursively"
                                - git reset --hard origin/main → "Discard all local changes and match remote main"
                                - curl -s url | jq '.data[]' → "Fetch JSON from URL and extract data array elements"\
                            """
                }
                
            }
        }
    @property
    def read_only(self) -> bool:
        return False
    @property
    def concurrent_safe(self) -> bool:
        return False

    def _get_shell_path(self) -> str:
        shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
        return shell

    def execute(self, ctx: AgentRunContext, **kwargs) -> ToolResult:
        input = BashToolInput(**kwargs)
        shell_path = self._get_shell_path()
        logger.debug(f"[{self.name}] command={input.command!r}, shell={shell_path}, timeout={input.timeout}ms")
        kwargs = dict(
            shell=True,
            executable=shell_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.getcwd(),
        )
        try:
            proc = subprocess.Popen(input.command, **kwargs)
            try:
                stdout, stderr = proc.communicate(timeout=input.timeout / 1000)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                logger.debug(f"[{self.name}] timed out after {input.timeout/1000:.1f}s")
                return ToolResult(
                    success=False,
                    error=f"Timed out after {input.timeout/1000:.1f}s and process was killed.",
                    metadata={"returncode": -1, "timed_out": True},
                )

            out = stdout
            if stderr:
                out += "\n[stderr]\n" + stderr
            out = out.strip()

            truncated = False
            if len(out) > MAX_OUTPUT_CHARS:
                out = out[:MAX_OUTPUT_CHARS]
                truncated = True

            logger.debug(f"[{self.name}] returncode={proc.returncode}, output_chars={len(out)}, truncated={truncated}")
            return ToolResult(
                success=proc.returncode == 0,
                data=out,
                error="" if proc.returncode == 0 else f"Command exited with code {proc.returncode}",
                metadata={"returncode": proc.returncode, "truncated": truncated},
            )

        except FileNotFoundError as e:
            logger.debug(f"[{self.name}] executable not found: {e}")
            return ToolResult(success=False, error=f"Executable not found: {e}")
        except PermissionError as e:
            logger.debug(f"[{self.name}] permission denied: {e}")
            return ToolResult(success=False, error=f"Permission denied: {e}")
        except OSError as e:
            logger.debug(f"[{self.name}] OS error: {e}")
            return ToolResult(success=False, error=f"OS error: {e}")




