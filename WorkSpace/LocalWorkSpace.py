import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import platform
import subprocess
from typing import Any
from pathlib import Path
from pydantic import BaseModel
from utils import utils
import exceptions

logger = logging.getLogger("myagent.workspace")

class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30

class WorkSpace:
    def __init__(self, config: LocalEnvironmentConfig) -> None:
        self.config = config

    def get_template_variable(self):
        return utils.recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ)

    def execute(self, action: dict, cwd: str | None = None, *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the local environment and return a structured result."""
        command = action.get("command")
        if not command:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": "Missing command in action",
                "timed_out": False,
                "extra": {
                    "exception_type": "ValueError",
                    "exception": "Missing command",
                },
            }

        effective_cwd = cwd or self.config.cwd or os.getcwd()
        effective_cwd = str(Path(effective_cwd).resolve())
        effective_timeout = self.config.timeout if timeout is None else timeout
        effective_env = {**os.environ, **self.config.env}

        try:
            logger.debug(f"Running cmd : {command}, cwd:{effective_cwd}"  )
            result = subprocess.run(
                command,
                shell=True,  
                text=True,
                cwd=effective_cwd,
                env=effective_env,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
                "timed_out": False,
                "extra": {
                    "cwd": effective_cwd,
                    "command": command,
                },
            }

        except subprocess.TimeoutExpired as e:
            raw_output = e.output.decode("utf-8", errors="replace") if isinstance(e.output, bytes) else (e.output or "")
            output =  {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"Command timed out after {effective_timeout}s",
                "timed_out": True,
                "extra": {
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                    "cwd": effective_cwd,
                    "command": command,
                },
            }

        except OSError as e:
            output =  {
                "output": "",
                "returncode": -1,
                "exception_info": f"OS error while executing command: {e}",
                "timed_out": False,
                "extra": {
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                    "cwd": effective_cwd,
                    "command": command,
                },
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        #if no exception raised we need to return the output, otherwise last cmd is just task completion cmd
        return output
    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise exceptions.Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )