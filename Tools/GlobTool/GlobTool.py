# stdlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# third-party
from pydantic import BaseModel, Field

# local
from Tools.BaseTool import BaseTool, ToolResult, AgentRunContext, logger
from Tools.tool_utils import expandPath

GLOB_TOOL_NAME = 'Glob'
DESCRIPTION = """
- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
"""

@dataclass
class GlobOutput:
    filenames: list[str]
    numFiles: int


class GlobToolInput(BaseModel):
    pattern: str = Field(
        description="The glob pattern to match files against"
    )
    path: str = Field(
        default=os.getcwd(),
        description="""The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter "undefined" or "null" - simply omit it for the default behavior. Must be a valid directory path if provided."""
    )
    time_out: int = Field(
        default=30000,
        description="Default timeout for glob cmd in ms."
    )

class GlobTool(BaseTool):
    @property
    def name(self) -> str:
        return GLOB_TOOL_NAME
    @property
    def description(self) -> str:
        return DESCRIPTION
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type":"object",
            "properties":{
                "pattern":{
                    "type":"string",
                    "description":"The glob pattern to match files against"
                },
                "path":{
                    "type":"string",
                    "description":"""
                        The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter "undefined" or "null" - simply omit it for the default behavior. 
                         Must be a valid directory path if provided.
                    """
                }
            },
            "required": ["pattern"],
        }
    @property
    def read_only(self) -> bool:
        return True
    
    @property
    def concurrent_safe(self) -> bool:
        return True
    

    """
    Extracts the static base directory from a glob pattern.
    The base directory is everything before the first glob special character (* ? [ {).
    Returns the directory portion and the remaining relative pattern.
    """
    def extract_glob_base_directory(self, pattern: str) -> tuple[str, str]: 
        logger.debug(f"[{self.name}]: pattern before base dir extraction -- {pattern}")
        glob_chars = set("*?[{")
        glob_chars_index = [i for i, c in enumerate(pattern) if c in glob_chars]
        if glob_chars_index is []:
            p = Path(pattern)
            return str(p.parent), p.name
        
        prefix = pattern[:glob_chars_index[0]]
        last_sep = max(prefix.rfind("/"), prefix.rfind(os.sep))
        
        if last_sep == -1:
            return "", pattern
        #if only / in the pattern, so we retunr the home dir 
        base_dir = prefix[:last_sep] or ("/" if last_sep == 0 else "")
        relative_pattern = pattern[last_sep + 1:]

        return base_dir, relative_pattern
    
    def execute(self, ctx: AgentRunContext, **kwargs) -> ToolResult:
        input = GlobToolInput(**kwargs)
        search_dir = input.path
        search_pattern = input.pattern

        if Path(input.pattern).is_absolute():
            base_dir, relative_pattern = self.extract_glob_base_directory(input.pattern)
            logger.debug(f"[{self.name}]: base dir  after base dir extraction -- {base_dir}, pattern {relative_pattern}")
            if base_dir:
                search_dir = base_dir
                search_pattern = relative_pattern
        logger.debug(f"[{self.name}] search_dir={search_dir!r}, search_pattern={search_pattern!r}")
        args =[
            "rg",
            "--files",
            "--glob", search_pattern,
            "--sort=modified",
            search_dir
        ]
        proc= subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            stdout, stderr = proc.communicate(timeout=input.time_out / 1000)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            logger.debug(f"[{self.name}] timed out after {input.time_out/1000:.1f}s")
            return ToolResult(
                success=False,
                error=f"Timed out after {input.time_out/1000:.1f}s and process was killed.",
                metadata={"returncode": -1, "timed_out": True},
            )
        if proc.returncode == 2:
            logger.debug(f"[{self.name}] rg error (returncode=2): {stderr.decode().strip()}")
            return ToolResult(success=False, error=stderr.decode().strip(),
                      metadata={"returncode": 2})
        search_list = [line for line in stdout.decode().splitlines() if line]
        absolute_paths = [
            p if Path(p).is_absolute() else str(Path(search_dir) / p)
            for p in search_list
        ]
        logger.debug(f"[{self.name}] {len(absolute_paths)} files matched -> {absolute_paths}")
        return ToolResult(
            success=True,
            data=GlobOutput(
                filenames=absolute_paths,
                numFiles=len(absolute_paths)
            )
        )

    