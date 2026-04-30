from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field
import os
from Tools.BaseTool import BaseTool, ToolResult, logger, ToolUseContext
import subprocess
from Tools.GrepTool.Grep_prompt import get_description
from pathlib import Path
from Tools.tool_utils import expandPath
import logging

DEFAULT_HEAD_LIMIT = 250
DEFAULT_TIMEOUT_MS = 120_000
#TODO add logger 

@dataclass
class GrepOutput:
    mode: Literal["content", "files_with_matches", "count"]
    num_files: int
    filenames: list[str]
    content: Optional[str] = None
    num_lines: Optional[int] = None
    num_matches: Optional[int] = None
    applied_limit: Optional[int] = None
    applied_offset: Optional[int] = None


class GrepToolInput(BaseModel):
    time_out: int = Field(
        default=30000,
        description=""
    )
    pattern: str = Field(
        description="Regex pattern to search"
    )
    path: str = Field(
        default=os.getcwd(),
        description="File or dir to search"
    )
    output_mode: str = Field(
        default="files_with_matches",
        description=""
    )

    case_insensitive: bool = Field(
        default=False,
        description=""
    )
    show_line_numbers :bool= Field(
        default=False,
        description=""
    )

    head_limit : int = Field(
        default=DEFAULT_HEAD_LIMIT,
        description=""
    )

    offset: int = Field(
        default=0,
        description=""
    )

    context: int =Field(
        default=0,
        description=""
    )
    type: Optional[str] = Field(
        default=None,
        description="File type to search"
    )
    glob :Optional[str] = Field(
        default=None,
        description=""
    )


class GrepTool(BaseTool):
    @property
    def name(self) -> str:
        return "Grep"
    @property
    def description(self) -> str:
        return get_description()
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return{
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regular expression pattern to search for in file contents",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in (rg PATH). Defaults to current working directory.",
            },
            "glob": {
                "type": "string",
                "description": 'Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob',
            },
            "type": {
                "type": "string",
                "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than glob for standard file types.",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": (
                    'Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), '
                    '"files_with_matches" shows file paths (supports head_limit), '
                    '"count" shows match counts (supports head_limit). Defaults to "files_with_matches".'
                ),
            },
            "context": {
                "type": "number",
                "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise.",
            },
            "-n": {
                "type": "boolean",
                "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true.",
            },
            "-i": {
                "type": "boolean",
                "description": "Case insensitive search (rg -i)",
            },
            "head_limit": {
                "type": "number",
                "description": (
                    'Limit output to first N lines/entries, equivalent to "| head -N". '
                    "Works across all output modes. Defaults to 250 when unspecified. "
                    "Pass 0 for unlimited (use sparingly — large result sets waste context)."
                ),
            },
            "offset": {
                "type": "number",
                "description": 'Skip first N lines/entries before applying head_limit, equivalent to "| tail -n +N | head -N". Defaults to 0.',
            },
        },
        "required": ["pattern"],
    }
    @property
    def read_only(self) -> bool:
        return False
    
    def apply_head_limit(self, search_result:list, limit: Optional[int], offset :int= 0):
        if limit == 0:
            return search_result[offset:]
        effective = limit if limit is not None else DEFAULT_HEAD_LIMIT
        sliced = search_result[offset: offset + effective]
        return sliced
    def to_relative_path(self, absolute_path:str, cwd:str) -> str:
        try:
            return str(Path(absolute_path).relative_to(cwd))
        except ValueError:
            return absolute_path
    
    def split_glob_patterns(self, glob: str) -> list[str]:
        logger.debug(f"[{self.name}] : glob flag search pattern before split {glob}")
        patterns: list[str] = []
        for raw in glob.split():
            if "{" in raw and "}" in raw:
                patterns.append(raw)
            else:
                patterns.extend(p for p in raw.split(",") if p)
        logger.debug(f"[{self.name}] : glob flag search pattern after split {patterns}")
        return patterns
    
    def execute(self,ctx:ToolUseContext, cwd: Optional[str] = None, **kwargs) -> ToolResult:
        cwd = cwd or os.getcwd()
        kwargs["path"] = expandPath(kwargs["path"])
        logger.debug(f"[{self.name}] -- cwd: {cwd}, absolute path {kwargs['path']}")
        input = GrepToolInput(**kwargs)
        for field_name, value in input.model_dump().items():
            logger.debug(f"  {field_name}: {value}")
        args = ["rg", "--hidden"]
        if input.case_insensitive:
            args.append("-i")
        if input.output_mode == "files_with_matches":
            args.append("-l")
        elif input.output_mode == "count":
            args.append("-c")
        
        if input.show_line_numbers and input.output_mode == "content":
            args.append("-n")

                # Patterns starting with '-' must use -e to avoid being parsed as flags
        if input.pattern.startswith("-"):
            args += ["-e", input.pattern]
        else:
            args.append(input.pattern)
        
        if input.type:
            args += ["--type", input.type]

        if input.glob:
            for pat in self.split_glob_patterns(input.glob):
                args += ["--glob", pat]
        args.append(str(input.path))
        logger.debug(f"[{self.name} -- rg cmd : {args}]")
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(timeout=input.time_out/1000)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            logger.debug(f"[{self.name}] --Timed out after {input.time_out/1000:.1f}s and process was killed. ")
            return ToolResult(
                success=False,
                error=f"Timed out after {input.time_out/1000:.1f}s and process was killed.",
                metadata={"returncode": -1, "timed_out": True},
            )    
        if proc.returncode == 2:
             return ToolResult(success=False, error=stderr.decode().strip(),
                      metadata={"returncode": 2})
        search_list = [line for line in stdout.decode().splitlines() if line]
        if not search_list:
            logger.debug(f"[{self.name}] no matches found for pattern={input.pattern!r} path={input.path}")
            return ToolResult(
                success=True,
                data=GrepOutput(
                    mode=input.output_mode, num_files=0, filenames=[], content="",
                    num_matches=0
                ),
                metadata={"extra_info":"No matches found"}
            )
        if input.output_mode == "content":
            limited = self.apply_head_limit(search_list, input.head_limit, input.offset)
            final_lines = []
            for line in limited:
                colon = line.index(":") if ":" in line else -1
                if colon > 0:
                    file_part = line[:colon]
                    rest = line[colon:]
                    final_lines.append(self.to_relative_path(file_part, cwd) + rest)
                else:
                    final_lines.append(line)
            logger.debug(f"[{self.name}] content mode: {len(final_lines)} lines returned (head_limit={input.head_limit}, offset={input.offset})")
            return ToolResult(
                success=True,
                data=GrepOutput(
                    mode="content",
                    num_files=0,
                    filenames=[],
                    content="\n".join(final_lines),
                    num_lines=len(final_lines),
                    applied_offset=input.offset if input.offset > 0 else None,
                )

            )
        if input.output_mode == "count":
            limited = self.apply_head_limit(search_list, input.head_limit, input.offset)
            final_count_lines = []
            total_matches = 0
            file_count = 0
            for line in limited:
                colon = line.rfind(":")
                if colon > 0:
                    file_part = line[:colon]
                    count_str = line[colon + 1:]
                    final_count_lines.append(self.to_relative_path(file_part, cwd) + ":" + count_str)
                    try:
                        total_matches += int(count_str)
                        file_count += 1
                    except ValueError:
                        pass
                else:
                    final_count_lines.append(line)
            logger.debug(f"[{self.name}] count mode: {file_count} files, {total_matches} total matches")
            return ToolResult(
                success=True,
                data=GrepOutput(
                    mode="count",
                    num_files=file_count,
                    filenames=[],
                    content="\n".join(final_count_lines),
                    num_matches=total_matches,
                    applied_offset=input.offset if input.offset > 0 else None,
                )
            )
        stats: list[tuple[str, float]] = []
        for path_str in search_list:
            try:
                mtime = Path(path_str).stat().st_mtime
            except OSError:
                mtime = 0.0
            stats.append((path_str, mtime))
        sorted_matches = [p for p,_ in sorted(stats, key=lambda x:(-x[1], x[0]))]
        final_matches = self.apply_head_limit(sorted_matches, input.head_limit, input.offset)
        relative_matches = [self.to_relative_path(p, cwd) for p in final_matches]
        logger.debug(f"[{self.name}] files_with_matches: {len(relative_matches)} files -> {relative_matches}")
        return ToolResult(
            success=True,
            data=GrepOutput(
                mode="files_with_matches",
                num_files=len(relative_matches),
                filenames=relative_matches,
                applied_offset=input.offset if input.offset > 0 else None,
            )
        )
                


        




    





