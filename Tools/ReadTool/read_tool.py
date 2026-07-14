"""Read tool implementation using class-based architecture."""
# stdlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# third-party
from pydantic import BaseModel, Field

# local
from Tools.BaseTool import BaseTool, ToolResult, AgentRunContext, logger
from Tools.ReadTool.Read_prompt import render_prompt_template, FILE_READ_TOOL_NAME
from Tools.filestateCache import FileState
from Tools.tool_utils import expandPath



def get_read_tool_name()-> str:
    return FILE_READ_TOOL_NAME


DEFAULT_MAX_OUTPUT_TOKENS = 15000
DEFAULT_MAX_OUTPUT_CHARS = 50000
BINARY_EXTENSIONS = {
    'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip', 'tar', 'gz',
    'exe', 'bin', 'so', 'dll', 'pyc', 'mp3', 'mp4', 'mov'
}
BLOCKED_DEVICE_PATHS = {
  '/dev/zero',
  '/dev/random',
  '/dev/urandom',
  '/dev/full',
  '/dev/stdin',
  '/dev/tty',
  '/dev/console',
  '/dev/stdout',
  '/dev/stderr',
  '/dev/fd/0',
  '/dev/fd/1',
  '/dev/fd/2',
}
@dataclass
class FileReadOutput():
    file_path: str
    content: str
    num_lines: int
    start_line: int
    total_line: int


class FileReadInput(BaseModel):
    """LLM -> tool call, should in this format """
    
    file_path: str = Field(
        description="The absolute path to the file to read"
    )
    offset: Optional[int] = Field(
        default=None,
        ge=0,  # >= 0
        description="The line number to start reading from..."
    )
    limit: Optional[int] = Field(
        default=None,
        gt=0,  # > 0
        description="The number of lines to read..."
    )
    pages: Optional[str] = Field(
        default=None,
        description="Page range for PDF files (e.g., '1-5', '3', '10-20')..."
    )


class ReadTool(BaseTool):
    """Tool for reading file contents.
    
    Reads a file's contents and returns it with line numbers.
    Supports reading large files in chunks using limit/offset.
    """
    
    @property
    def name(self) -> str:
        """Return the tool name."""
        return FILE_READ_TOOL_NAME
    
    @property
    def description(self) -> str:
        """Return the tool description."""
        return render_prompt_template()
    @property
    def input_schema(self) -> Dict[str, Any]:
        """Return the tool's input schema."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string", 
                    "description": "The absolute path to the file to read"
                },
                "limit":     {
                    "type": "integer", 
                    "description": "'The number of lines to read. Only provide if the file is too large to read at once."
                },
                "offset":    {
                    "type": "integer", 
                    "description": "The line number to start reading from. Only provide if the file is too large to read at once"
                },
            },
            "required": ["file_path"],
        }
    
    @property
    def read_only(self) -> bool:
        """Read tool is read-only (no side effects)."""
        return True
    
    @property
    def concurrent_safe(self) -> bool:
        """Read tool is safe for concurrent execution."""
        return True
    
    def _validate_contentsize_char(self, content:str):
        if len(content) <= DEFAULT_MAX_OUTPUT_CHARS:
            return None
        return ToolResult(
            success=False,
            error=f"File content {len(content)} chars exceeds maximum allowed chars {DEFAULT_MAX_OUTPUT_CHARS}. Use offset and limit parameters to read specific portions of the file, or search for specific content instead of reading the whole file."
        )
        

    def execute(
        self,
        ctx: AgentRunContext,
        **kwargs
    ) -> ToolResult:
        """Execute the read tool with cache support.
        
        Args:
            file_path: Absolute path to the file to read
            limit: Maximum number of lines to read (optional)
            offset: Starting line number, 0-indexed (optional)
            
        Returns:
            ToolResult: Success with file contents, or error info
        """
        input = FileReadInput(**kwargs)
        # Step 1: Path normalization and validation
        fullpath = Path(expandPath(input.file_path))
        fullpath_str = str(fullpath)
        
        # Check if file exists
        if not fullpath.exists():
            return ToolResult(
                success=False,
                error=f"file not found: {input.file_path}"
            )
        
        # Check if it's a directory
        if fullpath.is_dir():
            return ToolResult(
                success=False,
                error=f"{input.file_path} is a directory"
            )
        
        # Security checks for binary files and blocked paths
        if fullpath.is_file():
            ext = fullpath.suffix
            
            # Check for binary files
            if ext.lstrip('.').lower() in BINARY_EXTENSIONS:
                return ToolResult(
                    success=False,
                    error=f"This tool cannot read binary files. The file appears to be a binary {ext} file. Please use appropriate tools for binary file analysis."
                )
            
            # Check for blocked device paths
            if any(fullpath.is_relative_to(device) for device in BLOCKED_DEVICE_PATHS):
                return ToolResult(
                    success=False,
                    error=f"Cannot read '{fullpath}': this device file would block or produce infinite output."
                )
        valid_offset = input.offset or 0
        valid_limit = input.limit or float("inf")
        
        # Step 2: Check cache with mtime validation
        logger.debug(f"[read] {fullpath_str} offset={valid_offset} limit={valid_limit}")
        if ctx.filecachestate.is_valid(fullpath_str):
            file_states = ctx.filecachestate.get_file_state(fullpath_str)
            cached_content = file_states.content.splitlines(keepends=True)  # type: ignore
            
            # Cache always contains the whole file, extract requested chunk
            start = valid_offset
            end = start + valid_limit if valid_limit != float("inf") else None
            chunk = cached_content[start:end]

            
            if chunk:
                logger.debug(f"[read] cache HIT: {fullpath_str} lines={len(chunk)}")
                output = "".join(
                    f"{start + i + 1}\t{l}" for i, l in enumerate(chunk)
                )
                #avoid large read file output
                validation_file_output_size = self._validate_contentsize_char(output)
                if validation_file_output_size is not None:
                    return validation_file_output_size
                return ToolResult(
                    success=True,
                    data=FileReadOutput(
                        file_path=fullpath_str,
                        content=output,
                        num_lines=len(chunk),
                        start_line=start +1,
                        total_line= len(cached_content)
                    ),
                    metadata={
                        "source": "cache"
                    }
                )
        
        # Step 3: Cache MISS or INVALID - read from disk
        logger.debug(f"[read] cache MISS: {fullpath_str} — reading from disk")
        try:
            content = fullpath.read_text(errors="replace")
            lines = content.splitlines(keepends=True)
            
            # Extract requested chunk
            start = valid_offset
            end = start + valid_limit if valid_limit != float("inf") else None
            chunk = lines[start:end]
            
            # Store whole file in cache
            new_file_state = FileState(
                content=content,
                read_timestamp=int(datetime.now().timestamp()),
                modify_timestamp=int(os.path.getmtime(fullpath_str)),
            )
            ctx.filecachestate.set(fullpath_str, new_file_state)
            output = "".join(
                f"{start + i + 1}\t{l}" for i, l in enumerate(chunk)
            )
            return ToolResult(
                success=True,
                data=FileReadOutput(
                    file_path=fullpath_str,
                    content=output,
                    num_lines=len(chunk),
                    start_line=start + 1,
                    total_line=start + len(chunk)
                ),
                metadata={
                    "source": "disk"
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
