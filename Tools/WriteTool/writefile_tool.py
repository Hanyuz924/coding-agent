# stdlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

# third-party
from pydantic import BaseModel, Field

# local
import Tools.tool_utils as utils
from Tools.BaseTool import BaseTool, ToolResult, AgentRunContext, logger
from Tools.WriteTool.write_prompt import get_write_tool_description
from Tools.filestateCache import FileState

@dataclass
class FileWriteOutput:
    type:str
    file_path:str
    content:str
    original_file:Union[str, None]
    #TODO Add UI display stuff here


class FileWriteInput(BaseModel):
    file_path: str = Field(
        description="The absolute path to the file to write (must be absolute, not relative)"
    )
    content: str = Field(
        description="The content to write to the file"
    )


class WriteTool(BaseTool):
    """Tool for reading file contents.
    
    Reads a file's contents and returns it with line numbers.
    Supports reading large files in chunks using limit/offset.
    """
    
    @property
    def name(self) -> str:
        """Return the tool name."""
        return "Write"
    
    @property
    def description(self) -> str:
        return get_write_tool_description()
    
    @property
    def concurrent_safe(self) -> bool:
        return False
    
    @property
    def get_pre_read_instruction(self) -> str:
        """Return the pre read instruction to llm"""
        return "If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first."

    @property
    def input_schema(self) -> Dict[str, Any]:
        """Return the tool's input schema.This schema will send to llm , let llm know how to use tool"""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description":"The absolute path to the file to write (must be absolute, not relative)"
                },
                "content":{
                    "type": "string",
                    "description": "The content to write to the file"
                },
            },
            "required": ["file_path", "content"],
        }
    
    def execute(
        self,
        ctx: AgentRunContext,
        **kwargs
    ) -> ToolResult:
        input = FileWriteInput(**kwargs)
        # Step 1: Path normalization and validation
        fullpath = Path(utils.expandPath(input.file_path))
        fullpath_str = str(fullpath)
        
        # Check if path points to an existing directory
        if fullpath.is_dir():
            return ToolResult(
                success=False,
                error=f"{input.file_path} is a directory"
            )

        # For existing files: perform additional validation
        if fullpath.is_file():
            ext = fullpath.suffix
            # Check for binary files
            if ext.lstrip('.').lower() in utils.BINARY_EXTENSIONS:
                return ToolResult(
                    success=False,
                    error=f"This tool cannot write binary files. The file appears to be a binary {ext} file."
                )
            
            # Cache check for existing files
            if not ctx.filecachestate.has(fullpath_str):
                return ToolResult(
                    success=False,
                    error="File has not been read yet. Read it first before writing to it."
                )
            
            # Check if file was modified since read
            file_cache_state = ctx.filecachestate.get_file_state(fullpath_str)
            latest_mtime = int(os.path.getmtime(fullpath))
            if latest_mtime > file_cache_state.modify_timestamp: # type: ignore
                return ToolResult(
                    success=False,
                    error="File has been modified since read, either by the user or by a linter. Read it again before attempting to write it."
                )

        # Step 2: Create parent directories if needed
        try:
            fullpath.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error creating parent directories: {str(e)}"
            )

        # Step 3: Write content to file
        try:
            is_new_file = not fullpath.is_file()
            if not is_new_file:
                original_content = fullpath.read_text(encoding="utf-8")
            fullpath.write_text(input.content, encoding="utf-8")
            
            # Step 4: Update cache with new file state
            new_mtime = int(os.path.getmtime(fullpath_str))
            new_file_state = FileState(
                content=input.content,
                read_timestamp=int(datetime.now().timestamp()),
                modify_timestamp=new_mtime,
            )
            ctx.filecachestate.set(fullpath_str, new_file_state)
            
            return ToolResult(
                success=True,
                data=FileWriteOutput(
                    type= "create" if is_new_file else "update",
                    file_path=fullpath_str,
                    content=input.content,
                    original_file=None if is_new_file else original_content
                ),
                metadata={
                    "source": "disk"
                }
            )
        except PermissionError as e:
            return ToolResult(
                success=False,
                error=f"Permission denied: Cannot write to {input.file_path}. {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error writing to {input.file_path}: {str(e)}"
            )


