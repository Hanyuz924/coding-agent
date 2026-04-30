from pathlib import Path
from typing import Any, Dict, Union
from pydantic import BaseModel, Field
import os
from Tools.BaseTool import BaseTool, ToolResult, ToolUseContext,logger
from Tools.filestateCache import FileState
from datetime import datetime
import Tools.tool_utils as utils
from dataclasses import dataclass
from Tools.EditTool.Edit_prompt import get_edit_tool_description


class FileEditInput(BaseModel):
    file_path: str =Field(
        description="The absolute path to the file to modify"
    )
    old_string: str = Field(
        description="The text to replace"

    )
    new_string : str = Field(
        description="The text to replace it with(must be different from old_string)"

    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string(default false)"

    )
@dataclass
class FileEditOutput:
    file_path:str
    old_string:str
    new_string:str
    originalFile:str
    replace_all:bool

class EditTool(BaseTool):
    
    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return get_edit_tool_description()
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type":"object",
            "properties":{
                "file_path":{
                    "type":"string",
                    "description":"The absolute path to the file to modify"
                },
                "old_string":{
                    "type": "string",
                    "description":"The text to replace"
                },
                "new_string":{
                    "type":"string",
                    "description":"The text to replace it with (must be different from old_string)"
                },
                "replace_all":{
                    "type": "boolean",
                    "description":"Replace all occurrences of old_string (default false)"
                }
            },
            "required":["file_path", "old_string", "new_string"]
        }

    def execute(self,toolusecontext:ToolUseContext, **kwargs) -> ToolResult:
        input = FileEditInput(**kwargs)
        fullpath = Path(utils.expandPath(input.file_path))
        fullpath_str = str(fullpath)
        if fullpath.is_dir():
            return ToolResult(
                success=False,
                error=f"{input.file_path} is a directory"
            )
        if not fullpath.exists():
            #we got new file here 
            if input.old_string != "":
                return ToolResult(
                    success=False,
                    error=f"file does not exsist, but old string is not empty, is {input.old_string}"
                )
            if input.new_string == "":
                return ToolResult(
                    success=False,
                    error=f"new file create, but new_string is empty"
                )
            fullpath.write_text(input.new_string, encoding="utf-8")
            #TODO add the new file to cache ???
            new_mtime = int(os.path.getmtime(fullpath_str))
            new_file_state = FileState(
                content=input.new_string,
                read_timestamp=int(datetime.now().timestamp()),
                modify_timestamp=new_mtime,
            )
            toolusecontext.filecachestate.set(fullpath_str, new_file_state)
            return ToolResult(
                success=True,
                data=FileEditOutput(
                    file_path=fullpath_str,
                    old_string="",
                    new_string=input.new_string,
                    originalFile="",
                    replace_all=input.replace_all
                )
            )
        #here for path exsist and we need to do the actual replacement for content
        if not toolusecontext.filecachestate.has(fullpath_str):
                return ToolResult(
                    success=False,
                    error="File has not been read yet. Read it first before writing to it."
                )
        file_cache_state = toolusecontext.filecachestate.get_file_state(fullpath_str)
        latest_mtime = int(os.path.getmtime(fullpath))
        if latest_mtime > file_cache_state.modify_timestamp: # type: ignore
            return ToolResult(
                success=False,
                error="File has been modified since read, either by the user or by a linter. Read it again before attempting to write it."
            )

        original_content = fullpath.read_text(encoding="utf-8")
        if input.old_string not in original_content:
            return ToolResult(
                success=False,
                error=f"old_string not find in {fullpath_str}"
            )
        if not input.replace_all and original_content.count(input.old_string) >1 :
            return ToolResult(
                success=False,
                error=f"old_string is not unique in {fullpath_str}"
            )
        new_content = original_content.replace(input.old_string, input.new_string) if input.replace_all \
                    else original_content.replace(input.old_string, input.new_string, 1)
        fullpath.write_text(new_content)
        new_mtime = int(os.path.getmtime(fullpath_str))
        toolusecontext.filecachestate.set(fullpath_str, FileState(
            content=new_content,
            read_timestamp=int(datetime.now().timestamp()),
            modify_timestamp=new_mtime,
        ))
        return ToolResult(
            success=True,
            data=FileEditOutput(
                file_path=fullpath_str,
                old_string=input.old_string,
                new_string=input.new_string,
                originalFile=original_content,
                replace_all=input.replace_all
            )
        )
            

