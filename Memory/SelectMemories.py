import json
import anthropic
import logging
from anthropic.types import Message
from pathlib import Path
import os
from dataclasses import dataclass

from Model.anthropic_base import AnthropicModelClass
from Agent.types import Message, Attachment
from Memory.ExtractMemories import (
    MemoryHeader, 
    format_memory_manifest, 
    get_auto_mem_path, 
    memory_frontmatter_parser,
    FRONTMATTER_MAX_LINES
)
from Tools.permission_check import FILE_READ_TOOL_NAME


logger = logging.getLogger("myagent.memory.selection")

SELECT_MEMORIES_SYSTEM_PROMPT = """You are selecting memories that will be useful to Claude Code as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a list of filenames for the memories that will clearly be useful to Claude Code as it processes the user's query (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful in processing the user's query, then do not include it in your list. Be selective and discerning.
- If there are no memories in the list that would clearly be useful, feel free to return an empty list.
- If a list of recently-used tools is provided, do not select memories that are usage reference or API documentation for those tools (Claude Code is already exercising them). DO still select memories containing warnings, gotchas, or known issues about those tools — active use is exactly when those matter.
"""
MEMORY_TRUNCATED_NOTE = (
    "\n\n> This memory file was truncated ({limit_desc}). "
    "Use the {read_tool} tool to view the complete file at: {file_path}"
)

MAX_MEMORY_LINES = 200
MAX_MEMORY_BYTES = 4096


@dataclass(frozen=True)
class RelevantMemoriesAttachment(Attachment):
    memories: list[MemoryHeader]
    def render(self) -> list[dict]:
        return [{"type": "text", "text": _render_memories(self.memories)}]
    

def _render_memories(mem_header:list[MemoryHeader]):
    if not mem_header:
        return ""
    blocks = [f"{m.memory_name}\n{m.content}" for m in mem_header]
    body = "\n\n".join(blocks)
    return (
        "<relevant_memories>\n"
        "The following memories may be relevant to the current task:\n\n"
        f"{body}\n"
        "</relevant_memories>"
    )


def get_memory_attachment(model_response) -> RelevantMemoriesAttachment | None:
    file_names = parse_selected_filenames(model_response)
    if not file_names:
        return
    mem_header = read_mem_file(file_names)
    if not mem_header:
        return
    mem_attchment = RelevantMemoriesAttachment(
        memories=mem_header
    )
    return mem_attchment


def read_mem_file(mem_files: list[str]) -> list[MemoryHeader] | None:
    if not mem_files:
        return None
    #file name in string here is not the abs path
    auto_mem_path = Path(get_auto_mem_path()) 
    mem_read_result = []
    for file in mem_files:
        file_path = (auto_mem_path/file).resolve()
        if not file_path.exists():
            logger.warning("memory file not found: %s", file_path)
            continue
        with open(file_path) as f:
            content = f.readlines()
        truncated = len(content) > MAX_MEMORY_LINES
        mem_header = memory_frontmatter_parser(content[:FRONTMATTER_MAX_LINES])
        lines = content[:MAX_MEMORY_LINES]
        text = "".join(lines)
        if truncated:
            text += MEMORY_TRUNCATED_NOTE.format(
                limit_desc=f"first {MAX_MEMORY_LINES} lines",
                read_tool=FILE_READ_TOOL_NAME,
                file_path=str(file_path),
            )
        mem_header.content = text
        mem_read_result.append(mem_header)
    return mem_read_result


def parse_selected_filenames(response, known_filenames: set[str] | None = None) -> list[str]:

    if response.stop_reason == "max_tokens":
        logger.warning("output truncated (max_tokens); result may be incomplete")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        logger.warning("no text block in response (stop_reason=%s)", response.stop_reason)
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.exception("failed to parse structured output as JSON: %r", text[:200])
        return []
    filenames = data.get("selected_memories", [])
    if not isinstance(filenames, list):
        logger.warning("selected_memories is not a list: %r", filenames)
        return []
    filenames = [f for f in filenames if isinstance(f, str)]
    if known_filenames is not None:
        filenames = [f for f in filenames if f in known_filenames]

    return filenames



async def select_relevant_memories(user_query:Message, memory_list:list[MemoryHeader]):
    tem_model = AnthropicModelClass(
        model_name="claude-haiku-4-5",
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    memory_manifest = format_memory_manifest(mem_headers=memory_list)
    try:
        resp = await tem_model.side_query(
            messages=[
                {"role":"user",
                       "content": f"Query: {user_query}\n\nAvailable memories:\n{memory_manifest}"
                }
            ],
            system=SELECT_MEMORIES_SYSTEM_PROMPT,
            output_format={
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected_memories": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["selected_memories"],
                    "additionalProperties": False,
                },
            },
            max_tokens=256,
            max_retries=3,
            query_source="mem_selection_side_query"
        )
    except anthropic.APIStatusError as e:
        logger.warning("memory selection API error (status %s): %s", e.status_code, e)
        return []                               
    except anthropic.APIConnectionError as e:
        logger.warning("memory selection connection error: %s", e)
        return []
    except Exception:
        logger.exception("memory selection failed unexpectedly")
        return []

    try:
        return parse_selected_filenames(resp)    # list of filenames
    except Exception:
        logger.exception("failed to parse selected memory filenames")
        return []

    