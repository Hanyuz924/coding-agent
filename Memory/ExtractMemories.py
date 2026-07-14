
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field

from ForkAgent.fork_agent import build_cache_safe_params, run_fork_agent_background
from Agent.types import Message, AgentRunContext, AgentDefinition, AgentState
from Tools.BashTool.bash_prompt import (
    FILE_EDIT_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
)
from Memory.MemoryPrompt import build_extract_prompt

logger = logging.getLogger("myagent.memory.extraction")

MAX_MEMORY_FILES = 200

FRONTMATTER_MAX_LINES = 30

@dataclass
class MemoryHeader:
    filename: str = ""
    file_path: str = ""
    mtime: float = 0.0
    description: str | None = None
    type: str | None = None
    memory_name: str | None = None
    meta: dict = field(default_factory=dict)
    content: str = ""

@dataclass
class MemExtractionState:
    last_memory_message_id: str | None = None
    turn_since_last_extraction: int = 0

"""
Format memory headers as a text manifest: one line per file with
[type] filename (timestamp): description. Used by both the recall
selector prompt and the extraction-agent prompt.
"""

def format_memory_manifest(mem_headers:list[MemoryHeader]) -> str:
    def fmt(m: MemoryHeader) -> str:
        tag = f"[{m.type}] " if m.type else ""
        ts = datetime.fromtimestamp(m.mtime / 1000, tz=timezone.utc).isoformat()
        base = f"- {tag}{m.filename} ({ts})"
        return f"{base}: {m.description}" if m.description else base
    return "\n".join(fmt(m) for m in mem_headers)


def memory_frontmatter_parser(content: list[str]) -> MemoryHeader:
    header = MemoryHeader()
    in_frontmatter = False
    delimiter_count = 0
    for line in content:
        stripped = line.strip()
        if stripped == "---":
            delimiter_count += 1
            in_frontmatter = delimiter_count == 1
            continue
        if not in_frontmatter:
            continue
        if stripped.startswith("description:"):
            header.description = stripped[12:].strip()
        elif stripped.startswith("name:"):
            header.memory_name = stripped[5:].strip()
        elif stripped.startswith("node_type:"):
            header.meta["node_type"] = stripped[10:].strip()
        elif stripped.startswith("originSessionId:"):
            header.meta["originSessionId"] = stripped[16:].strip()
        elif stripped.startswith("type:"):
            header.type = stripped[5:].strip()
            header.meta["type"] = stripped[5:].strip()
    return header

def scan_memory_files(memory_dir: str) -> list[MemoryHeader]:
    d = Path(memory_dir)
    if not d.exists():
        return []
    headers = []
    for f in d.rglob("*.md"):
        if f.name == "MEMORY.md":
            continue
        with f.open() as fh:
            lines = fh.readlines()[:FRONTMATTER_MAX_LINES]
        header = memory_frontmatter_parser(lines)
        header.filename = f.name
        header.file_path = str(f.absolute())
        header.mtime = f.stat().st_mtime
        headers.append(header)
    return headers

def get_written_file_paths(block: dict) -> str | None:
    if block["type"] != "tool_use":
        return None
    if block["name"] not in (FILE_EDIT_TOOL_NAME, FILE_WRITE_TOOL_NAME):
        return None
    if isinstance(block["input"], dict) and "file_path" in block["input"]:
        return block["input"]["file_path"]
    return None

def get_auto_mem_path() -> str:
    # TODO: configurable via env var AGENT_MEMORY_DIR
    save_path = Path(__file__).parent / "store"
    return str(save_path)


def extract_written_paths(messages: list[Message]):
    paths = []
    for message in messages:
        if message.role != "assistant":
            continue
        if not isinstance(message.content, list):
            continue
        for block in message.content:
            file_path = get_written_file_paths(block)
            if file_path:
                paths.append(file_path)
    return paths

def count_new_message(messages: list[Message], last_extraction_id: str | None) -> int:
    count = 0
    for i in range(len(messages)-1 , -1, -1):
        if messages[i].message_id == last_extraction_id:
            return count
        count +=1
    return count


def memory_extraction(
    messages: list[Message],
    agent_def: AgentDefinition,
    ctx: AgentRunContext,
    mem_extraction_state: MemExtractionState,
) -> asyncio.Task | None:
    mem_dir = get_auto_mem_path()
    logger.debug(f"[memory_extraction] triggered mem_dir={mem_dir} last_id={mem_extraction_state.last_memory_message_id} msgs={len(messages)}")

    new_message_count = count_new_message(messages, mem_extraction_state.last_memory_message_id)
    logger.debug(f"[memory_extraction] new_message_count={new_message_count}")
    if new_message_count == 0:
        logger.debug("[memory_extraction] skipping — no new messages since last extraction")
        return None

    existing_mem_headers = scan_memory_files(mem_dir)
    logger.debug(f"[memory_extraction] found {len(existing_mem_headers)} existing memory files in {mem_dir}")
    existing_mem = format_memory_manifest(existing_mem_headers)

    extraction_prompt = Message(
        role="user",
        content=build_extract_prompt(
            new_message_count=new_message_count,
            memory_dir=mem_dir,
            existing_memories=existing_mem,
            skip_index=True,
        ),
    )

    if ctx.rendered_system_prompt is None:
        logger.warning("[memory_extraction] skipping — ctx.rendered_system_prompt is None, cannot build cache_safe_params")
        return None

    cache_safe_params = build_cache_safe_params(
        model=ctx.options.model,
        system_prompt=ctx.rendered_system_prompt,
        tools=ctx.options.tools,
        fork_context_messages=messages,
    )
    logger.debug(f"[memory_extraction] launching background fork model={ctx.options.model} tools={len(ctx.options.tools)} fork_msgs={len(messages)}")

    extraction_task = run_fork_agent_background(
        parent_ctx=ctx,
        agent_def=agent_def,
        cachesafeparams=cache_safe_params,
        user_message=extraction_prompt,
        agent_type="memory-extraction",
        disable_compact=True,
        max_turns=5,
    )
    logger.debug(f"[memory_extraction] background task created: {extraction_task}")
    return extraction_task



    
    







