import logging
import time
import anthropic
from pathlib import Path
import os
from datetime import datetime
from typing import Generator
from Model import anthropic_base
from Model.anthropic_base import ThinkingChunk, TextChunk, AssistantTurn, get_context_window
from Tools.ReadTool.read_tool import get_read_tool_name
from Tools.BaseTool import ToolUseContext
from Tools.filestateCache import FileStateCache
from Tools import BaseTool
# from Tools.BaseTool import ToolResult
from Tools.permission_check import check_permission
from Tools.registry import dispatch, get_tool_schemas
from Agent.compact import CompactionMixin

logger = logging.getLogger("myagent.agent")

from dataclasses import dataclass, field


NONPERSISTABLE_TOOLS = ["Read"]
MAX_TOOL_RESULT_CHARS = 50_000
TOOL_RESULT_PERSIST_THRESHOLD = 30 * 1024 # 30KB 
PREVIEW_TOOL_RESULT_LINES = 150



#Functions 
def truncate_tool_result(result:str) -> str:
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    head_tail_keep =(MAX_TOOL_RESULT_CHARS - 60) // 2
    return (
        result[:head_tail_keep]
        + f"\n\n[... truncated {len(result) - head_tail_keep * 2} chars ...]\n\n"
        + result[-head_tail_keep:]
    )

def persistLargeResult(tool_name: str, tool_result:str, tool_call_id:int) -> str:
    #should never persist a read file tool result to avoid circular issue
    if tool_name in NONPERSISTABLE_TOOLS:
        return tool_result
    encoded = tool_result.encode()
    if len(encoded) <= TOOL_RESULT_PERSIST_THRESHOLD:
        return tool_result
    save_path = Path.home()/"coding-agent"/"tmp"/"tool_result"
    os.makedirs(save_path, exist_ok=True)
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S") +"_"+ tool_name + "_" + str(tool_call_id)
    save_file_path = Path.joinpath(save_path, file_name)
    with open(save_file_path, mode="w") as f:
        f.write(tool_result)
    f.close()
    lines = tool_result.split("\n")
    preview = "\n".join(lines[:PREVIEW_TOOL_RESULT_LINES])
    return (
        f"Result too large ({len(encoded) // 1024} KB, {len(lines)} lines). "
        f"Full output saved to {save_file_path}, use {get_read_tool_name()} to read it.\n\n"
        f"Preview (first {PREVIEW_TOOL_RESULT_LINES} lines):\n{preview}"
    )


@dataclass
class AgentState:
    """
    A new state will be created at the begining of each turn.
    Mutable session state across loop with in the same turn.
    """
    messages : list = field(default_factory= list)
    tool_id_to_turn: dict[str, dict] = field(default_factory=dict)
    total_input_tokens : int = 0
    total_output_tokens : int = 0
    turn_count : int = 0
    transition_reason: str | None = None    
    current_turn_number: int = 0
    #TODO  Add tool use context later: purpose to save file state and file cache info 
    filecachestate: FileStateCache = field(default_factory=FileStateCache)

"""
Internal streaming state ,used for log and display
"""
@dataclass
class ToolStart:
    name:   str
    inputs: dict

#For UI display purpose we can set the result to ToolReuslt type 
@dataclass
class ToolEnd:
    name:      str
    result:    BaseTool.ToolResult | None
    permitted: bool = True

@dataclass
class TurnDone:
    input_tokens:  int
    output_tokens: int
    turn_done_reason: str

@dataclass
class PermissionRequest:
    description: str
    granted: bool = False

class BaseAgent(CompactionMixin):
    def __init__(
            self,
            model:anthropic_base.AnthropicModelClass, 
        ) -> None:
        self.model = model
        self.last_api_call_time = None
        self.effective_context_window = get_context_window(self.model.model_name)

        
        self.current_cost = 0.0
        logger.info(f"Agent initialized with model: {model.model_name}")
    
    def _estimate_tokens(self, messages: list,) -> int:
        total_chars = 0
        message_count = 0
        for message in messages:
            message_count +=1
            content = message.get("content","")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        for v in block.values():
                            total_chars+=len(v)
            #user message with content is string
            else:
                total_chars+=len(content)
        content_tokens = int(total_chars / 2.8)
        framing_tokens = message_count * 4
        return int((content_tokens + framing_tokens) * 1.1)
    
    def _estimate_tokens_by_api(self, system_prompt: str, messages: list, tool_use_schemas: list) -> int:
        return self.model.count_tokens_api(system_message=system_prompt, messages=messages, tool_use_schemas=tool_use_schemas)


    def _get_tool_info(self, state: AgentState, tool_call_id :str):
        for message in state.messages:
            if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_call_id:
                    return {"name": block["name"], "input": block.get("input", {})}
        return None


    """
    For each new turn we just  need to recieve a new user message.
    For continue conversation, we need to use the chat history 
    After each turn end, we need to add the messsage back to chat history?

    TBD: do we need to take new sysrem prompt everytime, every turn ?

    Multi-turn agent loop (generator) streaming purpose.
    Yields: TextChunk | ThinkingChunk | ToolStart | ToolEnd |
            PermissionRequest | TurnDone

    """
    def queryLoop(
        self,
        user_message: str,
        system_prompt:str,
        state: AgentState,
        max_retry: int = 3,
    ) -> Generator:
        
        new_user_message = {"role": "user", "content":user_message}
        state.messages.append(new_user_message)
        state.current_turn_number +=1
        while 1 :
            assistant_turn: AssistantTurn | None = None
            #for every api call check sinp tool result and microcompat 
            self._snip_tool_use_result(state)
            self._microcompact_anthropic(state)
            
            #add retry for LLM call to, we may see rate limit error
            for attempt in range(max_retry + 1):
                try:
                    for stream in self.model.stream_anthropic(
                        system_message= system_prompt,
                        tool_use_schemas=get_tool_schemas(),
                        messages=state.messages
                    ):
                        if isinstance(stream, (ThinkingChunk, TextChunk)):
                            yield stream
                        elif isinstance(stream, AssistantTurn):
                            assistant_turn = stream
                    if assistant_turn is not None:
                        break
                except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                    if attempt >= max_retry:
                        raise
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"{type(e).__name__}, retrying in {wait}s (attempt {attempt + 1}/{max_retry})")
                    time.sleep(wait)
                except Exception as e:
                    logger.error(f"Unexpected error during API call: {type(e).__name__}: {e}")
                    raise
            #This should only happen when 3 retry all reach the rate limit here
            if assistant_turn is None:
                break
            self.last_api_call_time = time.time()
            #At this point we got the actual assitant response here
            assistant_content = []
            if assistant_turn.text:
                assistant_content.append({"type": "text", "text": assistant_turn.text})
            for tc in assistant_turn.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            state.messages.append({"role": "assistant", "content": assistant_content})
            state.turn_count += 1
            state.total_input_tokens += assistant_turn.in_tokens
            state.total_output_tokens += assistant_turn.out_tokens
            #For streaming purpose just yield Turn done for UI 
            yield TurnDone(assistant_turn.in_tokens, assistant_turn.out_tokens, assistant_turn.stop_reason)

            if assistant_turn.stop_reason == "end_turn":
                break# conversation round complete 

            #Handle Tool calls here and collect the tool call result info
            tool_calls_result = []
            for tool_call in assistant_turn.tool_calls:
                yield ToolStart(name=tool_call["name"], inputs=tool_call["input"])
                #1. For each tool call we check the permission first
                permissions = check_permission(tool_call["name"], tool_call["input"])
                if permissions["action"] == "deny":
                    tool_calls_result.append({
                        "type":"tool_result",
                        "tool_use_id":tool_call["id"],
                        "content":f"Action denied: {permissions.get('message', '')}"
                    })

                    yield ToolEnd(name=tool_call["name"], result=None, permitted=False)
                    continue

                if permissions["action"] == "confirm":
                    user_input_permission = PermissionRequest(
                        description=permissions["message"])
                    yield user_input_permission
                    if not user_input_permission.granted:
                        tool_calls_result.append({
                            "type":"tool_result",
                            "tool_use_id":tool_call["id"],
                            "content":"Action denied by user."
                        })
                        yield ToolEnd(name=tool_call["name"], result=None, permitted=False)
                        continue

                tool_use_ctx = ToolUseContext(
                    filecachestate=state.filecachestate,
                    tooluse_id=tool_call["id"],
                )
                try:
                    tool_result = dispatch(tool_name= tool_call["name"], tool_input=tool_call["input"], ctx=tool_use_ctx)
                except Exception as e:
                    logger.error(f"Tool execution error for {tool_call['name']}: {type(e).__name__}: {e}")
                    tool_result = BaseTool.ToolResult(
                        success=False,
                        data=None,
                        error=f"{type(e).__name__}: {str(e)}",
                        metadata={"tool_name": tool_call["name"], "error_type": type(e).__name__}
                    )
                tool_calls_result.append({
                    "type":"tool_result",
                    "tool_use_id":tool_call["id"],
                    "content":tool_result.to_api_content()
                })
                #only add the tool call to quick look up dict when there is actual tool call happen
                state.tool_id_to_turn[tool_call["id"]] = {
                    "tool_call_turn":state.current_turn_number,
                    "tool_name":tool_call["name"]
                }
            state.messages.append({"role":"user", "content":tool_calls_result})
            self._check_and_compact(state,self.effective_context_window)

    
   
    
    














