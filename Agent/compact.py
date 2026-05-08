from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from Agent.compact_prompt import get_partial_compact_prompt, get_compact_user_summary_message
from Model.anthropic_base import get_max_output_tokens

if TYPE_CHECKING:
    from Agent.agent import AgentState

logger = logging.getLogger("myagent.agent")

SNIP_THRESHOLD = 0.60
MICROCOMPACT_IDLE_S = 5 * 60
KEEP_RECENT_RESULTS = 3
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
SNIP_READ_PLACEHOLDER = "[File content snipped - superseded by a more recent read]"
SNIP_TURN = 1
SNIP_READ_TURN = 1
AUTO_COMPACT_THRESHOLD = 0.9

_COMPACT_THRESHOLD = 0.70
_COMPACT_GRAY_ZONE = 0.85


class CompactionMixin:
    def _snip_tool_use_result(self, state: AgentState) -> None:
        read_file_snip: dict = {}
        other_tool_snip: list = []
        logger.debug(f"[Snip Tool Result] Current turn {state.current_turn_number}")
        for index, message in enumerate(state.messages):
            if message.get("role") != "user" and not isinstance(message.get("content"), list):
                continue
            for tool_call_index, tool_call in enumerate(message["content"]):
                if isinstance(tool_call, dict) and tool_call.get("type") == "tool_result" and isinstance(tool_call.get("content"), str) and tool_call["content"] != SNIP_PLACEHOLDER:
                    tool_use_info = self._get_tool_info(state, tool_call["tool_use_id"])  # type: ignore[attr-defined]
                    if not tool_use_info:
                        logger.debug(f"[Snip Tool Result] No tool info found for tool id {tool_call['tool_use_id']}")
                        continue
                    if tool_use_info["name"] in ["Grep", "Glob", "Bash"]:
                        tool_use_turn = state.tool_id_to_turn.get(tool_call["tool_use_id"])
                        if not tool_use_turn:
                            logger.debug(f"[Snip Tool Result] tool used turn for tool id {tool_call['tool_use_id']} not found")
                            continue
                        if tool_use_turn["tool_call_turn"] + SNIP_TURN <= state.current_turn_number:
                            other_tool_snip.append((index, tool_call_index))
                    elif tool_use_info["name"] == "Read":
                        read_file_path = tool_use_info["input"].get("file_path")
                        read_file_snip.setdefault(read_file_path, []).append((index, tool_call_index))

        for message_index, tool_call_index in other_tool_snip:
            tool_id = state.messages[message_index]["content"][tool_call_index]["tool_use_id"]
            tool_info = self._get_tool_info(state, tool_id)  # type: ignore[attr-defined]
            logger.debug(f"[Snip Tool Result]: tool={tool_info['name'] if tool_info else '?'} id={tool_id} msg={message_index}")
            state.messages[message_index]["content"][tool_call_index]["content"] = SNIP_PLACEHOLDER

        for file_path, indices in read_file_snip.items():
            for message_index, tool_call_index in indices[:-1]:
                tool_id = state.messages[message_index]["content"][tool_call_index]["tool_use_id"]
                logger.debug(f"[Snip Tool Result] duplicate-read snip: file={file_path} id={tool_id} msg={message_index}")
                state.messages[message_index]["content"][tool_call_index]["content"] = SNIP_READ_PLACEHOLDER
            last_msg_idx, last_tc_idx = indices[-1]
            last_turn = state.tool_id_to_turn.get(
                state.messages[last_msg_idx]["content"][last_tc_idx]["tool_use_id"]
            )
            if last_turn and last_turn["tool_call_turn"] + SNIP_READ_TURN <= state.current_turn_number:
                tool_id = state.messages[last_msg_idx]["content"][last_tc_idx]["tool_use_id"]
                logger.debug(f"[snip] age-based read snip: file={file_path} id={tool_id} msg={last_msg_idx}")
                state.messages[last_msg_idx]["content"][last_tc_idx]["content"] = SNIP_PLACEHOLDER

    def _microcompact_anthropic(self, state: AgentState) -> None:
        last_api_call_time = getattr(self, "last_api_call_time", None)  # type: ignore[attr-defined]
        if not last_api_call_time or (time.time() - last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        all_results = []
        for mi, msg in enumerate(state.messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for bi, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                    all_results.append((mi, bi))
        clear_count = len(all_results) - KEEP_RECENT_RESULTS
        if clear_count > 0:
            logger.debug(f"[microcompact] idle={time.time() - last_api_call_time:.0f}s, clearing {clear_count}/{len(all_results)} results, keeping last {KEEP_RECENT_RESULTS}")
        for i in range(max(0, clear_count)):
            mi, bi = all_results[i]
            logger.debug(f"[microcompact] clearing msg={mi} block={bi}")
            state.messages[mi]["content"][bi]["content"] = "[Old result cleared]"

    def _respect_tool_pairs(self, messages: list, raw: int) -> int:
        """Advance raw forward until it lands on a plain user message (not tool_result)."""
        for i in range(raw, len(messages)):
            msg = messages[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return i
            if isinstance(content, list) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                return i
        return len(messages)

    def find_split_point(self, messages: list, keep_ratio: float = 0.3) -> int:
        """Find index splitting messages so ~keep_ratio of tokens are in the recent portion.

        Returns 0 if no safe split exists (caller should compact everything).
        """
        if not messages:
            return 0
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        total = self._estimate_tokens(messages)  # type: ignore[attr-defined]
        target = int(total * keep_ratio)
        running = 0
        raw = 0
        for i in range(len(messages) - 1, -1, -1):
            running += self._estimate_tokens([messages[i]])  # type: ignore[attr-defined]
            if running >= target:
                raw = i
                break
        adjusted = self._respect_tool_pairs(messages, raw)
        if adjusted >= len(messages):
            return 0
        return adjusted

    def _check_and_compact(self, state: AgentState, context_window: int) -> None:
        if state.total_input_tokens > context_window * AUTO_COMPACT_THRESHOLD:
            logger.info(f"[compact] context at {state.total_input_tokens}/{context_window} tokens, triggering auto_compact")
            self.auto_compact(state)

    def auto_compact(self, state: AgentState) -> None:
        split = self.find_split_point(state.messages)
        old = state.messages[:split] if split > 0 else state.messages
        kept = state.messages[split:] if split > 0 else []
        

        model = self.model  # type: ignore[attr-defined]
        summary_resp = model.client.messages.create(
            model=model.model_name,
            max_tokens=get_max_output_tokens(model= model.model_name),
            system=get_partial_compact_prompt(direction="up_to"),
            messages=[
                *old,
                {"role": "user", "content": "Please provide your summary now."},
            ],
        )
        summary_text = summary_resp.content[0].text if summary_resp.content and summary_resp.content[0].type == "text" else "No summary available."
        logger.info(f"[compact] split={split}, compacted {len(old)} msgs, kept {len(kept)}")
        print("\n[Auto Compaction Summary]\n" + summary_text + "\n")
        summary_user_content = get_compact_user_summary_message(
            summary_text,
            recent_messages_preserved=bool(kept),
        )
        state.messages = [
            {"role": "user", "content": summary_user_content},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation. How can I continue helping?"},
            *kept,
        ]
        state.total_input_tokens = 0
