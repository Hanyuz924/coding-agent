import json
import time
import logging
import litellm
from typing import Any
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.model")

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

class ModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict = Field(default_factory=dict)

    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "<returncode>{{output.returncode}}</returncode>\n"
        "<output>\n{{output.output}}</output>"
    )

class BaseModelClass:
    def __init__(self, model_name: str, config: ModelConfig) -> None:
        self.model_name = model_name
        self.config = config
        logger.info(f"BaseModelClass initialized with model: {model_name}")
        logger.debug(f"Model config: {config}")

    def get_template_variable(self):
        return self.config.model_dump()
    
    def pre_process_prompt(self):
        pass
    
    def call_llm_api(self, prompt: list[dict[str, str]]):
        try:
            logger.info(f"Calling LLM API with model={self.model_name}, {len(prompt)} messages")
            logger.info(f"Prompt messages: {prompt}")
            response = litellm.completion(
                model= self.model_name,
                messages=prompt,
                tools=[BASH_TOOL],
                **self.config.model_kwargs
            )
            logger.info(f"LLM API response received: {response}")
            return response
        except litellm.exceptions.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
    def parse_actions(self, llm_response:litellm.ModelResponse) -> list[dict]:
        tool_calls = llm_response.choices[0].message.tool_calls 
        if tool_calls is None:
            #invalid tool calls just return or raise error???   
            return []
        actions_list: list[dict] = []
        for tool_call in tool_calls:
            error_msg = ""
            cmd_and_arguments = {}
            try:
                cmd_and_arguments = json.loads(tool_call.function.arguments)
            except Exception as e:
                error_msg = f"Error parsing tool call arguments: {e}."
            if tool_call.function.name != "bash":
                error_msg += f"Unknown tool '{tool_call.function.name}'."
            if not isinstance(cmd_and_arguments, dict) or "command" not in cmd_and_arguments:
                error_msg += "Missing 'command' argument in bash tool call."
            if not error_msg:
                actions_list.append({"command": cmd_and_arguments["command"], "tool_call_id": tool_call.id})
        return actions_list
    def parse_response(self, llm_response:litellm.ModelResponse):
        raw_content = llm_response.choices[0].message.content
        role = llm_response.choices[0].message.role
        raw_tool_calls = llm_response.choices[0].message.tool_calls or []
        cost = self._calculate_cost(llm_response)

        tool_calls = []
        for tc in raw_tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

        return {
            "role":role,
            "content": raw_content,
            "tool_calls": tool_calls,
            "cost": cost,
            "timestamp":time.time()
        }
    def format_toolcall_result_message(
        self,
        action_list:list[dict],
        output_list:list[dict],
        observation_template:str,
    )-> list[dict]:
        finished_actions= {"output": "", "returncode": -1, "exception_info": ""}
        total_output = output_list + [finished_actions] * (len(action_list) - len(output_list))
        ret:list[dict] = []
        for index, (action, output) in enumerate(zip(action_list, total_output)):
            content = Template(observation_template, undefined=StrictUndefined).render(
                output=output, 
            )
            if index >= len(output_list):
                content += f"\n<exception_info>Execution stopped after action {len(output_list)} because task was submitted.</exception_info>"

            msg = {
                "content": content,
                # "extra": {
                #     "raw_output": output.get("output", ""),
                #     "returncode": output.get("returncode"),
                #     "timestamp": time.time(),
                #     "exception_info": output.get("exception_info"),
                #     **output.get("extra", {}),
                # },
            }
            if action.get("tool_call_id", None) is not None:
                msg["tool_call_id"] = action.get("tool_call_id")
                msg["role"] = "tool"
            else:
                # human issued command
                msg["role"] = "user"
            ret.append(msg)
        return ret
    
    def _calculate_cost(self, response: litellm.ModelResponse) -> float:
        cost = litellm.completion_cost(completion_response=response)
        return cost


        
        

        
