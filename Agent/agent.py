import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from Model import litellm_base
from WorkSpace import LocalWorkSpace
from pathlib import Path
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel
import exceptions as ex
from utils import utils

logger = logging.getLogger("myagent.agent")

class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None

class BaseAgent:
    def __init__(self, model:litellm_base.BaseModelClass, workspace: LocalWorkSpace.WorkSpace, config_info:AgentConfig) -> None:
        self.config = config_info
        self.model = model
        self.this_task:str = ""
        self.workspace = workspace
        self.chat_history: list[dict] = []
        self.llm_response: list[dict] = []
        self.current_cost = 0.0
        logger.info(f"Agent initialized with model: {model.model_name}")

    def get_template_variable(self, **kwargs) -> dict:
        return utils.recursive_merge(
            self.config.model_dump(),
            self.workspace.get_template_variable(),
            self.model.get_template_variable(),
            {"task":self.this_task}
        )

    def task_loop(self, task:str =""):
        #base version for a new task everytime just start over new message 
        self.chat_history = []
        self.this_task =  task
        logger.info(f"Starting task_loop with task: {task}")
        #add the first system message 
        self.chat_history.append({"role":"system","content":self.config.system_template})
        logger.info(f"Added system message: {self.config.system_template[:100]}...")
        #add second message , the first user message where the include the task 

        instance_message = Template(self.config.instance_template, undefined=StrictUndefined).render(**self.get_template_variable())
        self.chat_history.append({"role":"user","content":instance_message})
        logger.info(f"Added instance message: {instance_message[:100]}...")
        #main loop
        step_count = 0
        while 1:
            step_count += 1
            try:
                logger.info(f"Executing step {step_count}")
                self.step()
            except ex.Submitted as e:
                logger.info(f"Task submitted after {step_count} steps, returning results")
                #oh we get the loop end so we return from here
                return e.messages[0].get("extra",[])
            #later handle other exceptions 
    
    def step(self):
        logger.info("Step started")
        parsed_response = self.query_llm()
        logger.info(f"Parsed response: role={parsed_response.get('role')}, actions count={len(parsed_response.get('actions', []))}")
        execution_results = self.run_actions(parsed_response=parsed_response)
        if execution_results:
            action_list = parsed_response["actions"]
            logger.info(f"Executing {len(action_list)} actions, got {len(execution_results)} results")
            
            # Add assistant message with tool_calls to history before tool results
            assistant_message = {
                "role": parsed_response.get("role", "assistant"),
                "content": parsed_response.get("content") or "",
            }
            if parsed_response.get("tool_calls"):
                assistant_message["tool_calls"] = parsed_response["tool_calls"]
            self.chat_history.append(assistant_message)
            logger.info(f"Added assistant message with {len(parsed_response.get('tool_calls', []))} tool_calls to history")
            
            format_observation = self.model.format_toolcall_result_message(action_list, execution_results, self.model.config.observation_template)
            self.chat_history.extend(format_observation)
            logger.info(f"Extended chat history with {len(format_observation)} observation messages")

    def query_llm(self) -> dict:
        logger.debug(f"Querying LLM with {len(self.chat_history)} messages in history")
        llm_response = self.model.call_llm_api(self.chat_history)

        logger.debug(f"Received LLM response: {llm_response}")
        action_list = self.model.parse_actions(llm_response)

        logger.info(f"Parsed {len(action_list)} actions from LLM response")
        parsed_response = self.model.parse_response(llm_response)

        logger.info(f"New response with cost: {parsed_response.get('cost')}")
        self.current_cost += parsed_response.get('cost',0)
        parsed_response["actions"] = action_list
        
        logger.info(f"LLM response content: {str(parsed_response.get('content', ''))[:100]}...")
        return parsed_response
    
    def run_actions(self, parsed_response:dict) -> list[dict]:
        actions = parsed_response.get("actions",[])
        logger.info(f"Running {len(actions)} actions")
        if not actions:
            logger.info("No actions to run")
            return []
        results = [self.workspace.execute(action) for action in actions]
        logger.info(f"Action execution completed, got {len(results)} results")
        return results














