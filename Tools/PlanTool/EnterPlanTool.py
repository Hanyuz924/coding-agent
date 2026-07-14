from typing import Any, Dict

from Tools.BaseTool import BaseTool, ToolResult, AgentRunContext, logger
from enter_plan_tool_prompt import ENTER_PLAN_MODE_TOOL_NAME, get_enter_plan_mode_tool_prompt


class EnterPlanMode(BaseTool):

    @property
    def name(self) -> str:
        return ENTER_PLAN_MODE_TOOL_NAME
    
    @property
    def description(self) -> str:
        return get_enter_plan_mode_tool_prompt()
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type":"object",
            "properties":{

            }
        }
    
    