from langchain_experimental.plan_and_execute.agent_executor import (
    PlanAndExecute,
    load_chat_planner,
    load_agent_executor
)
from app.agents.prompts.deep_router_prompt import DEEP_ROUTER_PROMPT
from app.agents.prompts.system_rules import SYSTEM_RULES

def build_deep_router_agent(llm, tools):
    planner = load_chat_planner(
        llm,
        system_prompt=SYSTEM_RULES + "\n" + DEEP_ROUTER_PROMPT
    )

    executor = load_agent_executor(
        llm=llm,
        tools=tools,
        verbose=True
    )

    return PlanAndExecute(planner=planner, executor=executor)
