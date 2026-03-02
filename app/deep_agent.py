from deepagents import create_deep_agent
from app.tools.data_tools import query_mongo_requests, query_sql_campaigns
from app.agents.prompts.ml_analist_prompt import SUBAGENT_DELEGATION_INSTRUCTIONS, ML_WORKFLOW_INSTRUCTIONS
from app.agents.ml_analyst_agent import ml_analyst_agent
from langchain_google_genai import ChatGoogleGenerativeAI


from utils import show_prompt


model = ChatGoogleGenerativeAI()
# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS = (
    ML_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    +  SUBAGENT_DELEGATION_INSTRUCTIONS
)

tools = [
    query_sql_campaigns,
    query_mongo_requests,
]

show_prompt(INSTRUCTIONS)

agent = create_deep_agent(
      model=model,
      tools=tools,
      system_prompt=INSTRUCTIONS,
      subagents=[ml_analyst_agent]
)