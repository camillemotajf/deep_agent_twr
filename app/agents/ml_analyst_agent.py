# from deepagents import create_sub_agent
from app.agents.prompts.ml_analist_prompt import ML_ANALYST_INSTRUCTIONS, ML_OPERATOR_INSTRUCTIONS
from app.tools.ml_tool import *


ml_analyst_tools = [
    summarize_misclassifications,
    find_low_trust_samples,
    analyze_user_agent_patterns,
    plot_ml_diagnostics,
    plot_risk_score_distribution,
]


# ml_analyst_agent = create_sub_agent(
#     name="ml-analyst-agent",
#     system_prompt=ML_ANALYST_INSTRUCTIONS,
#     tools = [ml_analyst_tools]
# )

ml_analyst_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": ML_ANALYST_INSTRUCTIONS,
    "tools": [ml_analyst_tools],
}


