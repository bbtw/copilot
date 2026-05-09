from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from services.mc_sensitivity import make_mc_sensitivity_tools
from state import RetirementPlanState

_SYSTEM = (
    "You are a retirement income expert specializing in decumulation — "
    "how a client's portfolio funds retirement expenses after they stop working. "
    "You have tools to probe how the confidence score and wealth distribution change under different scenarios. "
    "Use them to assess sustainability: sequence-of-returns risk, expense sensitivity, and portfolio longevity. "
    "Be specific and quantitative. Return a structured analysis in 3-4 paragraphs."
)


def build_withdrawal_node(model: ChatOpenAI | None = None) -> Callable[[RetirementPlanState], dict[str, Any]]:
    def run_withdrawal_agent(state: RetirementPlanState) -> dict[str, Any]:
        _model = model or _default_model()
        tools = make_mc_sensitivity_tools(state.customer_profile, state.contribution_allocation)
        agent = create_react_agent(_model, tools, prompt=_SYSTEM)
        result = agent.invoke({"messages": [HumanMessage(content=_build_prompt(state))]})
        return {"withdrawal_analysis": result["messages"][-1].content}

    return run_withdrawal_agent


def _default_model() -> ChatOpenAI:
    from services.llm import build_chat_model
    return build_chat_model()


def _build_prompt(state: RetirementPlanState) -> str:
    p = state.customer_profile
    dist = state.wealth_distribution
    net_withdrawal = max(0.0, p.retirement_annual_expenses - p.expected_retirement_income)

    return f"""
Client: age {p.age}, retiring at {p.retirement_age}, planning {p.retirement_years_to_plan} retirement years
Retirement annual expenses: ${p.retirement_annual_expenses:,.0f}/yr
Expected retirement income (Social Security, pension, etc.): ${p.expected_retirement_income:,.0f}/yr
Net annual portfolio withdrawal needed: ${net_withdrawal:,.0f}/yr

Monte Carlo result (1,000 paths, today's dollars):
  p10=${dist.p10:,.0f} / p50=${dist.p50:,.0f} / p90=${dist.p90:,.0f}
  Confidence score: {state.confidence_score:.1%} ({state.confidence_band})

Analyze the withdrawal sustainability of this plan. Use your tools to probe 2-3 scenarios
(e.g., higher expenses, longer retirement, different withdrawal rates) to assess the plan's resilience.
""".strip()
