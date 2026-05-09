from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from services.lp_sensitivity import make_lp_sensitivity_tools
from state import RetirementPlanState

_SYSTEM = (
    "You are a retirement savings expert specializing in the accumulation phase — "
    "how a client builds wealth before retirement. "
    "You have tools to probe how the optimizer's recommendations change under different scenarios. "
    "Use them to surface the most important insights: what drives the allocation, which constraints are binding, "
    "and what levers would most improve the outcome. "
    "Be specific and quantitative. Return a structured analysis in 3-4 paragraphs."
)


def build_accumulation_node(model: ChatOpenAI | None = None) -> Callable[[RetirementPlanState], dict[str, Any]]:
    def run_accumulation_agent(state: RetirementPlanState) -> dict[str, Any]:
        _model = model or _default_model()
        tools = make_lp_sensitivity_tools(state.customer_profile)
        agent = create_react_agent(_model, tools, prompt=_SYSTEM)
        result = agent.invoke({"messages": [HumanMessage(content=_build_prompt(state))]})
        return {"accumulation_analysis": result["messages"][-1].content}

    return run_accumulation_agent


def _default_model() -> ChatOpenAI:
    from services.llm import build_chat_model
    return build_chat_model()


def _build_prompt(state: RetirementPlanState) -> str:
    p = state.customer_profile
    alloc = state.contribution_allocation

    pre50 = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.pre50.items() if v > 0)
    post50 = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.post50.items() if v > 0)

    binding = []
    if state.optimization_diagnostics:
        binding = [
            c.name for c in state.optimization_diagnostics.constraints
            if c.binding and c.category != "variable_lower_bound"
        ]

    return f"""
Client: age {p.age}, retiring at {p.retirement_age}
Income: ${p.annual_income:,.0f}/yr, expenses: ${p.annual_expenses:,.0f}/yr, savings capacity: ${p.savings_capacity:,.0f}/yr
Employer match: {p.employer_match_rate:.0%} up to {p.employer_match_cap:.0%} of salary | HDHP enrolled: {p.hdhp_enrolled}
Tax rates: {p.current_marginal_tax_rate:.0%} now → {p.assumed_retirement_marginal_tax_rate:.0%} in retirement

LP allocation (pre-50): {pre50 or 'none'}
LP allocation (post-50 / catch-up): {post50 or 'none'}
Projected after-tax wealth: ${state.projected_wealth:,.0f}
Binding constraints: {', '.join(binding) or 'none identified'}

Analyze this accumulation strategy. Use your tools to probe 2-3 key scenarios that would most change the outcome.
""".strip()
