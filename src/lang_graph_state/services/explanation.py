import logging
from time import perf_counter

from lang_graph_state.domain.models import ConfidenceBand, OptimizationDiagnostics, PlanExplanationRequest
from lang_graph_state.services.llm import GatewayClient

logger = logging.getLogger(__name__)

_STANDARD_SYSTEM = (
    "You are a certified financial planner giving a client an overview of their retirement savings plan. "
    "Be clear, warm, and specific. Avoid jargon. Use dollar amounts and percentages where helpful. "
    "Keep the overview to 2-3 paragraphs covering what the plan recommends and why, at a summary level."
)

_SYNTHESIS_SYSTEM = (
    "You are a certified financial planner synthesizing a complete retirement plan explanation for a client. "
    "You will receive three expert analyses — a plan overview, an accumulation strategy analysis, and a "
    "withdrawal strategy analysis. Weave them into a single coherent 4-5 paragraph explanation. "
    "Be clear, warm, and specific. Avoid jargon. Do not repeat yourself across paragraphs. "
    "Shape the overall tone using the confidence band: high = emphasize strength and drivers; "
    "medium = balanced with clear uncertainty; low = honest about retirement-readiness risk, suggest levers."
)

_ACCUMULATION_SYSTEM = (
    "You are a retirement savings expert specializing in the accumulation phase: "
    "how a client builds wealth before retirement. Use the supplied optimizer "
    "result and sensitivity probes to explain what drives the allocation, which "
    "constraints matter, and what levers would most improve the outcome. "
    "Be specific and quantitative. Return a structured analysis in 3-4 paragraphs."
)

_WITHDRAWAL_SYSTEM = (
    "You are a retirement income expert specializing in decumulation: "
    "how a client's portfolio funds retirement expenses after they stop working. "
    "Use the supplied Monte Carlo result and sensitivity probes to assess "
    "sustainability, sequence-of-returns risk, expense sensitivity, and portfolio "
    "longevity. Be specific and quantitative. Return a structured analysis in 3-4 paragraphs."
)

_CONFIDENCE_GUIDANCE: dict[ConfidenceBand, str] = {
    "high": (
        "Explain the allocation primarily as the optimizer's selected strategy. "
        "Mention uncertainty briefly and emphasize the major drivers: employer match, HSA eligibility, "
        "tax treatment, and binding constraints."
    ),
    "medium": (
        "Explain the allocation and include a clear uncertainty paragraph. "
        "Describe the projection as plausible but sensitive to market and inflation paths. "
        "Encourage the client to revisit the plan periodically."
    ),
    "low": (
        "Do not present the plan as likely to meet the client's retirement goal. "
        "State that simulated paths often exhaust assets before the planned retirement horizon. "
        "Suggest levers: increasing savings, adjusting retirement age, reducing expenses, or advisor review. "
        "Avoid saying the allocation is bad; frame the issue as retirement-readiness risk."
    ),
}


class ExplanationService:
    def __init__(self, llm: GatewayClient | None = None) -> None:
        self._llm = llm or GatewayClient()

    def standard_analysis(self, request: PlanExplanationRequest) -> str:
        logger.info("Standard explanation prompt build start")
        prompt = self._build_standard_prompt(request)
        logger.info("Standard explanation prompt built chars=%s", len(prompt))
        started = perf_counter()
        result = self._llm.complete(prompt, system=_STANDARD_SYSTEM)
        logger.info("Standard explanation complete elapsed=%.2fs output_chars=%s", perf_counter() - started, len(result or ""))
        return result

    async def astandard_analysis(self, request: PlanExplanationRequest) -> str:
        logger.info("Standard explanation prompt build start")
        prompt = self._build_standard_prompt(request)
        logger.info("Standard explanation prompt built chars=%s", len(prompt))
        started = perf_counter()
        result = await self._llm.acomplete(prompt, system=_STANDARD_SYSTEM)
        logger.info("Standard explanation complete elapsed=%.2fs output_chars=%s", perf_counter() - started, len(result or ""))
        return result

    async def aaccumulation_analysis(self, request: PlanExplanationRequest, probe_results: list[str]) -> str:
        logger.info("Accumulation explanation prompt build start probe_count=%s", len(probe_results))
        prompt = self._build_accumulation_prompt(request, probe_results)
        logger.info("Accumulation explanation prompt built chars=%s", len(prompt))
        started = perf_counter()
        result = await self._llm.acomplete(prompt, system=_ACCUMULATION_SYSTEM, max_tokens=1536)
        logger.info("Accumulation explanation complete elapsed=%.2fs output_chars=%s", perf_counter() - started, len(result or ""))
        return result

    async def awithdrawal_analysis(self, request: PlanExplanationRequest, probe_results: list[str]) -> str:
        logger.info("Withdrawal explanation prompt build start probe_count=%s", len(probe_results))
        prompt = self._build_withdrawal_prompt(request, probe_results)
        logger.info("Withdrawal explanation prompt built chars=%s", len(prompt))
        started = perf_counter()
        result = await self._llm.acomplete(prompt, system=_WITHDRAWAL_SYSTEM, max_tokens=1536)
        logger.info("Withdrawal explanation complete elapsed=%.2fs output_chars=%s", perf_counter() - started, len(result or ""))
        return result

    def synthesize(
        self,
        standard: str,
        accumulation: str,
        withdrawal: str,
        confidence_band: ConfidenceBand,
    ) -> str:
        logger.info(
            "Synthesis prompt build start standard_chars=%s accumulation_chars=%s withdrawal_chars=%s confidence_band=%s",
            len(standard),
            len(accumulation),
            len(withdrawal),
            confidence_band,
        )
        prompt = self._build_synthesis_prompt(standard, accumulation, withdrawal, confidence_band)
        logger.info("Synthesis prompt built chars=%s", len(prompt))
        started = perf_counter()
        result = self._llm.complete(prompt, system=_SYNTHESIS_SYSTEM, max_tokens=1536)
        logger.info("Synthesis complete elapsed=%.2fs output_chars=%s", perf_counter() - started, len(result or ""))
        return result

    async def asynthesize(
        self,
        standard: str,
        accumulation: str,
        withdrawal: str,
        confidence_band: ConfidenceBand,
    ) -> str:
        logger.info(
            "Synthesis prompt build start standard_chars=%s accumulation_chars=%s withdrawal_chars=%s confidence_band=%s",
            len(standard),
            len(accumulation),
            len(withdrawal),
            confidence_band,
        )
        prompt = self._build_synthesis_prompt(standard, accumulation, withdrawal, confidence_band)
        logger.info("Synthesis prompt built chars=%s", len(prompt))
        started = perf_counter()
        result = await self._llm.acomplete(prompt, system=_SYNTHESIS_SYSTEM, max_tokens=1536)
        logger.info("Synthesis complete elapsed=%.2fs output_chars=%s", perf_counter() - started, len(result or ""))
        return result

    def _build_standard_prompt(self, request: PlanExplanationRequest) -> str:
        p = request.customer_profile
        alloc = request.contribution_allocation
        dist = request.wealth_distribution

        pre50_lines = "\n".join(
            f"  - {acct}: ${amt:,.0f}/year"
            for acct, amt in alloc.pre50.items() if amt > 0
        )
        post50_lines = "\n".join(
            f"  - {acct}: ${amt:,.0f}/year"
            for acct, amt in alloc.post50.items() if amt > 0
        )
        optimizer_lines = self._format_optimizer_diagnostics(request.optimization_diagnostics)

        return f"""
Customer profile:
- Age: {p.age}, retiring at {p.retirement_age}
- Annual income: ${p.annual_income:,.0f}, annual expenses: ${p.annual_expenses:,.0f}
- Savings capacity: ${p.savings_capacity:,.0f}/year
- Employer match: {p.employer_match_rate:.0%} up to {p.employer_match_cap:.0%} of salary
- HDHP enrolled: {p.hdhp_enrolled}
- Current marginal tax rate: {p.current_marginal_tax_rate:.0%}, retirement tax rate: {p.assumed_retirement_marginal_tax_rate:.0%}

Recommended annual contributions (before age 50):
{pre50_lines}

Recommended annual contributions (age 50+, with catch-up):
{post50_lines}

Projected after-tax wealth at retirement: ${request.projected_wealth:,.0f}
Monte Carlo percentiles (today's dollars): p10=${dist.p10:,.0f} / p50=${dist.p50:,.0f} / p90=${dist.p90:,.0f}
Confidence score: {request.confidence_score:.1%} ({request.confidence_band})

Optimizer diagnostics:
{optimizer_lines}

Give a plain-English overview of this retirement plan. Explain the major allocation choices and what drives them.
Do not imply guaranteed results. Do not include accumulation mechanics or withdrawal sustainability detail — those will be covered separately.
""".strip()

    def _build_accumulation_prompt(self, request: PlanExplanationRequest, probe_results: list[str]) -> str:
        p = request.customer_profile
        alloc = request.contribution_allocation
        pre50 = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.pre50.items() if v > 0)
        post50 = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.post50.items() if v > 0)
        binding = [
            c.name for c in request.optimization_diagnostics.constraints
            if c.binding and c.category != "variable_lower_bound"
        ] if request.optimization_diagnostics else []
        probes = "\n\n".join(probe_results) or "No sensitivity probes were supplied."
        return f"""
Client: age {p.age}, retiring at {p.retirement_age}
Income: ${p.annual_income:,.0f}/yr, expenses: ${p.annual_expenses:,.0f}/yr, savings capacity: ${p.savings_capacity:,.0f}/yr
Employer match: {p.employer_match_rate:.0%} up to {p.employer_match_cap:.0%} of salary
HDHP enrolled: {p.hdhp_enrolled}
Tax rates: {p.current_marginal_tax_rate:.0%} now, {p.assumed_retirement_marginal_tax_rate:.0%} in retirement

LP allocation before age 50: {pre50 or 'none'}
LP allocation age 50+ with catch-up: {post50 or 'none'}
Projected after-tax wealth: ${request.projected_wealth:,.0f}
Binding constraints: {', '.join(binding) or 'none identified'}

Sensitivity probes:
{probes}

Analyze this accumulation strategy using the optimizer result and probes.
""".strip()

    def _build_withdrawal_prompt(self, request: PlanExplanationRequest, probe_results: list[str]) -> str:
        p = request.customer_profile
        dist = request.wealth_distribution
        net_withdrawal = max(0.0, p.retirement_annual_expenses - p.expected_retirement_income)
        probes = "\n\n".join(probe_results) or "No sensitivity probes were supplied."
        return f"""
Client: age {p.age}, retiring at {p.retirement_age}, planning {p.retirement_years_to_plan} retirement years
Retirement annual expenses: ${p.retirement_annual_expenses:,.0f}/yr
Expected retirement income: ${p.expected_retirement_income:,.0f}/yr
Net annual portfolio withdrawal needed: ${net_withdrawal:,.0f}/yr

Monte Carlo result (1,000 paths, today's dollars):
p10=${dist.p10:,.0f} / p50=${dist.p50:,.0f} / p90=${dist.p90:,.0f}
Confidence score: {request.confidence_score:.1%} ({request.confidence_band})

Sensitivity probes:
{probes}

Analyze withdrawal sustainability using the Monte Carlo result and probes.
""".strip()

    def _build_synthesis_prompt(
        self,
        standard: str,
        accumulation: str,
        withdrawal: str,
        confidence_band: ConfidenceBand,
    ) -> str:
        guidance = _CONFIDENCE_GUIDANCE[confidence_band]
        return f"""
Confidence band: {confidence_band}
Confidence-band guidance: {guidance}

=== Plan Overview ===
{standard}

=== Accumulation Strategy Analysis ===
{accumulation}

=== Withdrawal Strategy Analysis ===
{withdrawal}

Synthesize the above into a single coherent client-facing explanation.
Do not present the raw confidence band label as a customer-facing judgment.
""".strip()

    def _format_optimizer_diagnostics(self, diagnostics: OptimizationDiagnostics | None) -> str:
        if not diagnostics:
            return "No optimizer diagnostics provided."

        binding = [c for c in diagnostics.constraints if c.binding and c.category != "variable_lower_bound"]
        selected = [v for v in diagnostics.variables if v.value > 1e-5]
        zero = [v for v in diagnostics.variables if v.at_lower_bound and (v.reduced_cost or 0) > 1e-5]

        binding_lines = "\n".join(
            f"  - {c.name}: slack ${c.slack:,.2f}, shadow price {c.dual_value if c.dual_value is not None else 'n/a'}"
            for c in binding[:8]
        ) or "  - None"
        selected_lines = "\n".join(
            f"  - {v.name}: ${v.value:,.0f}/year, terminal value per $1: {v.objective_coefficient:,.2f}"
            for v in selected[:12]
        ) or "  - None"
        zero_lines = "\n".join(
            f"  - {v.name}: reduced-cost signal {v.reduced_cost:,.2f}"
            for v in zero[:8]
        ) or "  - None"

        return (
            f"Selected variables:\n{selected_lines}\n"
            f"Binding constraints:\n{binding_lines}\n"
            f"Zero-selected with reduced cost:\n{zero_lines}"
        )
