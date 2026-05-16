from pydantic import BaseModel

from lang_graph_state.sources.customer_profile import CustomerProfileResult
from lang_graph_state.sources.insights import InsightsResult
from lang_graph_state.sources.monte_carlo import MonteCarloResult
from lang_graph_state.sources.optimizer import OptimizerResult
from lang_graph_state.synthesis.parsing import SynthesisOutput


class GraphState(BaseModel):
    fs_req_id: str
    customer_profile: CustomerProfileResult | None = None
    insights: InsightsResult | None = None
    optimizer: OptimizerResult | None = None
    monte_carlo: MonteCarloResult | None = None
    synthesis: SynthesisOutput | None = None
