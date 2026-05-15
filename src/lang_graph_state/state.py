from typing import Optional
from pydantic import BaseModel


# --- Source Models ---

class CustomerProfileResult(BaseModel):
    pass


class InsightsResult(BaseModel):
    pass


class OptimizerResult(BaseModel):
    pass


class MonteCarloResult(BaseModel):
    pass


# --- Synthesis Output ---

class SynthesisOutput(BaseModel):
    pass


# --- Graph State ---

class GraphState(BaseModel):
    fs_req_id: str
    customer_profile: Optional[CustomerProfileResult] = None
    insights: Optional[InsightsResult] = None
    optimizer: Optional[OptimizerResult] = None
    monte_carlo: Optional[MonteCarloResult] = None
    synthesis: Optional[SynthesisOutput] = None
