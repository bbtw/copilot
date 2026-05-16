import httpx
from pydantic import BaseModel


class SynthesisOutput(BaseModel):
    pass


def parse_synthesis_response(response: httpx.Response) -> SynthesisOutput:
    return SynthesisOutput()
