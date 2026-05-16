class SourceFetchError(Exception):
    def __init__(self, source: str, cause: Exception) -> None:
        super().__init__(f"Failed to fetch {source}: {cause}")
        self.source = source
        self.cause = cause


class SynthesisError(Exception):
    def __init__(self, cause: Exception) -> None:
        super().__init__(f"Synthesis failed: {cause}")
        self.cause = cause
