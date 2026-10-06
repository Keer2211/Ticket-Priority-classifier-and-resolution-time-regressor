"""LLM integration points for support-ticket workflows."""


def generate_response(prompt: str) -> str:
    """Generate a response using the configured LLM provider."""
    raise NotImplementedError("Configure an LLM provider before generating responses.")
