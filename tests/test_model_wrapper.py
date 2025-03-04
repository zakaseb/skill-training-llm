import pytest
from model_wrapper import LLMWrapper

def test_generate_response():
    llm = LLMWrapper()
    response = llm.generate_response("Test prompt: Provide a brief feedback.")
    assert isinstance(response, str)
    assert len(response) > 0
