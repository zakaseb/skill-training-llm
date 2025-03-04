from model_wrapper import LLMWrapper

def test_batch_generate():
    llm = LLMWrapper()
    prompts = ["Test prompt one.", "Test prompt two."]
    responses = llm.batch_generate(prompts)
    assert isinstance(responses, list)
    assert len(responses) == len(prompts)
