from training_modules import get_impromptu_prompt, evaluate_training

def test_impromptu_prompt():
    prompt = get_impromptu_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0

def test_evaluate_training():
    feedback = evaluate_training("impromptu", "I am nervous.")
    assert isinstance(feedback, str)
    assert "feedback" in feedback.lower() or len(feedback) > 10
