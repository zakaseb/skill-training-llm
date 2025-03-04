from model_wrapper import LLMWrapper


def assess_presentation(input_text):
    # Pre-process the text (e.g., remove filler words)
    processed_text = input_text.lower().replace("um", "").replace("uh", "")

    prompt = (
            "Analyze the following presentation for structure (introduction, body, conclusion), "
            "delivery (pacing, filler words), and content (persuasiveness, vocabulary). "
            "Provide scores and actionable tips. Presentation text: " + processed_text
    )

    llm = LLMWrapper()
    return llm.generate_response(prompt)
