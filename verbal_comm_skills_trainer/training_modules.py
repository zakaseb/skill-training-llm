import json
import random
from model_wrapper import LLMWrapper

# Load training prompts and configuration
with open("config.json", "r") as f:
    config = json.load(f)


def get_impromptu_prompt():
    return random.choice(config.get("impromptu_prompts", []))


def get_storytelling_prompt():
    return random.choice(config.get("storytelling_prompts", []))


def get_conflict_resolution_prompt():
    return random.choice(config.get("conflict_resolution_prompts", []))


# Initialize LLM for evaluation
llm = LLMWrapper()


def evaluate_training(module, user_response):
    if module == "impromptu":
        instructions = "Critique the structure, fluency, and clarity of this impromptu speech."
    elif module == "storytelling":
        instructions = "Evaluate the narrative quality and engagement of the story."
    elif module == "conflict":
        instructions = "Assess the diplomatic and effective handling of the conflict scenario."
    else:
        return "Unknown module."

    prompt = (
            "You are a communication expert. " + instructions +
            " Here is the user's response: " + user_response
    )
    return llm.generate_response(prompt)
