import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache
import json
import os

class LLMWrapper:
    def __init__(self, config_path="config.json"):
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)
        self.model_name = config.get("model_name", "mistralai/Mistral-7B")
        self.quantization = config.get("quantization", True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.load_model()

    def load_model(self):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                load_in_4bit=self.quantization,  # Enable 4-bit quantization if supported
                trust_remote_code=True
            )
            return model
        except Exception as e:
            print("Error loading model:", e)
            raise

    @lru_cache(maxsize=128)
    def generate_response(self, prompt, max_length=200, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        try:
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print("Error during generation:", e)
            return "Error generating response."

    def batch_generate(self, prompts, max_length=200, temperature=0.7):
        # Simple batching: process prompts sequentially (can be optimized further)
        responses = []
        for prompt in prompts:
            responses.append(self.generate_response(prompt, max_length, temperature))
        return responses

if __name__ == "__main__":
    # Quick test when running module directly
    llm = LLMWrapper()
    test_prompt = "You are a communication expert. Provide feedback on clarity, tone, and suggestions for improvement. Input: I feel nervous."
    print(llm.generate_response(test_prompt))
