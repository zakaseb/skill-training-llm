import gradio as gr
from model_wrapper import LLMWrapper

# Initialize LLMWrapper instance once to optimize pre-loading
llm = LLMWrapper()

def chat_response(user_input):
    prompt = (
        "You are a communication expert. Provide detailed feedback on clarity, tone, "
        "and suggestions for improvement. Here is the user's input: " + user_input
    )
    return llm.generate_response(prompt)

# Gradio interface for text-based chat
iface = gr.Interface(
    fn=chat_response,
    inputs=gr.Textbox(lines=4, placeholder="Enter your message here..."),
    outputs="text",
    title="Verbal Communication Skills Trainer - Chat",
    description="Enter your message to receive expert communication feedback."
)

if __name__ == "__main__":
    iface.launch()
