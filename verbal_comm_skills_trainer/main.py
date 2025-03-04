import gradio as gr
from chat_interface import chat_response
from voice_interface import process_audio
import training_modules
import presentation_assessment


def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Verbal Communication Skills Trainer")

        with gr.Tab("Chat"):
            chat_input = gr.Textbox(label="Enter your message", lines=4,
                                    placeholder="I feel nervous about public speaking...")
            chat_output = gr.Textbox(label="Feedback")
            chat_btn = gr.Button("Submit")
            chat_btn.click(fn=chat_response, inputs=chat_input, outputs=chat_output)

        with gr.Tab("Voice"):
            voice_demo = gr.Interface(
                fn=process_audio,
                inputs=gr.Audio(source="upload", type="filepath"),
                outputs=[gr.Textbox(label="Feedback Text"), gr.Audio(label="Feedback Audio")],
                title="Voice Feedback"
            )
            voice_demo.render()

        with gr.Tab("Training Modules"):
            gr.Markdown("### Impromptu Speaking")
            impromptu_prompt = gr.Textbox(label="Prompt", value=training_modules.get_impromptu_prompt(),
                                          interactive=False)
            impromptu_response = gr.Textbox(label="Your Response",
                                            placeholder="Record your impromptu speech response here...")
            impromptu_btn = gr.Button("Submit Impromptu Response")
            impromptu_feedback = gr.Textbox(label="Feedback")
            impromptu_btn.click(fn=lambda resp: training_modules.evaluate_training("impromptu", resp),
                                inputs=impromptu_response, outputs=impromptu_feedback)

            gr.Markdown("### Storytelling")
            storytelling_prompt = gr.Textbox(label="Prompt", value=training_modules.get_storytelling_prompt(),
                                             interactive=False)
            storytelling_response = gr.Textbox(label="Your Story", placeholder="Tell your story here...")
            storytelling_btn = gr.Button("Submit Story")
            storytelling_feedback = gr.Textbox(label="Feedback")
            storytelling_btn.click(fn=lambda resp: training_modules.evaluate_training("storytelling", resp),
                                   inputs=storytelling_response, outputs=storytelling_feedback)

            gr.Markdown("### Conflict Resolution")
            conflict_prompt = gr.Textbox(label="Prompt", value=training_modules.get_conflict_resolution_prompt(),
                                         interactive=False)
            conflict_response = gr.Textbox(label="Your Response",
                                           placeholder="Describe how you would handle this scenario...")
            conflict_btn = gr.Button("Submit Conflict Resolution Response")
            conflict_feedback = gr.Textbox(label="Feedback")
            conflict_btn.click(fn=lambda resp: training_modules.evaluate_training("conflict", resp),
                               inputs=conflict_response, outputs=conflict_feedback)

        with gr.Tab("Presentation Assessment"):
            pres_input = gr.Textbox(label="Enter your presentation script", lines=6,
                                    placeholder="Paste your presentation text here...")
            pres_btn = gr.Button("Assess Presentation")
            pres_output = gr.Textbox(label="Assessment Feedback")
            pres_btn.click(fn=presentation_assessment.assess_presentation, inputs=pres_input, outputs=pres_output)

    demo.launch()


if __name__ == "__main__":
    launch_app()
