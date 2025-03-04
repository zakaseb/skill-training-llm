# Skill Training Application



## Overview

This project is a verbal communication skills trainer built as a wrapper around an open‑source LLM. It supports text chat and voice interactions, offers three training modules (Impromptu Speaking, Storytelling, Conflict Resolution), and provides presentation assessments with detailed, actionable feedback. The design emphasizes modularity, performance (via quantization, caching, and batch inference), and ease of deployment.

## Model Selection

I chose [Mistral‑7B](https://huggingface.co/mistralai/Mistral-7B) for its strong performance in language understanding and generation while still being efficient enough for local or GPU‑accelerated inference. The model is loaded using Hugging Face Transformers with 4‑bit quantization (via bitsandbytes) to reduce memory usage.

## Features

- **Chat Interface:** Text-based conversational coaching.
- **Voice Interface:** Speech-to‑text using OpenAI’s Whisper and text‑to‑speech using a TTS engine (gTTS is used as a placeholder; you can replace it with Mozilla TTS for a fully open‑source solution).
- **Training Modules:**  
  - *Impromptu Speaking* – Random prompts and performance evaluation.
  - *Storytelling* – Narrative evaluation.
  - *Conflict Resolution* – Scenario simulation and feedback.
- **Presentation Assessment:** Analyze a presentation script (or transcribed audio) for structure, delivery, and content.
- **Robustness & Optimization:**  
  - Quantization, LRU caching for prompt responses, and batching.
  - Extensible design for swapping models via configuration.
  - Error handling (e.g., OOM, inference timeouts) with fallbacks.



## Project Structure

skill-training-llm/verbal_comm_skills_trainer/ \
├── README.md\
├── requirements.txt\
├── config.json\
├── main.py\
├── model_wrapper.py\
├── chat_interface.py\
├── voice_interface.py\
├── training_modules.py\
├── presentation_assessment.py\
└── tests/\
          ├── test_model_wrapper.py\
          ├── test_training_modules.py\
          └── test_integration.py


## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/zakaseb/skill-training-llm.git
   cd verbal_comm_skills_trainer

2. **Install Dependencies:**
```
pip install -r requirements.txt
```


3. **Download the Model:**

The model_wrapper.py script downloads the Mistral‑7B model automatically. Ensure you have sufficient disk space and, if available, a GPU.
4. **Run the Application:**
```
python main.py
```

5. **Testing:**
```
pytest tests/
```

