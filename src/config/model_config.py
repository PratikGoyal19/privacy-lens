'''
Model configuration for the PrivacyLens evaluation.
Defines the four LLMs used in the experiment and their
generation settings. Each model is assigned a short internal 
name and its corresponding Ollama model identifier.
Temperature is set to 0.0 for all models to make generation
deterministic and reduce variation between runs.
Models:
- Llama 3.2 3B
- Qwen 2.5 3B
- Ministral 3 3B
- Gemma 3 4B
'''

models = {
    "llama3.2": {
        "name": "llama3.2",
        "temperature": 0.0,
    },

    "qwen2.5": {
        "name": "qwen2.5:7b",
        "temperature": 0.0,
    },

    "mistral": {
        "name": "mistral",
        "temperature": 0.0,
    },

    "deepseek": {
        "name": "deepseek-r1:8b",
        "temperature": 0.0,
    },
}