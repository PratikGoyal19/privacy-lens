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
   "llama": {
        "name": "llama3.2:3b",
        "temperature": 0.0,
    },
    "qwen": {
        "name": "qwen2.5:3b",
        "temperature": 0.0,
    },
    "mistral": {
        "name": "ministral-3:3b",
        "temperature": 0.0,
    },
    "gemma": {
        "name": "gemma3:4b",
        "temperature": 0.0,
    }
}