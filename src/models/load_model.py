"""
Model loader for the PrivacyLens pipeline.
Loads a model configuration from model_config.py, validates the
requested model key, checks that the Ollama model is available,
and returns its configuration for inference.
"""

import ollama

from config.model_config import models

def load_model(model_key: str):
    if model_key not in models:
        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Available models: {list(models.keys())}"
        )

    config = models[model_key]
    model_name = config["name"]
    ollama.show(model_name)
    print(f"Using model: {model_name}")
    return config