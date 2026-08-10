"""
LLM client for interacting with Ollama models.
Sends a system prompt and user prompt to the configured model,
applies the model's generation settings, and optionally enforces
a structured response format such as a JSON schema.
Returns the generated response as a string.
"""

import ollama


def generate_response(
    model_config,
    system_prompt,
    user_prompt,
    response_format=None
):
    response = ollama.chat(
        model=model_config["name"],
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        options={
            "temperature": model_config["temperature"],
        },
        format=response_format
    )

    return response["message"]["content"]