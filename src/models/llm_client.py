"""
LLM client for the PrivacyLens pipeline.

Sends system and user prompts to the configured Ollama model,
optionally applies a response format, and returns the generated
text response.
"""

import ollama

def generate_response(
    model_config,
    system_prompt,
    user_prompt,
    response_format=None
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    kwargs = {
        "model": model_config["name"],
        "messages": messages,
        "options": {
            "temperature": model_config["temperature"],
            "num_predict": 4000,

        },
         "think": False,
        "stream": False,
    }

    if response_format is not None:
        kwargs["format"] = response_format

    print("Calling ollama.chat()", flush=True)

    response = ollama.chat(**kwargs)

    message = response["message"]

    content = message.get("content", "").strip()

    print("RAW OLLAMA RESPONSE:", repr(response), flush=True)
    print("THINKING:", repr(message.get("thinking", "")), flush=True)
    print("CONTENT:", repr(content), flush=True)

    return content