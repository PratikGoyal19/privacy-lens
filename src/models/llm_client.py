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
            "temperature": model_config["temperature"]
        }
    }
    if response_format is not None:
        kwargs["format"] = response_format

    response = ollama.chat(**kwargs)
    return response["message"]["content"]