"""
LLM client for the PrivacyLens pipeline.

Sends system and user prompts to the configured Ollama model,
optionally applies a response format, and returns the generated
text response.
"""

import json
import re

import ollama

# Reasoning models need room for their internal deliberation before the answer.
# deepseek-r1 restates its conclusion several times and often runs past a fixed
# budget, so it is left uncapped; the others are capped to stop llama3.2 from
# entering a repetition loop and hanging the run.
REASONING_BUDGET = -1
STANDARD_BUDGET = 4000

JSON_OBJECT = re.compile(r"\{[^{}]*\"action\"[^{}]*\}", re.DOTALL)


def recover_json(thinking):
    """
    Pull the answer out of a reasoning trace.

    When generation stops on length, `content` is empty but the model has
    usually already written the finished JSON inside its reasoning, often more
    than once. The last complete object is its final position, so that is the
    one taken.
    """
    if not thinking:
        return ""
    candidates = JSON_OBJECT.findall(thinking)
    for candidate in reversed(candidates):
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    return ""


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

    is_reasoning_model = "deepseek" in model_config["name"].lower()

    kwargs = {
        "model": model_config["name"],
        "messages": messages,
        "options": {
            "temperature": model_config["temperature"],
            "num_predict": REASONING_BUDGET if is_reasoning_model else STANDARD_BUDGET,
        },
        "think": False,
        "stream": False,
    }

    if response_format is not None:
        kwargs["format"] = response_format

    response = ollama.chat(**kwargs)
    message = response["message"]

    content = (message.get("content") or "").strip()

    if not content:
        content = recover_json(message.get("thinking") or "")
        if content:
            print("  recovered JSON from reasoning trace", flush=True)

    return content
