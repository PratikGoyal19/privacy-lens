'''
This script evaluates the four configured LLMs on Dataset C, which contains
20 German privacy cases covering public legitimate information, public
sensitive information, and private information.

The pipeline:

Loads data/dataset_c.csv as the input dataset.
Evaluates each sentence with Llama 3.2, Qwen2.5, Mistral, and DeepSeek-R1.
Uses PRIVACY_PROMPT for Llama 3.2, Qwen2.5, and Mistral, and
DEEPSEEK_PROMPT for DeepSeek-R1.
Parses and validates the model's JSON response.
Accepts four possible actions: none, mask, rewrite, and mask_and_rewrite.
Saves the predicted action and sanitized output to results/predictions_c1.csv.
Uses exactly one LLM call per sentence.
Supports resumable evaluation by skipping model-sentence pairs that have
already been processed.

The script is designed to compare how the four models distinguish between
information that should be preserved and information that should be redacted.
'''

import json
import csv
import os
import re
import pandas as pd

from models.load_model import load_model
from models.llm_client import generate_response
from prompts.privacy_prompt import PRIVACY_PROMPT, DEEPSEEK_PROMPT


def load_dataset_c(file_path):
    return (
        pd.read_csv(file_path)
        .to_dict(orient="records")
    )


def parse_json_response(response):

    response = response.strip()
    response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL
    ).strip()
    if response.startswith("```"):
        lines = response.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response = "\n".join(lines).strip()

    return json.loads(response)


def load_existing_results(predictions_file):
    existing_results = set()

    if not os.path.exists(predictions_file):
        return existing_results

    try:
        existing_df = pd.read_csv(predictions_file)

    except pd.errors.EmptyDataError:
        return existing_results

    for _, row in existing_df.iterrows():

        existing_results.add(
            (
                row["model"],
                row["sentence_id"]
            )
        )

    return existing_results


def main():
    model_names = [
    "llama3.2",
    "qwen2.5",
    "mistral",
    "deepseek"
]

    data = load_dataset_c(
        "data/dataset_c.csv"
    )

    predictions_file = (
        "results/predictions_c1.csv"
    )

    prediction_fields = [
        "sentence_id",
        "model",
        "config",
        "predicted_action",
        "output_text",
        "reason",
        "num_llm_calls"
    ]

    existing_results = load_existing_results(
        predictions_file
    )

    print(
        f"Existing completed results: "
        f"{len(existing_results)}"
    )

    csv_exists = os.path.exists(
        predictions_file
    )
    with open(
        predictions_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=prediction_fields
        )

        if (
            not csv_exists
            or os.path.getsize(predictions_file) == 0
        ):
            csv_writer.writeheader()

        for model_name in model_names:

            model_config = load_model(model_name)

            prompt = (
                DEEPSEEK_PROMPT
                if model_name == "deepseek"
                else PRIVACY_PROMPT
            )

            print()
            print("=" * 60)
            print(f"Using model: {model_config['name']}")
            print("=" * 60)

            for i, row in enumerate(data, start=1):

                sentence_id = row["id"]
                sentence = row["input_text"]

                result_key = (
                    model_config["name"],
                    sentence_id
                )

                if result_key in existing_results:
                    print(
                        f"[{i}/{len(data)}] "
                        f"SKIP {sentence_id} "
                        f"(already processed)"
                    )
                    continue

                print()
                print(f"[{i}/{len(data)}] {sentence_id}")
                print(sentence)

                print(
                    ">>> Calling Ollama...",
                    flush=True
                )

                try:
                    response = generate_response(
                        model_config,
                        prompt,
                        sentence,
                        response_format=None
                    )

                except Exception as e:
                    print(
                        f"ERROR on sentence "
                        f"{sentence_id}: {e}"
                    )
                    continue

                print(
                    "Model response:"
                )

                print(response)
                try:

                    parsed_response = (
                        parse_json_response(
                            response
                        )
                    )

                except json.JSONDecodeError:

                    print(
                        "WARNING: Model returned "
                        "invalid JSON."
                    )

                    print(
                        "Raw response:",
                        response
                    )

                    continue
                if "action" not in parsed_response:

                    print(
                        "WARNING: JSON does not "
                        "contain 'action'."
                    )

                    continue

                if "output" not in parsed_response:

                    print(
                        "WARNING: JSON does not "
                        "contain 'output'."
                    )

                    continue
                predicted_action = (
                    parsed_response["action"]
                )

                allowed_actions = {
                    "none",
                    "mask",
                    "rewrite",
                    "mask_and_rewrite"
                }

                if predicted_action not in allowed_actions:

                    print(
                        "WARNING: Invalid action: "
                        f"{predicted_action}"
                    )

                    continue

                
                output_text = parsed_response["output"]
                reason = parsed_response.get("reason", "")

                csv_writer.writerow({

                    "sentence_id":
                        sentence_id,

                    "model":
                        model_config["name"],

                    "config":
                        "single_pass",

                    "predicted_action":
                        predicted_action,

                    "output_text":
                        output_text,

                    "reason":
                        reason,
                    

                    "num_llm_calls":
                        1
                })

                existing_results.add(
                    result_key
                )

                csv_file.flush()

                print(
                    "Saved successfully."
                )
    print(
        f"Predictions: "
        f"{predictions_file}"
    )


if __name__ == "__main__":
    main()

