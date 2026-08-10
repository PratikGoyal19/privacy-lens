"""
Main evaluation pipeline for PrivacyLens.

Runs the configured LLMs on the test dataset using the privacy
leak detection prompt and structured JSON response schema.

For each model and test sentence, the pipeline:
- sends the sentence to the selected LLM
- detects whether a privacy leak is present
- identifies explicit and implicit leaks
- generates a masked or rewritten version
- stores the complete model response in JSONL format
- stores the required evaluation fields in CSV format

All configured models are evaluated on the same test set using
a single-pass configuration.

Outputs:
- results/all_models_main_results.jsonl
    Detailed model responses, including the input sentence,
    ground truth, model, configuration, and structured output.

- results/predictions.csv
    Evaluation-ready predictions containing the sentence ID,
    model, configuration, predicted sensitivity, output text,
    and number of LLM calls.

The generated predictions.csv can be passed to score.py to
calculate precision, recall, F1, leak rate, and average
LLM calls per sentence.
"""

import json
import csv
import pandas as pd

from models.load_model import load_model
from models.llm_client import generate_response
from prompts.privacy_prompt import SYSTEM_PROMPT


def load_test_data(file_path):
    return pd.read_csv(file_path).to_dict(orient="records")


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_leak": {
            "type": "boolean"
        },
        "has_explicit": {
            "type": "boolean"
        },
        "has_implicit": {
            "type": "boolean"
        },
        "implicit_attribute": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "health",
                    "ethnicity"
                ]
            }
        },
        "action": {
            "type": "string",
            "enum": [
                "none",
                "mask",
                "rewrite",
                "mask_and_rewrite"
            ]
        },
        "output": {
            "type": "string"
        }
    },
    "required": [
        "has_leak",
        "has_explicit",
        "has_implicit",
        "implicit_attribute",
        "action",
        "output"
    ]
}


def main():

    # All four models
    model_names = [
        "llama",
        "qwen",
        "mistral",
        "gemma",
    ]

    # Test set
    data = load_test_data(
        "data/sentences.csv"
    )

    # One output file for ALL models
    jsonl_file = (
        "results/all_models_main_results.jsonl"
    )

    predictions_file = (
        "results/predictions.csv"
    )

    prediction_fields = [
        "sentence_id",
        "model",
        "config",
        "predicted_sensitive",
        "output_text",
        "num_llm_calls"
    ]

    with open(
        jsonl_file,
        "w",
        encoding="utf-8"
    ) as json_file, open(
        predictions_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=prediction_fields
        )

        csv_writer.writeheader()

        # Run every model
        for model_name in model_names:

            model_config = load_model(model_name)

            print(
                f"\n=============================="
            )
            print(
                f"Using model: {model_config['name']}"
            )
            print(
                f"=============================="
            )

            # Run every sentence
            for i, row in enumerate(data, start=1):

                sentence = row["text"]

                print(
                    f"[{i}/{len(data)}] {sentence}"
                )

                response = generate_response(
                    model_config,
                    SYSTEM_PROMPT,
                    sentence,
                    response_format=RESPONSE_SCHEMA
                )

                print(
                    "Model response:",
                    response
                )

                # Parse JSON returned by model
                try:
                    parsed_response = json.loads(response)

                except json.JSONDecodeError:
                    print(
                        "WARNING: Model returned invalid JSON."
                    )

                    continue

                # Save detailed JSONL result
                result = {
                    "id": row["id"],
                    "text": sentence,
                    "ground_truth": bool(
                        row["has_sensitive_attribute"]
                    ),
                    "model": model_config["name"],
                    "config": "single_pass",
                    "response": parsed_response
                }

                json_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                # Save evaluation-format CSV row
                csv_writer.writerow({
                    "sentence_id": row["id"],
                    "model": model_config["name"],
                    "config": "single_pass",
                    "predicted_sensitive": parsed_response["has_leak"],
                    "output_text": parsed_response["output"],
                    "num_llm_calls": 1
                })

                # Make sure results are written immediately
                csv_file.flush()
                json_file.flush()

    print("\nExperiment completed.")
    print(
        f"JSONL results: {jsonl_file}"
    )
    print(
        f"Predictions:   {predictions_file}"
    )


if __name__ == "__main__":
    main()