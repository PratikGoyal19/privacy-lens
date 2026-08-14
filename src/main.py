
"""
Main evaluation pipeline for PrivacyLens.

Runs the configured LLMs on the test dataset using a short privacy
leak detection prompt.

The pipeline is resumable:

- already completed (model, sentence_id) pairs are skipped
- failed/invalid-JSON examples can be rerun
- increasing the test set from 120 to 150 does not rerun previous examples

Configuration:

- single-pass
- 150 test sentences
"""

import json
import csv
import os
import pandas as pd
import argparse

from models.load_model import load_model
from models.llm_client import generate_response
from prompts.privacy_prompt import SYSTEM_PROMPT


def load_test_data(file_path):
    return (
        pd.read_csv(file_path)
        .head(150)
        .to_dict(orient="records")
    )


def parse_json_response(response):
    """Parse JSON returned by the model."""

    response = response.strip()
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
        existing_df = pd.read_csv(
            predictions_file
        )

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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sentences",
        default="~/evaldatasets/dataset_a/sentences.csv",
        help="Path to Dataset A sentences.csv"
    )

    args = parser.parse_args()

    data = load_test_data(
        args.sentences
    )

   
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
        jsonl_file,
        "a",
        encoding="utf-8"
    ) as json_file, open(
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

        #
        for model_name in model_names:

            model_config = load_model(
                model_name
            )
            print(
                f"Using model: "
                f"{model_config['name']}"
            )

            for i, row in enumerate(
                data,
                start=1
            ):

                sentence_id = row["id"]
                sentence = row["text"]

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
                print(
                    f"[{i}/{len(data)}] "
                    f"{sentence}"
                )

                print(
                    " Sending request to Ollama",
                    flush=True
                )
                try:

                    response = generate_response(
                        model_config,
                        SYSTEM_PROMPT,
                        sentence,
                        response_format=None
                    )

                    print(
                        ">>> Response received",
                        flush=True
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


                if "has_leak" not in parsed_response:

                    print(
                        "WARNING: JSON does not "
                        "contain 'has_leak'."
                    )

                    continue

                if "output" not in parsed_response:

                    print(
                        "WARNING: JSON does not "
                        "contain 'output'."
                    )

                    continue

                predicted_sensitive = bool(
                    parsed_response["has_leak"]
                )

                result = {
                    "id": sentence_id,
                    "text": sentence,
                    "ground_truth": bool(
                        row["has_sensitive_attribute"]
                    ),
                    "model": model_config["name"],
                    "config": "single_pass_no_schema",
                    "response": parsed_response
                }

                json_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                csv_writer.writerow({

                    "sentence_id":
                        sentence_id,

                    "model":
                        model_config["name"],

                    "config":
                        "single_pass_no_schema",

                    "predicted_sensitive":
                        predicted_sensitive,

                    "output_text":
                        parsed_response["output"],

                    "num_llm_calls":
                        1
                })

                existing_results.add(
                    result_key
                )

                csv_file.flush()
                json_file.flush()

                print(
                    "Saved successfully."
                )

    print()
    print(
        "Experiment completed."
    )

    print(
        f"JSONL results: "
        f"{jsonl_file}"
    )

    print(
        f"Predictions:   "
        f"{predictions_file}"
    )


if __name__ == "__main__":
    main()

