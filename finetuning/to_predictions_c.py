#!/usr/bin/env python3
"""
Convert dataset_c_results.json into the predictions_c.csv format score_c.py
expects, so the four conditions can be scored with the shared scorer.

    python to_predictions_c.py \
        dataset_c_base/dataset_c_results.json:base:default \
        dataset_c_lora/dataset_c_results.json:lora:default \
        dataset_c_base_legitimacy/dataset_c_results.json:base:legitimacy \
        dataset_c_lora_legitimacy/dataset_c_results.json:lora:legitimacy \
        --out predictions_c.csv

Each argument is path:model:config.
"""

import argparse
import csv
import json
import re
from pathlib import Path

PLACEHOLDER = re.compile(r"\[[A-ZÄÖÜ_]+\]")


def infer_action(source, output):
    """
    Map our free-text output onto main.py's four-way action field.

    none             output is the input unchanged
    mask             a placeholder was substituted, the rest left intact
    rewrite          wording changed but no placeholder was introduced
    mask_and_rewrite both

    Whether the sentence was rewritten is judged by how much of the source
    survives, not by which words went missing: the words a placeholder
    replaces would otherwise make every mask look like a rewrite too.
    """
    if output.strip() == source.strip():
        return "none"

    masked = bool(PLACEHOLDER.search(output))
    stripped = PLACEHOLDER.sub(" ", output)
    source_words = re.findall(r"\w+", source.lower())
    output_words = set(re.findall(r"\w+", stripped.lower()))

    survived = sum(1 for word in source_words if word in output_words)
    retention = survived / len(source_words) if source_words else 1.0

    # A masked name typically costs two tokens out of a dozen or more, so a
    # pure mask retains most of the sentence; a genuine rewrite does not.
    rewritten = retention < 0.75

    if masked and rewritten:
        return "mask_and_rewrite"
    if masked:
        return "mask"
    return "rewrite"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="path:model:config")
    parser.add_argument("--out", type=Path, default=Path("predictions_c.csv"))
    args = parser.parse_args()

    rows = []
    for source in args.sources:
        path, model, config = source.rsplit(":", 2)
        results = json.loads(Path(path).read_text(encoding="utf-8"))
        for result in results:
            rows.append({
                "sentence_id": result["id"],
                "model": model,
                "config": config,
                "predicted_action": infer_action(result["input"], result["prediction"]),
                "output_text": result["prediction"],
                "num_llm_calls": 2,  # redaction pass plus justification pass
            })
        print(f"{path}: {len(results)} rows as {model}/{config}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence_id", "model", "config",
                                               "predicted_action", "output_text",
                                               "num_llm_calls"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
