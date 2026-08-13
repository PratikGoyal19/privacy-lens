#!/usr/bin/env python3
"""
Convert Dataset A predictions from the fine-tuning runs into the
predictions.csv format score.py reads, so the fine-tuned model and the
prompted models are scored by the same code.

    python to_predictions_a.py \
        baseline_outputs/dataset_a_predictions.json:llama3.2-base:zero_shot \
        lora_outputs/dataset_a_predictions.json:finetuned-llama3.2:finetuned_seed42 \
        lora_outputs_seed1/dataset_a_predictions.json:finetuned-llama3.2:finetuned_seed1 \
        lora_outputs_seed2/dataset_a_predictions.json:finetuned-llama3.2:finetuned_seed2 \
        --out predictions_a_finetuned.csv

Each argument is path:model:config.

No predicted_sensitive column is written. The fine-tuned model was trained to
rewrite sentences, not to classify them, so it has no sensitivity flag to
report, and deriving one from whether the text changed would give a
meaningless precision and a recall of 1.000. score.py treats the missing
column as not applicable and scores these rows on leak rate and cost, both of
which come from the output text and are directly comparable with the prompted
models.
"""

import argparse
import csv
import json
from pathlib import Path

FIELDS = ["sentence_id", "model", "config", "output_text", "num_llm_calls"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="path:model:config")
    parser.add_argument("--out", type=Path, default=Path("predictions_a_finetuned.csv"))
    parser.add_argument("--calls", type=int, default=1,
                        help="LLM calls per sentence (default 1, single pass)")
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
                "output_text": result["prediction"],
                "num_llm_calls": args.calls,
            })
        print(f"{path}: {len(results)} rows as {model}/{config}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {args.out}")
    print("no predicted_sensitive column, so precision/recall/F1 will read n/a")


if __name__ == "__main__":
    main()
