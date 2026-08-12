#!/usr/bin/env python3
"""
LoRA fine-tuning of Llama 3.2 3B for German privacy redaction.

Trains on Dataset B (minimal-pair training set) and generates redactions for
Dataset A, which is held out and scored separately by score_predictions.py.

    python lora_training.py                 # train and evaluate
    python lora_training.py --base-only     # baseline: no adapter, no training
    python lora_training.py --seed 1        # repeat with a different seed
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "dataset_b_train.jsonl"
DEV_PATH = REPO_ROOT / "data" / "dataset_b_dev.jsonl"
EVAL_PATH = Path.home() / "evaldatasets" / "dataset_a" / "sentences.csv"

SYSTEM_PROMPT = (
    "Du bist ein Datenschutz-Filter. Ersetze direkte Identifikatoren durch Platzhalter "
    "und formuliere Sätze so um, dass besondere Kategorien personenbezogener Daten "
    "(Gesundheit, ethnische Herkunft, Religion) nicht mehr ableitbar sind. Erfinde keine "
    "neuen Tatsachen. Wenn nichts Schützenswertes offengelegt wird, gib den Satz "
    "unverändert zurück."
)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

LEARNING_RATE = 1e-4
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
WARMUP_STEPS = 10
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 256
DEFAULT_SEED = 42


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_record(record):
    """Pull (system, user, assistant) out of one Dataset B row."""
    roles = {m["role"]: m["content"] for m in record["messages"]}
    return roles.get("system") or SYSTEM_PROMPT, roles["user"], roles["assistant"]


def build_example(tokenizer, system, user, assistant):
    """Tokenise one training pair, masking the prompt so loss covers only the
    redacted output."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + assistant + tokenizer.eos_token,
                         add_special_tokens=False)["input_ids"][:MAX_LENGTH]

    # Mask the shared prefix rather than assuming the prompt tokenises
    # identically inside the longer string; some tokenizers shift at the join.
    labels = list(full_ids)
    for i, (a, b) in enumerate(zip(prompt_ids, full_ids)):
        if a != b:
            break
        labels[i] = -100

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def build_dataset(tokenizer, records):
    examples, too_long = [], []
    for record in records:
        system, user, assistant = parse_record(record)
        example = build_example(tokenizer, system, user, assistant)
        labels = example["labels"]

        if not any(label == -100 for label in labels):
            # The prompt should always tokenise as a prefix of the full
            # sequence. Nothing masked means it diverged at the first token,
            # which would train the model on its own instructions.
            raise RuntimeError(
                f"prompt masking failed for: {user[:60]}...\n"
                "The tokenizer produced different ids for the prompt in "
                "isolation and in context."
            )
        if all(label == -100 for label in labels):
            too_long.append(user[:60])
        else:
            examples.append(example)

    if too_long:
        print(f"warning: {len(too_long)} examples exceed MAX_LENGTH={MAX_LENGTH} and were dropped")
        for text in too_long[:3]:
            print(f"  {text}...")
    if not examples:
        sys.exit("No usable training examples; check MAX_LENGTH and the input files.")
    return Dataset.from_list(examples)


class Collator:
    """Pads a batch, using -100 for label padding so it is ignored by the loss."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        width = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = width - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def load_eval_sentences(path, limit=None):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    sentences = [{"id": r["id"], "sentence": r["text"]} for r in rows if r["text"].strip()]
    return sentences[:limit] if limit else sentences


@torch.no_grad()
def generate(model, tokenizer, items, device):
    model.eval()
    predictions = []
    for i, item in enumerate(items, 1):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": item["sentence"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(output[0][encoded["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        predictions.append({"id": item["id"], "input": item["sentence"], "prediction": text})

        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)
    return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true",
                        help="generate with the base model, no training or adapter")
    parser.add_argument("--eval-only", action="store_true",
                        help="reuse the adapter already in the output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="8 rows, 1 epoch, 5 eval sentences")
    args = parser.parse_args()
    if args.base_only and args.eval_only:
        parser.error("--base-only and --eval-only cannot be combined")
    return args


def output_dir_for(args):
    if args.out:
        return args.out
    if args.base_only:
        return Path("baseline_outputs")
    if args.seed != DEFAULT_SEED:
        return Path(f"lora_outputs_seed{args.seed}")
    return Path("lora_outputs")


def main():
    args = parse_args()
    # Seeds Python, NumPy and torch together; LoRA init and data order all
    # depend on these, so the seed comparison is only meaningful with all set.
    set_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    out_dir = output_dir_for(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "final_model"

    mode = "base model" if args.base_only else f"LoRA (seed {args.seed})"
    print(f"{MODEL_ID} | {device} | {dtype} | {mode} -> {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loading directly onto MPS segfaults with recent safetensors builds, so
    # weights are loaded on CPU and moved afterwards.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map="cpu")
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.eval_only:
        if not adapter_dir.is_dir():
            sys.exit(f"No adapter found at {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

    elif not args.base_only:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
        ))
        model.print_trainable_parameters()

        train_records = load_jsonl(TRAIN_PATH)
        dev_records = load_jsonl(DEV_PATH)
        if args.smoke:
            train_records, dev_records = train_records[:8], dev_records[:4]
        print(f"train {len(train_records)} | dev {len(dev_records)}")

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(out_dir / "checkpoints"),
                num_train_epochs=1 if args.smoke else EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                learning_rate=LEARNING_RATE,
                lr_scheduler_type="cosine",
                warmup_steps=0 if args.smoke else WARMUP_STEPS,
                logging_steps=1,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                report_to=[],
                seed=args.seed,
            ),
            train_dataset=build_dataset(tokenizer, train_records),
            eval_dataset=build_dataset(tokenizer, dev_records),
            data_collator=Collator(tokenizer.pad_token_id),
        )

        start = time.time()
        trainer.train()
        print(f"trained in {(time.time() - start) / 60:.1f} min")

        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        metrics = trainer.evaluate()
        (out_dir / "dev_metrics.json").write_text(json.dumps(metrics, indent=2),
                                                  encoding="utf-8")
        print(f"dev loss {metrics['eval_loss']:.4f}")

    items = load_eval_sentences(EVAL_PATH, limit=5 if args.smoke else None)
    print(f"generating for {len(items)} sentences")
    predictions = generate(model, tokenizer, items, device)

    out_path = out_dir / "dataset_a_predictions.json"
    out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
