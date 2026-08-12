#!/usr/bin/env python3
"""
Dataset C: contextual legitimacy under GDPR Art. 9.

Each item appears twice with the same disclosure in different contexts — a
Bundeskanzler and an Abteilungsleiter attending the same service as a declared
Protestant. The first should be preserved (public office, self-declaration,
press freedom under Art. 85), the second redacted. Nothing but the role
distinguishes them, so the pair isolates contextual reasoning from
pattern-matching on Art. 9 vocabulary.

Each sentence gets two passes:
  1. redaction, using the same system prompt as Datasets A and B, so results
     stay comparable with those experiments
  2. a free-text justification, which is the material for comparing the
     model's reasoning against the dataset's own rationale_de

    python evaluate_dataset_c.py                 # fine-tuned adapter
    python evaluate_dataset_c.py --base-only     # untuned baseline
"""

import argparse
import csv
import difflib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATA_DIR = Path.home() / "evaldatasets" / "dataset_c"

REQUIRED_COLUMNS = {"id", "item_id", "framing", "art9_category", "legitimacy_basis",
                    "expected_action", "expected_output", "rationale_de"}

REASON_PROMPT = (
    "Du bist ein Datenschutz-Experte. Beurteile, ob der folgende Satz aus "
    "Datenschutzsicht (DSGVO) unverändert veröffentlicht werden darf oder ob "
    "er redigiert werden muss. Antworte mit BEIBEHALTEN oder REDIGIEREN, "
    "gefolgt von einer kurzen Begründung in einem Satz."
)

# The default system prompt (carried in the JSONL, shared with Datasets A and B)
# never mentions legitimate exemptions, so a model that redacts everything is
# arguably following it. This variant names the exemptions explicitly, which
# separates "cannot weigh legitimacy" from "was never asked to".
LEGITIMACY_PROMPT = (
    "Du bist ein Datenschutz-Filter. Ersetze direkte Identifikatoren durch "
    "Platzhalter und formuliere Sätze so um, dass besondere Kategorien "
    "personenbezogener Daten (Gesundheit, ethnische Herkunft, Religion) nicht "
    "mehr ableitbar sind. Erfinde keine neuen Tatsachen.\n\n"
    "Nicht jede Offenlegung ist schützenswert. Gib den Satz unverändert "
    "zurück, wenn eine der folgenden Bedingungen zutrifft:\n"
    "- Die Person ist Amtsträgerin oder Amtsträger und handelt in dieser "
    "Funktion; es besteht ein berechtigtes öffentliches Interesse und die "
    "Berichterstattung ist durch Art. 85 DSGVO gedeckt.\n"
    "- Die betroffene Person hat die Information selbst öffentlich gemacht "
    "(Art. 9 Abs. 2 lit. e DSGVO).\n"
    "- Die Angabe ist dem öffentlichen Amt inhärent und ohne sie wird die "
    "Aussage unverständlich.\n\n"
    "Wenn nichts Schützenswertes offengelegt wird, gib den Satz unverändert "
    "zurück."
)

MAX_NEW_TOKENS = 200
UNCHANGED_THRESHOLD = 0.95


def load_items(data_dir):
    with open(data_dir / "dataset_c.csv", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"dataset_c.csv is missing columns: {sorted(missing)}")
        meta = {row["id"]: row for row in reader}

    items, unmatched, malformed = [], [], []
    with open(data_dir / "dataset_c_eval.jsonl", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(number)
                continue
            messages = record.get("messages") or []
            if (not record.get("id") or len(messages) < 2
                    or not isinstance(messages[1], dict)
                    or "content" not in messages[1]):
                malformed.append(number)
            elif record["id"] not in meta:
                unmatched.append(record["id"])
            else:
                items.append({"record": record, "meta": meta[record["id"]]})

    if malformed:
        print(f"warning: skipped {len(malformed)} malformed JSONL lines: {malformed[:3]}")
    if unmatched:
        print(f"warning: skipped {len(unmatched)} ids absent from the CSV: {unmatched[:3]}")
    if not items:
        raise SystemExit("No items matched between the JSONL and the CSV.")
    return items


@torch.no_grad()
def ask(model, tokenizer, device, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output = model.generate(
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0][encoded["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def classify(source, output, threshold):
    """Infer the model's decision from how much of the sentence it changed."""
    similarity = difflib.SequenceMatcher(None, source, output).ratio()
    return ("PRESERVE" if similarity >= threshold else "REDACT"), similarity


def stated_decision(reasoning):
    """
    Read BEIBEHALTEN / REDIGIEREN out of the justification.

    Takes whichever keyword comes first and respects negation, so "nicht
    redigieren" is read as PRESERVE rather than its opposite.
    """
    head = reasoning[:200].upper()
    found = []
    for keyword, decision in (("REDIGIEREN", "REDACT"), ("REDIGIERT", "REDACT"),
                              ("BEIBEHALTEN", "PRESERVE")):
        index = head.find(keyword)
        if index == -1:
            continue
        preceding = head[max(0, index - 20):index]
        if any(negation in preceding for negation in ("NICHT", "KEINE", "KEIN ")):
            decision = "PRESERVE" if decision == "REDACT" else "REDACT"
        found.append((index, decision))
    return min(found)[1] if found else "UNCLEAR"


def load_model(base_only, adapter, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loading straight onto MPS segfaults with recent safetensors builds, so
    # weights load on CPU and move afterwards. These two lines belong apart.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cpu")
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    if not base_only:
        if not adapter.is_dir():
            raise SystemExit(f"No adapter at {adapter}")
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tokenizer


def evaluate(model, tokenizer, device, items, threshold, legitimacy_prompt=False):
    results, failures = [], []
    for index, item in enumerate(items, 1):
        record, meta = item["record"], item["meta"]
        source = record["messages"][1]["content"]

        messages = list(record["messages"])
        if legitimacy_prompt:
            messages[0] = {"role": "system", "content": LEGITIMACY_PROMPT}

        try:
            redacted = ask(model, tokenizer, device, messages)
            reasoning = ask(model, tokenizer, device, [
                {"role": "system", "content": REASON_PROMPT},
                {"role": "user", "content": source},
            ])
        except Exception as error:
            print(f"  {index}/{len(items)}  {record['id']}: failed ({type(error).__name__})")
            failures.append({"id": record["id"], "error": str(error)})
            continue

        decision, similarity = classify(source, redacted, threshold)
        expected = meta["expected_action"]

        results.append({
            "id": record["id"],
            "item_id": meta["item_id"],
            "framing": meta["framing"],
            "art9_category": meta["art9_category"],
            "legitimacy_basis": meta["legitimacy_basis"],
            "expected_action": expected,
            "decision_from_output": decision,
            "decision_stated": stated_decision(reasoning),
            "correct": decision == expected,
            "similarity": round(similarity, 3),
            "input": source,
            "expected_output": meta["expected_output"],
            "prediction": redacted,
            "reasoning": reasoning,
            "rationale_gold": meta["rationale_de"],
        })
        print(f"  {index}/{len(items)}  {record['id']}: expected {expected}, got {decision}")

    return results, failures


def report(results):
    correct = sum(r["correct"] for r in results)
    print(f"\nDecision accuracy: {correct}/{len(results)}")

    for framing in sorted({r["framing"] for r in results}):
        subset = [r for r in results if r["framing"] == framing]
        print(f"  {framing:<20} {sum(r['correct'] for r in subset)}/{len(subset)}")

    agree = sum(1 for r in results if r["decision_stated"] == r["decision_from_output"])
    print(f"\nStated decision matches behaviour: {agree}/{len(results)}")
    print("  a gap means the model can state a rule it does not apply when filtering")

    pairs = {}
    for r in results:
        pairs.setdefault(r["item_id"], []).append(r)
    complete = [v for v in pairs.values() if len(v) == 2]
    if complete:
        solved = sum(1 for v in complete if all(i["correct"] for i in v))
        print(f"\nBoth sides of a pair correct: {solved}/{len(complete)}")
        print("  same disclosure, different context — the contrast this dataset tests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true",
                        help="evaluate the untuned model, without the adapter")
    parser.add_argument("--adapter", type=Path, default=Path("lora_outputs/final_model"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--threshold", type=float, default=UNCHANGED_THRESHOLD,
                        help="similarity above which an output counts as unchanged")
    parser.add_argument("--legitimacy-prompt", action="store_true",
                        help="use a system prompt that names the Art. 85 and "
                             "Art. 9(2)(e) exemptions explicitly")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    condition = "base" if args.base_only else "lora"
    if args.legitimacy_prompt:
        condition += "_legitimacy"
    out_dir = args.out or Path(f"dataset_c_{condition}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prompt_label = "legitimacy prompt" if args.legitimacy_prompt else "default prompt"
    print(f"{MODEL_ID} | {device} | "
          f"{'base model' if args.base_only else 'fine-tuned'} | {prompt_label}")

    model, tokenizer = load_model(args.base_only, args.adapter, device)
    items = load_items(args.data_dir)
    print(f"{len(items)} items\n")

    results, failures = evaluate(model, tokenizer, device, items,
                                 args.threshold, args.legitimacy_prompt)
    if not results:
        raise SystemExit("No items were evaluated successfully.")

    if failures:
        path = out_dir / "dataset_c_failures.json"
        path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{len(failures)} items failed and are excluded below: {path}")

    report(results)

    json_path = out_dir / "dataset_c_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "dataset_c_review.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{json_path}\n{csv_path}")


if __name__ == "__main__":
    main()
