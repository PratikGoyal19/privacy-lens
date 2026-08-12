#!/usr/bin/env python3
"""
Score Dataset A predictions against the ground-truth spans.

Dataset A is built from minimal pairs: each positive sentence has a negative
counterpart using the same cue vocabulary for a non-sensitive reason (a pet's
insulin rather than a person's). A model that redacts on keywords alone will
handle the positive and fail the negative, so pair accuracy is the headline
figure rather than sentence accuracy.

    python score_predictions.py
    python score_predictions.py --preds baseline_outputs/dataset_a_predictions.json
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

EVAL_DIR = Path.home() / "evaldatasets" / "dataset_a"
SENTENCES_PATH = EVAL_DIR / "sentences.csv"
SPANS_PATH = EVAL_DIR / "spans.csv"

# Nobiliary particles carry no identifying information on their own, so a
# surviving "von" is not treated as a retained name fragment. Matched only in
# lowercase: capitalised Van, Ben and Du are surnames in this dataset.
PARTICLES = {"von", "van", "de", "der", "den", "zu", "zur", "vom", "di", "del", "bin"}


def normalise(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).casefold().strip()


def contains(haystack, needle):
    """Whole-word containment, so Insulin does not match inside Insulinpumpe."""
    haystack, needle = normalise(haystack), normalise(needle)
    if not needle:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


def is_particle(token):
    return token[:1].islower() and normalise(token) in PARTICLES


def name_fragments(name):
    """Given and family names separately, so a surviving Marie from
    Marie-Antoinette still counts as a leak."""
    return [t for t in re.split(r"[\s\-–]+", name) if len(t) >= 2 and not is_particle(t)]


def load_sentences():
    rows = {}
    with open(SENTENCES_PATH, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["id"] in rows:
                sys.exit(f"Duplicate sentence id: {row['id']}")
            rows[row["id"]] = row
    return rows


def load_spans():
    spans = defaultdict(list)
    with open(SPANS_PATH, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            spans[row["sentence_id"]].append(row)
    return spans


def score_item(row, spans, prediction, degenerate_ratio):
    output = prediction["prediction"]
    source = row["text"]
    is_positive = row["polarity"] == "positive"

    persons = [s["text"] for s in spans if s["kind"] == "PERSON"]
    cues = [s["text"] for s in spans if s["kind"] == "IMPLICIT_CUE"]
    decoys = [s["text"] for s in spans if s["kind"] == "DECOY_CUE"]

    # Deleting the sentence would otherwise satisfy every "X is absent" test.
    length, source_length = len(normalise(output)), len(normalise(source))
    degenerate = length == 0 or (source_length > 0 and length < degenerate_ratio * source_length)

    identifier_removed = None
    if persons:
        identifier_removed = not any(
            contains(output, part) for name in persons for part in [name] + name_fragments(name)
        )

    # String-level test: a paraphrase that preserves the meaning without the
    # exact token counts as removed here. Checked by hand on a sample.
    leak_removed = None
    if is_positive and cues:
        leak_removed = not any(contains(output, cue) for cue in cues)

    # Negatives carry the same cue vocabulary for a non-sensitive reason, so
    # removing it is over-redaction rather than a success.
    decoy_kept = None
    if not is_positive and decoys:
        decoy_kept = all(contains(output, decoy) for decoy in decoys)

    checks = [c for c in (identifier_removed, leak_removed, decoy_kept) if c is not None]
    if degenerate:
        correct = False
    elif not checks:
        correct = None
    else:
        correct = all(checks)

    return {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "polarity": row["polarity"],
        "category": row["category"],
        "difficulty": row["difficulty"],
        "name_origin": row["name_origin"],
        "marked": row["name_origin_marked"].strip().lower() == "true",
        "degenerate": degenerate,
        "identifier_removed": identifier_removed,
        "leak_removed": leak_removed,
        "decoy_kept": decoy_kept,
        "correct": correct,
        "input": source,
        "gold": row["redacted_text"],
        "prediction": output,
    }


def rate(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def show(value, n):
    return "     n/a" if value is None else f"{value * 100:5.1f}%  (n={n})"


def mcnemar(pairs):
    """Exact McNemar test over (baseline_correct, variant_correct) pairs."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1))
    return b, c, min(1.0, 2 * tail / 2 ** n)


def report(results, label, degenerate_ratio):
    print(f"\n{'=' * 60}\n  Dataset A — {label}   ({len(results)} sentences)\n{'=' * 60}\n")

    excluded = [r for r in results if r["correct"] is None]
    degenerate = [r for r in results if r["degenerate"]]
    print("Coverage")
    print(f"  scored               {len(results) - len(excluded)} / {len(results)}")
    print(f"  no applicable spans  {len(excluded)}")
    print(f"  degenerate outputs   {len(degenerate)}")
    for item in degenerate[:5]:
        print(f"      {item['id']}: {len(item['prediction'])} vs {len(item['input'])} chars")
    if len(degenerate) > 5:
        print(f"      and {len(degenerate) - 5} more")

    value, n = rate([r["correct"] for r in results])
    print(f"\nSentence accuracy          {show(value, n)}")

    pairs = defaultdict(list)
    for r in results:
        if r["pair_id"]:
            pairs[r["pair_id"]].append(r)
    complete = {k: v for k, v in pairs.items()
                if len(v) >= 2 and all(i["correct"] is not None for i in v)}
    value, n = rate([all(i["correct"] for i in v) for v in complete.values()])
    print(f"Pair accuracy              {show(value, n)}")

    print("\nComponents")
    for name, key in [("identifier removed", "identifier_removed"),
                      ("leak removed", "leak_removed"),
                      ("decoy kept", "decoy_kept")]:
        value, n = rate([r[key] for r in results])
        print(f"  {name:<21}{show(value, n)}")

    print("\nBy polarity")
    for polarity in ("positive", "negative"):
        subset = [r for r in results if r["polarity"] == polarity]
        value, n = rate([r["correct"] for r in subset])
        print(f"  {polarity:<21}{show(value, n)}")

    for field, title in [("category", "By category"), ("difficulty", "By difficulty")]:
        print(f"\n{title}")
        for key in sorted({r[field] for r in results if r[field]}):
            subset = [r for r in results if r[field] == key]
            value, n = rate([r["correct"] for r in subset])
            print(f"  {key:<21}{show(value, n)}")

    print("\nName origin")
    for flag, name in [(False, "unmarked"), (True, "marked")]:
        subset = [r for r in results if r["marked"] == flag]
        value, n = rate([r["correct"] for r in subset])
        print(f"  {name:<21}{show(value, n)}")

    paired = []
    for group in complete.values():
        marked = [i for i in group if i["marked"]]
        unmarked = [i for i in group if not i["marked"]]
        if len(marked) == 1 and len(unmarked) == 1:
            paired.append((unmarked[0]["correct"], marked[0]["correct"]))
    if paired:
        b, c, p = mcnemar(paired)
        print(f"  McNemar b={b} c={c} p={p:.4f} over {len(paired)} pairs")
    else:
        print("  name marking does not vary within pairs; compare the rates above")

    over = [r for r in results if r["decoy_kept"] is False]
    print(f"\nOver-redaction: {len(over)} negatives lost a legitimate cue")
    print(f"Degenerate threshold: {degenerate_ratio:.0%} of source length\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, default=Path("lora_outputs/dataset_a_predictions.json"))
    parser.add_argument("--label", default="fine-tuned")
    parser.add_argument("--dump-errors", type=Path)
    parser.add_argument("--degenerate-ratio", type=float, default=0.33)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = load_sentences()
    spans = load_spans()
    predictions = json.loads(args.preds.read_text(encoding="utf-8"))

    ids = [str(p["id"]) for p in predictions]
    for prediction, pid in zip(predictions, ids):
        prediction["id"] = pid

    unknown = [i for i in ids if i not in rows]
    if unknown:
        sys.exit(f"{len(unknown)} predictions are not in sentences.csv, e.g. {unknown[:3]}")

    duplicates = [i for i, count in Counter(ids).items() if count > 1]
    if duplicates:
        sys.exit(f"Duplicate prediction ids: {duplicates[:3]}")

    absent = [i for i in rows if i not in set(ids)]
    if absent and not args.allow_partial:
        sys.exit(f"{len(absent)} sentences have no prediction, e.g. {absent[:3]}")

    results = [score_item(rows[p["id"]], spans[p["id"]], p, args.degenerate_ratio)
               for p in predictions]

    report(results, args.label, args.degenerate_ratio)

    detail_path = args.preds.parent / "scores_detail.json"
    detail_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"per-item detail: {detail_path}")

    if args.dump_errors:
        errors = [r for r in results if r["correct"] is False]
        if errors:
            with open(args.dump_errors, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(errors[0]))
                writer.writeheader()
                writer.writerows(errors)
            print(f"{len(errors)} failing items: {args.dump_errors}")
        else:
            print("no failing items")


if __name__ == "__main__":
    main()
