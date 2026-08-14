"""
score_c.py — scoring for Dataset C (the context-sensitivity probe)

WHY THIS IS A SEPARATE SCRIPT FROM score.py
--------------------------------------------
Dataset A/B are binary: a sentence either leaks a protected attribute or it
doesn't, and success is "does the output still contain it". Dataset C tests a
genuinely different, harder question (from Prof. Dahlmeier's feedback): is
the SAME attribute, about the SAME kind of person, sensitive depending on
WHO they are and WHY it's being said (private citizen vs. public figure in a
journalistic/public-interest context). Rajul's own dataset_c_gen.py docstring
specifies the correct evaluation method, and it is NOT the same as Dataset A:

  "PRESERVE rows are not scored by string identity. Three criteria are
   recorded separately: the name appears unaltered; the Art. 9 attribute is
   still explicit; no fact is introduced that was absent from the input.
   A row is correct only if all three hold, and the failing criterion is
   recorded. REDACT rows are scored as in Dataset A, against whether the
   attribute remains inferable. Results are reported per item and per
   legitimacy_basis. Twenty sentences cannot support a significance test;
   analysis is descriptive."

This script implements exactly that.

IMPORTANT KNOWN LIMITATION — read before trusting a leak/failure verdict
--------------------------------------------------------------------------
Dataset C, unlike Dataset A, does not ship a hand-labelled spans.csv marking
exactly which words are "the sensitive part" of each sentence. To work
around this, this script AUTOMATICALLY derives two things by diffing each
REDACT row's input_text against its own expected_output:
  1. The person's name — NOT derived automatically. Automatic extraction
     (tried first) was WRONG on ~5 of 11 rows, because job titles like
     "Wirtschaftsminister" sit right before the first name and look
     identical to a real name to a simple pattern. Instead, every name below
     was read BY HAND directly from dataset_c_gen.py's source (every
     `text="..."` line), not guessed. See NAME_LOOKUP below.
  2. The "attribute terms" — words present in input_text but absent from
     expected_output (name excluded, common German function words filtered
     out). This is a best-effort heuristic, not a hand-labelled ground
     truth: it cannot perfectly distinguish "this word was removed because
     it's the sensitive content" from "this word was removed because the
     sentence got rephrased for other reasons" (e.g. a verb changing form).
     Because of this, this script reports WHICH exact term triggered a
     leak/failure, not just a bare True/False — always sanity-check a
     single-generic-word match yourself before treating it as a real leak.
     A stronger version of this would have Rajul hand-tag the 1-2 true
     "core" sensitive words per REDACT row directly in dataset_c.csv,
     the same way Dataset A has hand-labelled spans.csv. Recommended
     follow-up, not done here.

REQUIRED PREDICTIONS FILE FORMAT (predictions_c.csv)
-----------------------------------------------------
  sentence_id    : str  — matches an `id` in dataset_c.csv
  model          : str
  config         : str
  predicted_action : text — one of: none, mask, rewrite, mask_and_rewrite
                     (matches main.py's existing `action` field — reuse it
                     directly, don't invent a new field)
  output_text    : str  — the model's final output sentence
  num_llm_calls  : int

USAGE
-----
  python3 score_c.py --predictions predictions_c.csv --dataset dataset_c.csv \
                      --out results_table_c.csv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Hand-verified name lookup (see docstring above for why this is NOT
# auto-extracted). Read directly from every `text="..."` line in
# dataset_c_gen.py. If Rajul adds items beyond c01-c11, THIS TABLE MUST BE
# EXTENDED BY HAND THE SAME WAY — do not replace this with a regex guesser.
# ---------------------------------------------------------------------------
NAME_LOOKUP = {
    "c01": "Konrad Mahlberg",
    "c03": "Norbert Perscheid",
    "c04": "Marius Enzweiler",
    "c05": "Ilka Vester",
    "c07": "Manfred Rauhut",
    "c08": "Birthe Klingsöhr",
    "c09": "Reinhold Ahrendt",
    "c10": "Anselm Kortner",
    "c11": "Selma Yildiz",
}

# Deliberately small and conservative: words so common in German that their
# presence in an output sentence means nothing about whether an attribute
# leaked (e.g. checking for "die" or "ist" would flag almost every sentence).
GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einen", "einem",
    "und", "oder", "aber", "dass", "wenn", "weil", "als", "wie", "was", "wer", "wo",
    "ist", "sind", "war", "waren", "hat", "hatte", "haben", "wird", "wurde", "werden",
    "er", "sie", "es", "ihm", "ihr", "ihn", "ihre", "sein", "seiner", "seine",
    "im", "in", "am", "an", "auf", "zu", "zum", "zur", "mit", "von", "für", "bei", "nie",
    "nicht", "auch", "noch", "nur", "so", "selbst", "über", "durch",
}


def tokenize(text):
    return re.findall(r"\w+", text or "")


def boundary_contains(text, phrase):
    if not phrase or not text:
        return False
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def attribute_terms(input_text, gold_text, name_tokens):
    """Words present in input_text but gone from gold expected_output,
    excluding the person's name and common function words. Heuristic —
    see module docstring."""
    in_toks = tokenize(input_text)
    gold_toks = set(tokenize(gold_text))
    name_set = set(t.lower() for t in name_tokens)
    removed = [t for t in in_toks if t not in gold_toks and t.lower() not in name_set]
    return [t for t in removed if t.lower() not in GERMAN_STOPWORDS]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def open_csv_safely(path, description):
    try:
        f = open(path, encoding="utf-8-sig")
    except FileNotFoundError:
        sys.exit(f"FATAL: {description} not found at path: {path!r}")
    return f


def load_dataset_c(path):
    """Returns a dict: sentence_id -> row dict (with an added 'name' and
    'attribute_terms' field computed per the method above)."""
    f = open_csv_safely(path, "Dataset C file (--dataset)")
    with f:
        reader = csv.DictReader(f)
        required = {"id", "item_id", "expected_action", "legitimacy_basis",
                    "input_text", "expected_output", "art9_category"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"FATAL: Dataset C file (--dataset {path!r}) is missing required "
                      f"column(s): {sorted(missing)}\nGot columns: {reader.fieldnames}\n"
                      f"Did you pass the right file? --predictions and --dataset "
                      f"accidentally swapped is a common mistake — they're both plain CSVs.")
        rows = {}
        bad_actions = []
        bad_gold = []
        dupe_ids = []
        for row in reader:
            # Same defensive stripping as load_predictions() -- a stray space in
            # id/item_id (easy to introduce via an Excel edit) would otherwise
            # silently break the NAME_LOOKUP match without any visible error,
            # degrading that row's name-check without anyone noticing.
            row["id"] = row["id"].strip()
            row["item_id"] = row["item_id"].strip()
            if row["id"] in rows:
                dupe_ids.append(row["id"])
                continue
            if row["expected_action"] not in ("REDACT", "PRESERVE"):
                bad_actions.append((row["id"], row["expected_action"]))
            # A REDACT row's expected_output is the gold reference this script
            # diffs against to figure out what "sensitive" even means for that
            # row. If it's blank, or identical to the input (i.e. the gold data
            # itself shows no redaction happened), the diff produces a huge,
            # meaningless bag of nearly every word in the sentence instead of
            # the real sensitive terms -- silently making every leak-check for
            # that row absurd. Catch this at load time, not downstream.
            if row["expected_action"] == "REDACT":
                if not row["expected_output"].strip():
                    bad_gold.append((row["id"], "expected_output is empty"))
                elif row["expected_output"].strip() == row["input_text"].strip():
                    bad_gold.append((row["id"], "expected_output is identical to input_text "
                                                 "(REDACT row but gold shows no change)"))
            name = NAME_LOOKUP.get(row["item_id"], "")
            if not name:
                print(f"WARNING: no hand-verified name for item_id {row['item_id']!r} "
                      f"(id {row['id']!r}) — add it to NAME_LOOKUP in score_c.py. "
                      f"Name-preserved criterion will be skipped for this row.",
                      file=sys.stderr)
            row["_name"] = name
            row["_name_tokens"] = tokenize(name)
            if row["expected_action"] == "REDACT":
                row["_attribute_terms"] = attribute_terms(
                    row["input_text"], row["expected_output"], row["_name_tokens"])
            else:
                # For PRESERVE rows, the "attribute terms" that must STILL be
                # present are derived from this item's REDACT sibling (found
                # after the full pass below), not from this row itself
                # (since expected_output == input_text here, there's nothing
                # to diff against).
                row["_attribute_terms"] = None
            rows[row["id"]] = row

        if bad_actions:
            sys.exit(f"FATAL: {len(bad_actions)} row(s) in Dataset C have an "
                      f"expected_action that isn't exactly 'REDACT' or 'PRESERVE' — "
                      f"refusing to guess which bucket they belong in: {bad_actions}")
        if bad_gold:
            sys.exit(f"FATAL: {len(bad_gold)} REDACT row(s) have broken gold data — "
                      f"the attribute-term extraction depends on a real, different "
                      f"expected_output to diff against: {bad_gold}")
        if dupe_ids:
            sys.exit(f"FATAL: {len(dupe_ids)} duplicate id(s) found within Dataset C "
                      f"itself (not the predictions file) — fix the dataset file, "
                      f"duplicate ids: {dupe_ids}")

    # Second pass: fill in PRESERVE rows' attribute_terms from their item's
    # REDACT sibling (every item has at least one, guaranteed by Rajul's own
    # generator validation — "no REDACT/PRESERVE contrast" check).
    by_item = defaultdict(list)
    for row in rows.values():
        by_item[row["item_id"]].append(row)
    for item_id, item_rows in by_item.items():
        redact_terms = None
        for r in item_rows:
            if r["expected_action"] == "REDACT" and r["_attribute_terms"]:
                redact_terms = r["_attribute_terms"]
                break
        for r in item_rows:
            if r["expected_action"] == "PRESERVE":
                r["_attribute_terms"] = redact_terms or []

    return rows


# ---------------------------------------------------------------------------
# Predictions loading
# ---------------------------------------------------------------------------

REQUIRED_PRED_COLS = {"sentence_id", "model", "config", "predicted_action",
                       "output_text", "num_llm_calls"}


def load_predictions(path):
    f = open_csv_safely(path, "predictions file (--predictions)")
    with f:
        reader = csv.DictReader(f)
        missing = REQUIRED_PRED_COLS - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"FATAL: predictions file (--predictions {path!r}) is missing required "
                      f"column(s): {sorted(missing)}\nExpected: {sorted(REQUIRED_PRED_COLS)}\n"
                      f"Got columns: {reader.fieldnames}\n"
                      f"Did you pass the right file? --predictions and --dataset "
                      f"accidentally swapped is a common mistake — they're both plain CSVs.")
        rows = []
        for i, row in enumerate(reader, start=2):
            row["sentence_id"] = row["sentence_id"].strip()
            row["model"] = row["model"].strip()
            row["config"] = row["config"].strip()
            row["predicted_action"] = row["predicted_action"].strip()
            try:
                row["num_llm_calls"] = int(float(row["num_llm_calls"]))
            except ValueError as e:
                sys.exit(f"FATAL: bad value on predictions_c.csv line {i}: {e}")
            rows.append(row)
    if not rows:
        sys.exit("FATAL: predictions_c.csv has no data rows.")

    seen, deduped, dupes = {}, [], 0
    for row in rows:
        key = (row["model"], row["config"], row["sentence_id"])
        if key in seen:
            dupes += 1
            continue
        seen[key] = True
        deduped.append(row)
    if dupes:
        print(f"WARNING: {dupes} duplicate (model, config, sentence_id) row(s) — "
              f"kept only the first occurrence of each.", file=sys.stderr)
    return deduped


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_redact_row(gt_row, output_text):
    """Returns (leaked: bool, matched_terms: list[str])."""
    matched = []
    if gt_row["_name"] and boundary_contains(output_text, gt_row["_name"]):
        matched.append(f"NAME:{gt_row['_name']}")
    for term in gt_row["_attribute_terms"]:
        if boundary_contains(output_text, term):
            matched.append(term)
    return (len(matched) > 0), matched


def score_preserve_row(gt_row, output_text):
    """Returns dict with 3 criteria + overall correctness, per Rajul's spec:
    'A row is correct only if all three hold, and the failing criterion is
    recorded.'"""
    result = {}

    # Criterion 1: name appears unaltered.
    if gt_row["_name"]:
        result["name_preserved"] = boundary_contains(output_text, gt_row["_name"])
    else:
        result["name_preserved"] = None  # not computed — see WARNING at load time

    # Criterion 2: the Art.9 attribute is still explicit (i.e. still present).
    terms = gt_row["_attribute_terms"] or []
    matched = [t for t in terms if boundary_contains(output_text, t)]
    result["attribute_still_explicit"] = (len(matched) > 0) if terms else None
    result["attribute_terms_found"] = matched

    # Criterion 3: no fact introduced that was absent from the input.
    # HEURISTIC ONLY (see module docstring) — flags new content-like words
    # (len>=4, alphabetic) in the output that weren't anywhere in the input.
    # This cannot detect a genuinely new CLAIM built from existing words, and
    # can false-flag a harmless synonym choice. Treat as "worth a manual
    # look", not a verdict, same caution as the German-compound-word note in
    # score.py.
    in_toks = set(t.lower() for t in tokenize(gt_row["input_text"]))
    out_toks = [t for t in tokenize(output_text) if len(t) >= 4 and t.isalpha()]
    # Any word that appears inside square brackets in THIS output (e.g. "[NAME]",
    # "[MASKED]", "[REDACTED]") is a structural placeholder the pipeline itself
    # introduced, not a fact the model invented -- detected dynamically from the
    # actual bracket pattern used, rather than guessed from a fixed word list.
    # This matters in practice: the real pipeline uses "[MASKED]" (confirmed
    # from real DeepSeek output in the team chat), not "[NAME]" as first assumed
    # -- a hardcoded guess-list would have silently missed it and double-counted
    # every over-redaction as ALSO an "invented fact" failure.
    bracket_tokens = set(t.lower() for t in re.findall(r"\[(\w+)\]", output_text))
    new_words = [t for t in out_toks if t.lower() not in in_toks
                 and t.lower() not in GERMAN_STOPWORDS
                 and t.lower() not in bracket_tokens]
    result["no_invented_facts"] = (len(new_words) == 0)
    result["possible_invented_words"] = new_words

    criteria = [result["name_preserved"], result["attribute_still_explicit"], result["no_invented_facts"]]
    computed = [c for c in criteria if c is not None]
    result["overall_correct"] = all(computed) if computed else None
    failing = []
    if result["name_preserved"] is False:
        failing.append("name_preserved")
    if result["attribute_still_explicit"] is False:
        failing.append("attribute_still_explicit")
    if result["no_invented_facts"] is False:
        failing.append("no_invented_facts")
    result["failing_criteria"] = failing
    return result


def main():
    ap = argparse.ArgumentParser(description="Score Dataset C (context-sensitivity probe) predictions.")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--dataset", default="data/dataset_c.csv")
    ap.add_argument("--out", default="results_table_c.csv")
    args = ap.parse_args()

    gt = load_dataset_c(args.dataset)
    preds = load_predictions(args.predictions)

    groups = defaultdict(list)
    for r in preds:
        groups[(r["model"], r["config"])].append(r)

    detail_rows = []
    summary_rows = []

    for (model, config), rows in sorted(groups.items()):
        redact_total = redact_leaked = 0
        preserve_total = preserve_correct = 0
        by_basis = defaultdict(lambda: {"total": 0, "correct": 0})
        missing_gt = 0
        empty_output = 0
        calls = []

        for r in rows:
            sid = r["sentence_id"]
            if sid not in gt:
                missing_gt += 1
                continue
            gt_row = gt[sid]
            calls.append(r["num_llm_calls"])
            out = r["output_text"]

            # A blank/whitespace-only output means the model call failed, timed
            # out, or errored -- it is NOT a successful redaction just because
            # there's no text left to find a leak in. Score it as invalid, not
            # as a pass, so a crashed call can never silently inflate results.
            if not out or not out.strip():
                empty_output += 1
                detail_rows.append({
                    "model": model, "config": config, "sentence_id": sid,
                    "item_id": gt_row["item_id"], "expected_action": gt_row["expected_action"],
                    "leaked": "", "matched_terms": "",
                    "name_preserved": "", "attribute_still_explicit": "",
                    "no_invented_facts": "", "failing_criteria": "EMPTY_OUTPUT",
                    "overall_correct": "",
                })
                continue

            if gt_row["expected_action"] == "REDACT":
                leaked, matched = score_redact_row(gt_row, out)
                redact_total += 1
                if leaked:
                    redact_leaked += 1
                detail_rows.append({
                    "model": model, "config": config, "sentence_id": sid,
                    "item_id": gt_row["item_id"], "expected_action": "REDACT",
                    "leaked": leaked, "matched_terms": ";".join(matched),
                    "name_preserved": "", "attribute_still_explicit": "",
                    "no_invented_facts": "", "failing_criteria": "",
                    "overall_correct": (not leaked),
                })
            else:  # PRESERVE
                res = score_preserve_row(gt_row, out)
                preserve_total += 1
                if res["overall_correct"]:
                    preserve_correct += 1
                basis = (gt_row["legitimacy_basis"] or "").strip() or "(none)"
                by_basis[basis]["total"] += 1
                if res["overall_correct"]:
                    by_basis[basis]["correct"] += 1
                detail_rows.append({
                    "model": model, "config": config, "sentence_id": sid,
                    "item_id": gt_row["item_id"], "expected_action": "PRESERVE",
                    "leaked": "", "matched_terms": ";".join(res["attribute_terms_found"]),
                    "name_preserved": res["name_preserved"],
                    "attribute_still_explicit": res["attribute_still_explicit"],
                    "no_invented_facts": res["no_invented_facts"],
                    "failing_criteria": ";".join(res["failing_criteria"]),
                    "overall_correct": res["overall_correct"],
                })

        redact_leak_rate = (redact_leaked / redact_total) if redact_total else None
        preserve_correct_rate = (preserve_correct / preserve_total) if preserve_total else None
        avg_calls = (sum(calls) / len(calls)) if calls else None

        summary_rows.append({
            "model": model, "config": config,
            "n_redact": redact_total, "redact_leak_rate": redact_leak_rate,
            "n_preserve": preserve_total, "preserve_correct_rate": preserve_correct_rate,
            "avg_llm_calls_per_sentence": avg_calls,
            "missing_from_ground_truth": missing_gt,
            "empty_output_rows": empty_output,
            "preserve_by_legitimacy_basis": "; ".join(
                f"{basis}: {v['correct']}/{v['total']}" for basis, v in sorted(by_basis.items())
            ),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "config", "n_redact", "redact_leak_rate",
                      "n_preserve", "preserve_correct_rate",
                      "avg_llm_calls_per_sentence", "missing_from_ground_truth",
                      "empty_output_rows", "preserve_by_legitimacy_basis"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    detail_path = str(Path(args.out).with_name(Path(args.out).stem + "_detail" + Path(args.out).suffix))
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "config", "sentence_id", "item_id", "expected_action",
                      "leaked", "matched_terms", "name_preserved",
                      "attribute_still_explicit", "no_invented_facts",
                      "failing_criteria", "overall_correct"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(detail_rows)

    print(f"\nScored {len(preds)} prediction rows across {len(summary_rows)} (model, config) group(s).")
    print("REMINDER (per Rajul's own design note): n=20 sentences per model — this is a")
    print("descriptive, qualitative probe, NOT a statistically powered test. Report it that way.\n")
    header = f"{'model':<20} {'config':<14} {'REDACT leak%':>13} {'PRESERVE correct%':>19} {'calls/sent':>10}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        leak_s = f"{r['redact_leak_rate']*100:.1f}" if r["redact_leak_rate"] is not None else "n/a"
        pres_s = f"{r['preserve_correct_rate']*100:.1f}" if r["preserve_correct_rate"] is not None else "n/a"
        calls_s = f"{r['avg_llm_calls_per_sentence']:.2f}" if r["avg_llm_calls_per_sentence"] is not None else "n/a"
        print(f"{r['model']:<20} {r['config']:<14} {leak_s:>13} {pres_s:>19} {calls_s:>10}")
        if r["missing_from_ground_truth"]:
            print(f"   ! WARNING: {r['missing_from_ground_truth']} row(s) had a sentence_id not in Dataset C.")
        if r["empty_output_rows"]:
            print(f"   ! WARNING: {r['empty_output_rows']} row(s) had empty/blank output_text "
                  f"(likely a failed model call) — excluded from the rates above, NOT counted as a pass.")

    print(f"\nSummary written to: {args.out}")
    print(f"Per-row detail (incl. WHICH term triggered each leak/failure) written to: {detail_path}")
    print("Always spot-check detail rows with a single generic matched_term before calling it a real leak —")
    print("see the KNOWN LIMITATION note at the top of this script.")


if __name__ == "__main__":
    main()
