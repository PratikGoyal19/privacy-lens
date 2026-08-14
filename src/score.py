"""
score.py — PrivacyLens evaluation scoring script

WHAT THIS DOES
---------------
Takes:
  1. Dataset A ground truth (sentences.csv + spans.csv) — the sealed 150-sentence
     evaluation set used for ALL models (the 4 prompted models AND the fine-tuned model).
  2. A predictions file (predictions.csv) — produced by the pipeline (Harshitha's code),
     one row per (sentence, model, config) combination that was run.

Computes, per (model, config) group:
  - Detection metrics: precision, recall, F1 (did the model correctly say "this sentence
    is sensitive / not sensitive")
  - Leak rate: across ALL sentences that are ACTUALLY sensitive (ground truth), does the
    output text still contain the sensitive span text? This INCLUDES sentences the model
    missed entirely (false negatives) — a missed sentence is left unmasked, so by
    definition its sensitive content is still fully present, i.e. still leaked. This is
    deliberate: leak rate is meant to answer "if I actually use this pipeline, how often
    does my private info end up unprotected", which depends on both detection AND repair
    quality together, not repair quality alone. (Exact/partial span match, same
    methodology style as the LLM-Redactor paper's leak_meter.py.)
  - Cost: average number of LLM calls per sentence (relevant for the iteration loop —
    a single-pass model uses 1 call, a 3-round iterative model might use 1-3)

Outputs:
  - results_table.csv — one row per (model, config), ready to paste into the report
  - Prints a human-readable summary table to the terminal

REQUIRED INPUT FORMAT (predictions.csv) — this is the contract for the pipeline code:
  sentence_id        : str   — must match an `id` in sentences.csv (e.g. "de_health_003_pos")
  model               : str   — e.g. "llama3.2", "qwen2.5", "mistral", "deepseek-r1", "finetuned-llama3.2"
  config              : str   — e.g. "single_pass", "iterative_3round", "finetuned"
  predicted_sensitive : bool  — True/False — did the model flag this sentence as sensitive
  output_text         : str   — the model's final masked/rewritten sentence text
                                 (if predicted_sensitive is False, this can just equal the input text)
  num_llm_calls       : int   — how many LLM calls this sentence took (1 for single-pass,
                                 1-3 for iterative, 1 for fine-tuned single-shot)

USAGE
-----
  python3 score.py --predictions predictions.csv --sentences data/sentences.csv \
                    --spans data/spans.csv --out results_table.csv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def open_csv_safely(path, description):
    """Open a CSV for reading, or fail with a clear one-line message instead of
    a raw Python traceback. `description` is a human label used in the error
    (e.g. "predictions file", "Dataset A sentences file") so whoever hits this
    knows immediately which --argument to check, not just that "a file" failed.
    """
    try:
        f = open(path, encoding="utf-8-sig")
    except FileNotFoundError:
        sys.exit(f"FATAL: {description} not found at path: {path!r}\n"
                  f"Check the path / your current folder, or pass the correct "
                  f"--argument for this file.")
    return f


def require_columns(fieldnames, required, description, path):
    """Fail clearly (not with a KeyError later) if a CSV is missing columns
    it's supposed to have — e.g. this catches --sentences and --spans being
    accidentally swapped on the command line, since spans.csv doesn't have an
    'id'/'has_sensitive_attribute' column and sentences.csv doesn't have a
    'sentence_id'/'kind' column.
    """
    missing = set(required) - set(fieldnames or [])
    if missing:
        sys.exit(f"FATAL: {description} ({path!r}) is missing required column(s): "
                  f"{sorted(missing)}\nGot columns: {fieldnames}\n"
                  f"Did you pass the right file for this argument? (e.g. --sentences "
                  f"and --spans accidentally swapped is a common mistake — they're "
                  f"both plain CSVs.)")


def load_ground_truth(sentences_path, spans_path):
    """Returns:
      gt_label[sentence_id]      -> bool  (has_sensitive_attribute)
      gt_text[sentence_id]       -> str   (original sentence text)
      gt_spans[sentence_id]      -> list of (kind, text) for spans that must be
                                     GONE from a correct redaction/rewrite.
                                     We treat IMPLICIT_CUE and any ID/identifier span
                                     as "must not survive" spans. DECOY_CUE spans are
                                     deliberately excluded (they live in negative
                                     sentences and should NOT be redacted).
    """
    gt_label, gt_text = {}, {}
    f = open_csv_safely(sentences_path, "Dataset A sentences file (--sentences)")
    with f:
        reader = csv.DictReader(f)
        require_columns(reader.fieldnames, {"id", "text", "has_sensitive_attribute"},
                         "Dataset A sentences file (--sentences)", sentences_path)
        for row in reader:
            gt_label[row["id"]] = row["has_sensitive_attribute"] == "True"
            gt_text[row["id"]] = row["text"]

    gt_spans = defaultdict(list)
    # DECOY_CUE is the ONLY kind excluded — it's the harmless twin phrase that
    # lives in a negative sentence and must NOT be flagged/removed. Every other
    # kind (verified against Dataset A's actual spans.csv: ADDRESS, EMAIL,
    # ID_IBAN, ID_KFZ, ID_KUNDENNUMMER, ID_KVNR, ID_MITARBEITER, ID_STEUER,
    # ID_SVNR, IMPLICIT_CUE, PERSON, PHONE) represents something that must
    # disappear from a correct mask/rewrite.
    EXCLUDED_KINDS = {"DECOY_CUE"}
    f = open_csv_safely(spans_path, "Dataset A spans file (--spans)")
    with f:
        reader = csv.DictReader(f)
        require_columns(reader.fieldnames, {"sentence_id", "kind", "text"},
                         "Dataset A spans file (--spans)", spans_path)
        for row in reader:
            if row["kind"] not in EXCLUDED_KINDS:
                gt_spans[row["sentence_id"]].append((row["kind"], row["text"]))

    return gt_label, gt_text, gt_spans


# ---------------------------------------------------------------------------
# Leak checking (span-level, boundary-anchored — same style as Dataset A/B's
# own matcher, so results are comparable and reproducible)
# ---------------------------------------------------------------------------

def boundary_contains(text, phrase):
    """True if `phrase` appears in `text` as a whole-word(ish) match.
    Mirrors the boundary-anchored regex approach used in the dataset generators,
    so a leak check here means the same thing it means in Dataset A/B.

    KNOWN LIMITATION (worth a line in the report, not a bug to "fix" silently):
    this will NOT catch a cue word hiding inside a German compound word, e.g.
    "Chemotherapie" is found in "Er hat eine Chemotherapie." but NOT found in
    "Er hat einen Chemotherapietermin." — even though a human reader would
    still consider the second sentence a leak. This is a real limitation of
    exact/partial span matching in a compounding language like German (the
    original paper's leak_meter.py has the same blind spot, just less visible
    in English, which compounds far less). The semantic-judge check
    (semantic_judge_stub, once wired up) is what actually closes this gap —
    span matching alone should not be reported as a complete leak measure.
    """
    if not phrase or not text:
        return False
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def compute_leak(output_text, spans):
    """Returns (leaked: bool, partial_leak_count: int, total_spans: int).
    leaked=True if ANY must-remove span text is still findable in the output.
    This is the 'exact/partial' style check from the original paper's leak_meter.py —
    NOT the semantic-judge check. See semantic_judge_stub() below for that.
    """
    if not spans:
        return False, 0, 0
    hits = sum(1 for (_kind, text) in spans if boundary_contains(output_text, text))
    return hits > 0, hits, len(spans)


def semantic_judge_stub(sentence_id, output_text, model_name="llama3.2"):
    """
    PLACEHOLDER for the semantic-leak judge (LLM-as-judge), matching the
    methodology the LLM-Redactor paper used and that we agreed to reuse.

    Span-text matching (compute_leak above) catches leaks where the exact
    words survive. It CANNOT catch a leak where the model rewrote around the
    words but the meaning still gives the person away (e.g. rewriting
    "Chemotherapie" away but keeping "seit den Behandlungen im Onkologiezentrum").

    Wire this up once the Ollama pipeline exists: send output_text to a judge
    model (use a DIFFERENT model than the one that produced the rewrite, to
    avoid the model grading its own work — matches what we agreed on) and ask
    it to decide whether the sensitive attribute is still inferable.

    Returns None until implemented — score.py treats None as "not run" and
    reports span-level leak rate only, clearly labelled as such, so we never
    silently pass off partial results as complete ones.
    """
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def score_group(rows, gt_label, gt_spans):
    """rows: list of prediction dicts for ONE (model, config) group."""
    tp = fp = fn = tn = 0
    leak_flags = []
    call_counts = []
    missing_gt = 0
    classification_available = False  # True if ANY row had a real predicted_sensitive value

    for r in rows:
        sid = r["sentence_id"]
        if sid not in gt_label:
            missing_gt += 1
            continue

        truth = gt_label[sid]
        pred = r["predicted_sensitive"]

        # pred is None when the WHOLE FILE never had a predicted_sensitive column
        # (e.g. a fine-tuned model that only rewrites sentences and was never
        # trained to classify "is this sensitive, yes/no" in the first place —
        # there is no honest tp/fp/fn/tn to compute for it, only leak rate).
        if pred is not None:
            classification_available = True
            if truth and pred:
                tp += 1
            elif truth and not pred:
                fn += 1
            elif (not truth) and pred:
                fp += 1
            else:
                tn += 1

        call_counts.append(r["num_llm_calls"])

        # Leak rate covers ALL sentences that are actually sensitive per ground truth —
        # including ones the model never flagged (false negatives). A missed sentence's
        # output_text is (per the format contract) left unchanged, so it necessarily still
        # contains the sensitive content — that correctly counts as a leak. This makes
        # leak_rate an END-TO-END number (detection failure OR repair failure both show up
        # here), not a repair-quality-only number. See docstring at top of file. Note this
        # is computed from truth + output_text ONLY, never from predicted_sensitive — so
        # leak_rate works identically whether or not classification data exists at all.
        if truth:
            spans = gt_spans.get(sid, [])
            leaked, _hits, _total = compute_leak(r["output_text"], spans)
            leak_flags.append(leaked)

    if classification_available:
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    else:
        # Genuinely not applicable, not a real zero — reported as None/"n/a"
        # downstream so it's never mistaken for "scored zero precision".
        precision = recall = f1 = None
    leak_rate = (sum(leak_flags) / len(leak_flags)) if leak_flags else None
    avg_calls = (sum(call_counts) / len(call_counts)) if call_counts else None

    return {
        "n_rows": len(rows),
        "missing_from_ground_truth": missing_gt,
        "classification_available": classification_available,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_positive_evaluated_for_leak": len(leak_flags),
        "leak_rate": leak_rate,
        "avg_llm_calls_per_sentence": avg_calls,
    }


# ---------------------------------------------------------------------------
# Predictions loading + validation
# ---------------------------------------------------------------------------

# predicted_sensitive is intentionally NOT in this set -- it's optional. A
# fine-tuned model that only learned to rewrite sentences (never trained to
# classify "is this sensitive") has no honest value to put there. When the
# column is entirely absent from a file, that file is scored on leak_rate
# and cost only; precision/recall/F1 are reported as "n/a", never guessed.
REQUIRED_COLS = {"sentence_id", "model", "config",
                  "output_text", "num_llm_calls"}


def load_predictions(path):
    f = open_csv_safely(path, "predictions file (--predictions)")
    with f:
        reader = csv.DictReader(f)
        require_columns(reader.fieldnames, REQUIRED_COLS,
                         "predictions file (--predictions)", path)
        has_classification_col = "predicted_sensitive" in (reader.fieldnames or [])
        rows = []
        for i, row in enumerate(reader, start=2):  # start=2: line 1 is header
            try:
                # Strip whitespace on the 3 key/id-like columns. A stray leading
                # or trailing space (easy to introduce via a copy-paste into
                # Excel/Numbers) would otherwise silently create a phantom
                # extra (model, config) group, or make a valid sentence_id
                # look "missing from ground truth" for no visible reason.
                row["sentence_id"] = row["sentence_id"].strip()
                row["model"] = row["model"].strip()
                row["config"] = row["config"].strip()
                if has_classification_col:
                    row["predicted_sensitive"] = row["predicted_sensitive"].strip().lower() in ("true", "1", "yes")
                else:
                    # Column doesn't exist at all in this file -- e.g. a
                    # fine-tuned model that only rewrites sentences and was
                    # never trained to classify sensitive/not-sensitive.
                    # None is a real sentinel here, not a default False --
                    # score_group() treats it as "no honest value available",
                    # not as "predicted not sensitive".
                    row["predicted_sensitive"] = None
                # int(float(...)) instead of int(...): tolerates "2.0" as well as "2",
                # which is a common accident when a pipeline computes averages/ratios
                # in Python (floats creep in even for things that are conceptually counts).
                row["num_llm_calls"] = int(float(row["num_llm_calls"]))
            except ValueError as e:
                sys.exit(f"FATAL: bad value on predictions.csv line {i}: {e}")
            rows.append(row)
    if not rows:
        sys.exit("FATAL: predictions.csv has no data rows.")

    # Duplicate detection: the same sentence_id showing up more than once for the
    # SAME (model, config) is almost always a pipeline bug (e.g. a retry that got
    # appended instead of overwritten, or a batch re-run without clearing old output).
    # Left unchecked this silently skews precision/recall/F1 and leak rate with no
    # visible sign anything is wrong — so we detect it, warn loudly, and keep only
    # the FIRST occurrence of each duplicate (deterministic, doesn't crash the run,
    # but the warning below should not be ignored).
    seen = {}
    deduped = []
    dupe_count = 0
    for row in rows:
        key = (row["model"], row["config"], row["sentence_id"])
        if key in seen:
            dupe_count += 1
            continue
        seen[key] = True
        deduped.append(row)

    if dupe_count:
        print(f"WARNING: {dupe_count} duplicate (model, config, sentence_id) row(s) found "
              f"in predictions.csv — kept only the first occurrence of each. This usually "
              f"means the pipeline ran the same sentence twice for the same setup. "
              f"Fix the pipeline output before trusting these numbers.\n", file=sys.stderr)

    return deduped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Score PrivacyLens model predictions against Dataset A.")
    ap.add_argument("--predictions", required=True, help="Path to predictions.csv from the pipeline")
    ap.add_argument("--sentences", default="data/sentences.csv")
    ap.add_argument("--spans", default="data/spans.csv")
    ap.add_argument("--out", default="results_table.csv")
    args = ap.parse_args()

    gt_label, gt_text, gt_spans = load_ground_truth(args.sentences, args.spans)
    preds = load_predictions(args.predictions)

    groups = defaultdict(list)
    for r in preds:
        groups[(r["model"], r["config"])].append(r)

    results = []
    for (model, config), rows in sorted(groups.items()):
        m = score_group(rows, gt_label, gt_spans)
        m["model"] = model
        m["config"] = config
        results.append(m)

    # Write results table
    fieldnames = ["model", "config", "n_rows", "missing_from_ground_truth",
                  "classification_available", "tp", "fp", "fn", "tn",
                  "precision", "recall", "f1",
                  "n_positive_evaluated_for_leak", "leak_rate",
                  "avg_llm_calls_per_sentence"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Print human-readable summary
    print(f"\nScored {len(preds)} prediction rows across {len(results)} (model, config) group(s).\n")
    header = f"{'model':<20} {'config':<18} {'n':>4} {'P':>6} {'R':>6} {'F1':>6} {'leak%':>7} {'calls/sent':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        leak_str = f"{r['leak_rate']*100:.1f}" if r["leak_rate"] is not None else "n/a"
        calls_str = f"{r['avg_llm_calls_per_sentence']:.2f}" if r["avg_llm_calls_per_sentence"] is not None else "n/a"
        prf_str = (f"{r['precision']:.3f} {r['recall']:.3f} {r['f1']:.3f}"
                   if r["classification_available"] else "  n/a   n/a   n/a")
        print(f"{r['model']:<20} {r['config']:<18} {r['n_rows']:>4} "
              f"{prf_str} "
              f"{leak_str:>7} {calls_str:>10}")
        if not r["classification_available"]:
            print(f"   i NOTE: no predicted_sensitive column for this group — "
                  f"precision/recall/F1 not applicable (e.g. a fine-tuned model "
                  f"that only rewrites, never classifies). Leak rate is unaffected "
                  f"and still valid.")
        if r["missing_from_ground_truth"]:
            print(f"   ! WARNING: {r['missing_from_ground_truth']} row(s) had a sentence_id "
                  f"not found in Dataset A — check for typos or a mismatched dataset version.")
            if r["n_rows"] == r["missing_from_ground_truth"]:
                print(f"   ! ALL rows in this group were unmatched — precision/recall/F1/leak/cost "
                      f"above are meaningless (computed from zero valid rows).")

    print(f"\nFull results written to: {args.out}")


if __name__ == "__main__":
    main()
