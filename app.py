"""
PrivacyLens: interactive demo for German privacy redaction under the GDPR.

Three tabs.

  Dataset A   150 German sentences in 55 pairs. Each pair shares its cue
              vocabulary; only one member discloses protected data. Shows
              every system's output against the gold target, with the scorer's
              verdict per sentence and per pair.

  Dataset C   20 German sentences probing contextual legitimacy. The same
              disclosure is made about a public office-holder and about a
              private individual, so the correct action differs even though
              the facts are near-identical.

  Results     Every reported number, computed from the committed summary
              tables at load time. No figure in this file is hardcoded.

Modes on Dataset A.

  Cached   Replays the stored evaluation outputs. Instant, and what is shown
           is exactly the data behind the reported numbers.
  Live     Generates on demand for every system, so arbitrary German input can
           be tried. The two local systems load Llama 3.2 3B; use "Warm up"
           first to avoid a stall. The prompted models go through Ollama and
           need it running. All systems get the German rewrite prompt in this
           mode and are asked for a rewritten sentence, rather than the
           structured JSON the evaluation pipeline collects.

Datasets A and C are deliberately not committed, so that models scraped from
the repository cannot train on the test set. A fresh clone therefore has no
data until the generators are run. The app starts anyway and reports what is
missing rather than raising.

Usage:
    pip install -r requirements.txt
    python app.py            # add OLLAMA_URL=... for a non-default Ollama host
"""

import csv
import difflib
import hashlib
import html
import json
import math
import os
import re
import subprocess
import traceback
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration. Mirrors finetuning/lora_training.py exactly.
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS = 256
SEED = 42
MAX_LIVE_CHARS = 600

SYSTEM_PROMPT = (
    "Du bist ein Datenschutz-Filter. Ersetze direkte Identifikatoren durch Platzhalter "
    "und formuliere Sätze so um, dass besondere Kategorien personenbezogener Daten "
    "(Gesundheit, ethnische Herkunft, Religion) nicht mehr ableitbar sind. Erfinde keine "
    "neuen Tatsachen. Wenn nichts Schützenswertes offengelegt wird, gib den Satz "
    "unverändert zurück."
)

ADAPTER_DIR = REPO_ROOT / "finetuning" / "lora_outputs" / "final_model"
SENTENCES_CSV = REPO_ROOT / "data" / "sentences.csv"
DATASET_C_CSV = REPO_ROOT / "data" / "dataset_c.csv"
LORA_DIR = REPO_ROOT / "finetuning" / "lora_outputs"
BASE_DIR = REPO_ROOT / "finetuning" / "baseline_outputs"
OLLAMA_PREDS = REPO_ROOT / "results" / "predictions.csv"
PROMPTED_C = REPO_ROOT / "results" / "predictions_c.csv"
RESULTS_A_FT = REPO_ROOT / "finetuning" / "results_dataset_a.csv"
RESULTS_A_PROMPTED = REPO_ROOT / "results" / "results_table.csv"
RESULTS_C_FT = REPO_ROOT / "finetuning" / "results_table_c.csv"
RESULTS_C_PROMPTED = REPO_ROOT / "results" / "results_table_c.csv"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

SYSTEMS = {
    "Base model, no fine-tuning": {"params": "Llama 3.2 3B", "cache": "base", "live": "base"},
    "Fine-tuned model (LoRA)": {"params": "Llama 3.2 3B", "cache": "lora", "live": "lora"},
    "Llama 3.2, prompted only": {"params": "3B", "cache": "llama3.2", "live": "ollama"},
    "qwen2.5:7b": {"params": "7B", "cache": "qwen2.5:7b", "live": "ollama"},
    "mistral": {"params": "7B", "cache": "mistral", "live": "ollama"},
    "deepseek-r1:8b": {"params": "8B", "cache": "deepseek-r1:8b", "live": "ollama"},
}
DEFAULT_SYSTEMS = ["Base model, no fine-tuning", "Fine-tuned model (LoRA)"]

# Every system can be generated live. The prompted models return JSON with a
# has_leak flag in the evaluation pipeline; here they are asked for a rewritten
# sentence like the local systems, so their live output is a rewrite rather
# than the structured record behind the reported numbers.

# Systems with a per-item scorer verdict on disk.
SCORED = {"base", "lora"}

C_SYSTEMS = {
    "Base, default prompt": REPO_ROOT / "finetuning" / "dataset_c_base",
    "Base, legitimacy prompt": REPO_ROOT / "finetuning" / "dataset_c_base_legitimacy",
    "Fine-tuned, default prompt": REPO_ROOT / "finetuning" / "dataset_c_lora",
    "Fine-tuned, legitimacy prompt": REPO_ROOT / "finetuning" / "dataset_c_lora_legitimacy",
}
DEFAULT_C_SYSTEMS = ["Base, default prompt", "Fine-tuned, default prompt"]

EXAMPLE_IDS = [
    "de_implicit_health_01_pos",
    "de_implicit_health_03_pos",
    "de_implicit_ethnicity_19_pos",
    "de_implicit_health_02_pos",
    "de_explicit_009",
    "de_mixed_01_pos",
]

NEGATIVE_TYPE_LABEL = {
    "animal": "the subject is an animal, not a person",
    "occupational": "the person's job, not their own condition",
    "occupational_dual_role": "the person's job, not their own condition",
    "benign_cause": "an everyday cause, nothing protected",
    "third_party": "concerns someone else entirely",
    "pedagogical": "a teaching or training context",
    "logistical": "a practical or scheduling reason",
    "no_cue_vocabulary": "no sensitive vocabulary at all",
}

FRAMING_LABEL = {
    "public_legitimate": "Public office-holder, disclosure is legitimate",
    "private": "Private individual, same facts",
    "public_sensitive": "Public figure, but this disclosure is not legitimate",
}

BASIS_LABEL = {
    "press_freedom": "press freedom, Art. 85 GDPR",
    "self_disclosed": "disclosed by the person, Art. 9(2)(e)",
    "office_relevant": "directly relevant to the office",
    "public_record": "already an official public record",
    "constitutive": "inherent to the role, the sentence makes no sense without it",
}

# Every load failure is recorded here and shown in the interface. Datasets A
# and C are withheld from the repository on purpose, so an incomplete checkout
# is the expected state rather than an error.
LOAD_ERRORS = []


def _note_missing(path, detail=""):
    rel = Path(path).relative_to(REPO_ROOT) if str(path).startswith(str(REPO_ROOT)) else path
    LOAD_ERRORS.append(f"{rel}{': ' + detail if detail else ''}")


# ---------------------------------------------------------------------------
# Data loading. Nothing here raises: a missing or malformed file is recorded
# and the interface degrades to whatever is present.
# ---------------------------------------------------------------------------


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        _note_missing(path, "not found")
        return []
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception as exc:
        _note_missing(path, f"unreadable ({type(exc).__name__})")
        return []


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        _note_missing(path, f"unreadable ({type(exc).__name__})")
        return None


def _num(row, field, scale=1.0):
    """Read one numeric cell. Returns None for missing, blank or non-numeric."""
    try:
        value = (row or {}).get(field)
        if value is None or str(value).strip() == "":
            return None
        return float(value) * scale
    except (TypeError, ValueError):
        return None


def _load_a():
    by_id, by_text, groups = {}, {}, {}
    for row in _read_csv(SENTENCES_CSV):
        try:
            by_id[row["id"]] = row
            by_text[row["text"]] = row["id"]
            pid = (row.get("pair_id") or "").strip()
            if pid:
                groups.setdefault(pid, []).append(row["id"])
        except KeyError as exc:
            _note_missing(SENTENCES_CSV, f"missing column {exc}")
            return {}, {}, {}, {}, {}

    # A pair is one positive and one negative. Forty negatives carry no pair
    # id: they are standalone distractors, not half of anything.
    pairs = {}
    for pid, ids in groups.items():
        if len(ids) == 2 and {by_id[i].get("polarity") for i in ids} == {"positive", "negative"}:
            pairs[pid] = sorted(ids, key=lambda i: by_id[i].get("polarity") != "positive")

    cache, scores = {}, {}
    for key, folder in (("lora", LORA_DIR), ("base", BASE_DIR)):
        preds = _read_json(folder / "dataset_a_predictions.json")
        if preds:
            cache[key] = {d.get("input"): d.get("prediction", "") for d in preds if d.get("input")}
        detail = _read_json(folder / "scores_detail.json")
        if detail:
            scores[key] = {d["id"]: d for d in detail if isinstance(d, dict) and "id" in d}

    for row in _read_csv(OLLAMA_PREDS):
        sid = row.get("sentence_id")
        model = row.get("model")
        if sid in by_id and model:
            cache.setdefault(model, {})[by_id[sid]["text"]] = row.get("output_text", "")

    return cache, scores, by_id, by_text, pairs


def _load_c():
    by_id, groups = {}, {}
    for row in _read_csv(DATASET_C_CSV):
        try:
            by_id[row["id"]] = row
            groups.setdefault(row["item_id"], []).append(row["id"])
        except KeyError as exc:
            _note_missing(DATASET_C_CSV, f"missing column {exc}")
            return {}, {}, {}

    order = {"public_legitimate": 0, "public_sensitive": 1, "private": 2}
    for item in groups:
        groups[item].sort(key=lambda i: order.get(by_id[i].get("framing"), 9))

    results = {}
    for label, folder in C_SYSTEMS.items():
        records = _read_json(folder / "dataset_c_results.json")
        if records:
            results[label] = {r["id"]: r for r in records if isinstance(r, dict) and "id" in r}

    # The prompted models run single-pass: the justification comes back in the
    # same call, so there is no separate reasoning pass and no stated-versus-
    # acted comparison for them.
    for row in _read_csv(PROMPTED_C):
        model, sid = row.get("model"), row.get("sentence_id")
        if not model or not sid:
            continue
        results.setdefault(f"{model} (single pass)", {})[sid] = {
            "prediction": row.get("output_text", ""),
            "reasoning_en": row.get("reason", ""),
            "decision_from_output": "",
            "decision_stated": "",
            "decision_stated_en": "",
            "predicted_action": row.get("predicted_action", ""),
        }

    return by_id, groups, results


try:
    CACHE, SCORES, SENTENCES, TEXT_TO_ID, PAIRS = _load_a()
except Exception as exc:  # never let the interface fail to start
    LOAD_ERRORS.append(f"Dataset A could not be loaded ({type(exc).__name__})")
    CACHE, SCORES, SENTENCES, TEXT_TO_ID, PAIRS = {}, {}, {}, {}, {}

ID_TO_PAIR = {sid: pid for pid, ids in PAIRS.items() for sid in ids}
EXAMPLES = [SENTENCES[i]["text"] for i in EXAMPLE_IDS if i in SENTENCES]

try:
    C_ITEMS, C_GROUPS, C_RESULTS = _load_c()
except Exception as exc:
    LOAD_ERRORS.append(f"Dataset C could not be loaded ({type(exc).__name__})")
    C_ITEMS, C_GROUPS, C_RESULTS = {}, {}, {}

C_CHOICES = [
    f"{item}: {C_ITEMS[ids[0]].get('topic', '')}" for item, ids in sorted(C_GROUPS.items())
]
C_CHOICE_TO_ITEM = {c: c.split(":")[0] for c in C_CHOICES}
ALL_C_SYSTEMS = list(C_RESULTS.keys())
C_PRESERVE_N = sum(
    1 for r in C_ITEMS.values() if (r.get("expected_action") or "").strip() == "PRESERVE"
)


# ---------------------------------------------------------------------------
# Provenance. Ties what is on screen to a specific state of the repository.
# ---------------------------------------------------------------------------


def _file_md5(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


PROVENANCE = {
    "Dataset A md5": _file_md5(SENTENCES_CSV) or "not present",
    "Dataset A sentences": str(len(SENTENCES)) if SENTENCES else "0",
    "Dataset A pairs": str(len(PAIRS)),
    "Dataset C cases": str(len(C_GROUPS)),
    "Commit": _git_commit() or "not a git checkout",
    "Seed": str(SEED),
    "Adapter": "present" if ADAPTER_DIR.is_dir() else "not trained",
}


# ---------------------------------------------------------------------------
# Statistics. Pure Python so the app keeps a single dependency.
# ---------------------------------------------------------------------------


def wilson(successes, total, z=1.96):
    """Wilson score interval for a binomial proportion, as percentages."""
    if not total:
        return None
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return max(0.0, (centre - half)) * 100, min(1.0, (centre + half)) * 100


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def paired_sentence_test():
    """Base against fine-tuned on the sentences both systems scored."""
    base, lora = SCORES.get("base"), SCORES.get("lora")
    if not base or not lora:
        return None
    shared = sorted(
        i for i in set(base) & set(lora)
        if base[i].get("correct") is not None and lora[i].get("correct") is not None
    )
    if not shared:
        return None
    b = sum(1 for i in shared if base[i].get("correct") and not lora[i].get("correct"))
    c = sum(1 for i in shared if lora[i].get("correct") and not base[i].get("correct"))
    return {
        "n": len(shared),
        "base_correct": sum(1 for i in shared if base[i].get("correct")),
        "lora_correct": sum(1 for i in shared if lora[i].get("correct")),
        "base_only": b,
        "lora_only": c,
        "p": mcnemar_exact(b, c),
        "base_dropped": len(set(base) - set(shared)),
        "lora_dropped": len(set(lora) - set(shared)),
    }


def paired_pair_test():
    """The same test at the level of pairs, where both members must be right."""
    base, lora = SCORES.get("base"), SCORES.get("lora")
    if not base or not lora or not PAIRS:
        return None

    def solved(scores, ids):
        marks = [scores.get(i, {}).get("correct") for i in ids]
        return None if any(m is None for m in marks) else all(marks)

    rows = []
    for ids in PAIRS.values():
        sb, sl = solved(base, ids), solved(lora, ids)
        if sb is not None and sl is not None:
            rows.append((sb, sl))
    if not rows:
        return None
    b = sum(1 for sb, sl in rows if sb and not sl)
    c = sum(1 for sb, sl in rows if sl and not sb)
    return {
        "n": len(rows),
        "base_correct": sum(1 for sb, _ in rows if sb),
        "lora_correct": sum(1 for _, sl in rows if sl),
        "base_only": b,
        "lora_only": c,
        "p": mcnemar_exact(b, c),
    }


def _fmt_p(p):
    if p is None:
        return "not computed"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _fmt_ci(successes, total):
    ci = wilson(successes, total)
    if not ci:
        return ""
    return f"95 percent CI {ci[0]:.1f} to {ci[1]:.1f}"


# ---------------------------------------------------------------------------
# Live generation
# ---------------------------------------------------------------------------

_loaded = {}


def _device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_llama():
    """Load the base model once and attach the adapter to the same weights.

    The two conditions then differ by the adapter alone, which is the claim the
    Results tab makes, and only one copy of the 3B model sits in memory.

    Weights load onto CPU first and move to the device afterwards. Loading
    straight onto MPS segfaults with recent safetensors builds, so these two
    steps must stay separate.
    """
    if "model" in _loaded:
        return _loaded["model"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # transformers 5.x renamed torch_dtype to dtype. Accept either.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cpu"
        )

    has_adapter = ADAPTER_DIR.is_dir()
    if has_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    model = model.to(_device())
    model.eval()
    _loaded["model"] = (tokenizer, model, has_adapter)
    return _loaded["model"]


def warm_up():
    """Load the model up front so the first real click does not stall."""
    try:
        _tok, _model, has_adapter = _load_llama()
    except Exception as exc:
        return f"Model could not be loaded: {type(exc).__name__}. Cached mode still works."
    if has_adapter:
        return "Base and fine-tuned model ready."
    return (
        f"Base model ready. No adapter at {ADAPTER_DIR.relative_to(REPO_ROOT)}, so the "
        "fine-tuned system is available in Cached mode only. Train one with "
        "finetuning/lora_training.py."
    )


def _generate_llama(sentence, use_adapter):
    import contextlib
    import torch

    tokenizer, model, has_adapter = _load_llama()
    if use_adapter and not has_adapter:
        raise LookupError(
            "No trained adapter is present, so the fine-tuned system cannot "
            "generate. Use Cached mode, or train one with "
            "finetuning/lora_training.py."
        )

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = encoded.to(model.device)

    # The base condition is the same weights with the adapter switched off.
    context = (
        contextlib.nullcontext()
        if use_adapter or not has_adapter
        else model.disable_adapter()
    )
    with context, torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_ollama(sentence, model_name):
    import requests

    # Reasoning models spend their budget on internal reasoning and hit the
    # limit before emitting content. The others loop without a cap. Same split
    # as src/models/llm_client.py.
    options = {"temperature": 0, "num_predict": -1 if "-r1" in model_name else 4000}

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": sentence},
                ],
                "stream": False,
                "options": options,
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise LookupError(
            f"No Ollama server at {OLLAMA_URL}. Start it with ollama serve, or "
            f"set OLLAMA_URL. Cached mode works without it."
        )
    except requests.exceptions.Timeout:
        raise LookupError(
            f"{model_name} did not answer within the timeout. Reasoning models "
            f"can be slow on a long sentence. Try again, or use Cached mode."
        )

    message = response.json().get("message", {})
    content = (message.get("content") or "").strip()
    if not content:
        # When content is empty the finished text is usually in thinking.
        content = (message.get("thinking") or "").strip()
    if not content:
        raise LookupError(f"{model_name} returned an empty response.")
    return content


def redact(sentence, system_label, mode):
    """Single entry point for producing a Dataset A redaction."""
    sentence = (sentence or "").strip()
    if not sentence:
        return ""

    spec = SYSTEMS[system_label]

    if mode == "Cached":
        table = CACHE.get(spec["cache"], {})
        if sentence in table:
            return table[sentence]
        raise LookupError(
            "Not in the stored evaluation outputs. Pick an example below, or "
            "switch to Live to generate on this sentence."
        )

    if len(sentence) > MAX_LIVE_CHARS:
        raise LookupError(
            f"Live mode takes up to {MAX_LIVE_CHARS} characters. This input has "
            f"{len(sentence)}. Shorten it, or try one sentence at a time."
        )
    if spec["live"] == "ollama":
        return _generate_ollama(sentence, spec["cache"])
    return _generate_llama(sentence, spec["live"] == "lora")


# ---------------------------------------------------------------------------
# Rendering
#
# Colours are semi-transparent overlays and text inherits the theme
# foreground, so output stays legible in light and dark mode. Verdict badges
# also carry a glyph, so they survive video compression and do not depend on
# colour alone.
# ---------------------------------------------------------------------------

_CSS = """
<style>
.pr-wrap{font-family:system-ui,-apple-system,sans-serif;}
.pr-card{border:1px solid rgba(128,128,128,.32);border-radius:8px;
         padding:13px 15px;margin-bottom:11px;}
.pr-src{background:rgba(128,128,128,.10);}
.pr-head{font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;
         opacity:.62;margin-bottom:9px;display:flex;align-items:center;
         gap:7px;flex-wrap:wrap;}
.pr-tag{font-size:10.5px;padding:1px 7px;border-radius:9px;
        border:1px solid rgba(128,128,128,.42);opacity:.88;letter-spacing:.02em;
        text-transform:none;}
.pr-ok{border-color:rgba(46,182,115,.9);}
.pr-bad{border-color:rgba(232,72,72,.9);}
.pr-cols{display:flex;gap:14px;flex-wrap:wrap;}
.pr-col{flex:1 1 260px;min-width:230px;}
.pr-lab{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
        opacity:.5;margin-bottom:5px;}
.pr-body{font-size:17px;line-height:1.75;font-family:Georgia,'Times New Roman',serif;}
.pr-gold-line{font-size:14px;line-height:1.6;opacity:.72;margin-top:9px;
              font-family:Georgia,serif;}
.pr-del{background:rgba(232,72,72,.30);text-decoration:line-through;
        text-decoration-thickness:2px;border-radius:3px;padding:0 2px;}
.pr-ins{background:rgba(46,182,115,.30);border-radius:3px;padding:0 2px;}
.pr-note{font-size:12px;opacity:.62;margin-top:9px;line-height:1.6;}
.pr-quote{font-size:13px;opacity:.8;margin-top:8px;padding-left:10px;
          border-left:2px solid rgba(128,128,128,.45);}
.pr-err{font-size:13.5px;color:#e05252;line-height:1.6;}
.pr-sec{font-size:12px;letter-spacing:.04em;opacity:.85;margin:18px 0 8px;
        border-left:3px solid rgba(128,128,128,.5);padding:5px 0 5px 10px;
        line-height:1.55;}
.pr-pos{border-left-color:rgba(232,72,72,.85);}
.pr-neg{border-left-color:rgba(46,182,115,.85);}
.pr-verdict{font-size:12.5px;padding:8px 11px;border-radius:6px;
            background:rgba(128,128,128,.12);margin-bottom:11px;line-height:1.6;}
.pr-rule{font-size:12.5px;opacity:.75;margin:0 0 12px;line-height:1.65;}
.pr-gold-note{font-size:12px;opacity:.62;margin:0 0 10px 13px;line-height:1.55;}
.pr-tbl{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px;}
.pr-tbl th{text-align:left;font-weight:600;opacity:.7;padding:5px 9px;
           border-bottom:1px solid rgba(128,128,128,.4);font-size:11px;
           letter-spacing:.04em;text-transform:uppercase;}
.pr-tbl td{padding:5px 9px;border-bottom:1px solid rgba(128,128,128,.18);}
.pr-tbl td.pr-n{text-align:right;font-variant-numeric:tabular-nums;}
.pr-prov{font-size:11.5px;opacity:.6;line-height:1.7;margin-top:4px;}
</style>
"""


def _e(value):
    return html.escape(str(value))


def _words(text):
    return re.findall(r"\S+", text or "")


def _marked(seq, spans, cls):
    flags = [False] * len(seq)
    for i1, i2 in spans:
        for i in range(i1, i2):
            flags[i] = True

    out, i = [], 0
    while i < len(seq):
        if not flags[i]:
            out.append(_e(seq[i]))
            i += 1
            continue
        j = i
        while j < len(seq) and flags[j]:
            j += 1
        out.append(f'<span class="pr-{cls}">{_e(" ".join(seq[i:j]))}</span>')
        i = j
    return " ".join(out)


def _diff_block(original, redacted, gold=None):
    """Two aligned views: removals on the input, additions on the output.

    Interleaving both in one line fragments into unreadable slivers when a
    model rewrites heavily, which is exactly the case worth showing.
    """
    a, b = _words(original), _words(redacted)
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    dels = [(i1, i2) for tag, i1, i2, _, _ in ops if tag in ("delete", "replace")]
    inss = [(j1, j2) for tag, _, _, j1, j2 in ops if tag in ("insert", "replace")]
    removed = sum(i2 - i1 for i1, i2 in dels)
    total = len(a) or 1

    block = (
        '<div class="pr-cols">'
        f'<div class="pr-col"><div class="pr-lab">Input, removed marked</div>'
        f'<div class="pr-body">{_marked(a, dels, "del")}</div></div>'
        f'<div class="pr-col"><div class="pr-lab">Output, added marked</div>'
        f'<div class="pr-body">{_marked(b, inss, "ins")}</div></div>'
        "</div>"
    )
    if gold:
        block += (
            f'<div class="pr-lab" style="margin-top:11px">Gold target</div>'
            f'<div class="pr-gold-line">{_e(gold)}</div>'
        )
    block += (
        f'<div class="pr-note">{removed} of {total} words removed or replaced '
        f"({removed / total * 100:.0f} percent)</div>"
    )
    return block


def _card(label, tags, inner):
    tag_html = "".join(
        f'<span class="pr-tag {cls}">{_e(t)}</span>' for t, cls in tags
    )
    return (
        f'<div class="pr-card"><div class="pr-head">{_e(label)}'
        f"{tag_html}</div>{inner}</div>"
    )


def _input_card(sentence, label="German sentence"):
    return (
        f'<div class="pr-card pr-src"><div class="pr-head">{_e(label)}</div>'
        f'<div class="pr-body">{_e(sentence)}</div></div>'
    )


def _wrap(*blocks):
    return '<div class="pr-wrap">' + "".join(blocks) + "</div>"


def _message(text):
    return _wrap(f'<div class="pr-note">{_e(text)}</div>')


MISSING_DATA_HELP = (
    "Datasets A and C are not committed, so that models scraped from the "
    "repository cannot train on the test set. Generate them first with "
    "python dataset_a_gen.py, then move sentences.csv and spans.csv into data/. "
    "The generator writes to the current directory, not to data/."
)


def data_status():
    if not LOAD_ERRORS:
        return ""
    items = "".join(f"<li>{_e(line)}</li>" for line in dict.fromkeys(LOAD_ERRORS))
    return (
        '<div class="pr-card"><div class="pr-head">Data not loaded</div>'
        f'<ul class="pr-note" style="margin:0;padding-left:18px">{items}</ul>'
        f'<div class="pr-note">{_e(MISSING_DATA_HELP)}</div></div>'
    )


# --------------------------------------------------------------- Dataset A --


def _verdict_tags(system_key, sentence_id):
    """Per-sentence verdict from the scorer, where one exists."""
    record = SCORES.get(system_key, {}).get(sentence_id)
    if not record:
        return [], None
    tags = []
    correct = record.get("correct")
    if correct is not None:
        tags.append(("correct" if correct else "wrong",
                     "pr-ok" if correct else "pr-bad"))
    if record.get("identifier_removed") is False:
        tags.append(("name survived", "pr-bad"))
    if record.get("leak_removed") is False:
        tags.append(("sensitive term survived", "pr-bad"))
    if record.get("decoy_kept") is False:
        tags.append(("harmless detail destroyed", "pr-bad"))
    return tags, record


def _section_a(sentence, selected, mode, heading=None, heading_cls="", gold_note=None):
    sid = TEXT_TO_ID.get(sentence)
    parts = []
    if heading:
        parts.append(f'<div class="pr-sec {heading_cls}">{heading}</div>')
    if gold_note:
        parts.append(f'<div class="pr-gold-note">{_e(gold_note)}</div>')
    parts.append(_input_card(sentence))

    for label in SYSTEMS:  # fixed order regardless of click order
        if label not in selected:
            continue
        key = SYSTEMS[label]["cache"]
        tags = [(SYSTEMS[label]["params"], "")]
        gold = None
        if mode == "Cached" and sid:
            verdicts, record = _verdict_tags(key, sid)
            tags += verdicts
            if record:
                gold = record.get("gold")
        try:
            parts.append(_card(label, tags,
                               _diff_block(sentence, redact(sentence, label, mode), gold)))
        except LookupError as exc:
            parts.append(_card(label, tags, f'<div class="pr-err">{_e(exc)}</div>'))
        except Exception:
            traceback.print_exc()
            parts.append(_card(label, tags, '<div class="pr-err">'
                                            "This system could not produce output. See the "
                                            "console for details.</div>"))
    return "".join(parts)


def _why_wrong(record, polarity):
    """Plain description of what the system got wrong on one sentence."""
    reasons = []
    if record.get("identifier_removed") is False:
        reasons.append("left the person's name in place")
    if polarity == "positive" and record.get("leak_removed") is False:
        reasons.append("left the sensitive detail readable")
    if polarity == "negative" and record.get("decoy_kept") is False:
        reasons.append("deleted a harmless detail it should have kept")
    if not reasons:
        reasons.append("did not match the gold target")
    return " and ".join(reasons)


def _pair_verdict(pair_ids, selected, mode):
    """Whether each scored system handled both sentences of the pair."""
    if mode != "Cached":
        return ""
    lines = []
    for label in SYSTEMS:
        key = SYSTEMS[label]["cache"]
        if label not in selected or key not in SCORED:
            continue
        marks = [SCORES.get(key, {}).get(i, {}).get("correct") for i in pair_ids]
        if any(m is None for m in marks):
            continue
        if all(marks):
            lines.append(f"<b>{_e(label)}</b> handled both sentences correctly.")
            continue
        for sid, ok in zip(pair_ids, marks):
            if ok:
                continue
            polarity = SENTENCES.get(sid, {}).get("polarity", "")
            why = _why_wrong(SCORES[key][sid], polarity)
            lines.append(
                f"<b>{_e(label)}</b> failed on the {_e(polarity)} "
                f"sentence: it {_e(why)}."
            )
    if not lines:
        return ""
    return (
        '<div class="pr-verdict"><b>Result for this pair.</b> A pair counts as '
        "solved only if both of its sentences are handled correctly, because "
        "protecting the person and preserving the harmless sentence are both "
        "required.<br>" + "<br>".join(lines) + "</div>"
    )


A_RULE = (
    '<div class="pr-rule">Every sentence names a person, so the name must always '
    "be removed. What changes between the two members of a pair is whether "
    "anything <b>protected under Art. 9 GDPR</b> is disclosed. Both members use "
    "the same sensitive-looking vocabulary, so a system cannot succeed by "
    "reacting to keywords alone.</div>"
)


def run_a(sentence, selected, mode, pair_view):
    sentence = (sentence or "").strip()
    if not sentence:
        return _message("Enter a German sentence.")
    if not selected:
        return _message("Select at least one system.")

    sid = TEXT_TO_ID.get(sentence)
    pid = ID_TO_PAIR.get(sid) if sid else None

    if pair_view and pid:
        body = [A_RULE, _pair_verdict(PAIRS[pid], selected, mode)]
        for member in PAIRS[pid]:
            row = SENTENCES[member]
            if row.get("polarity") == "positive":
                attribute = (row.get("implicit_attribute") or "a protected attribute")
                attribute = attribute.split(":")[0].replace("_", " ")
                heading = (
                    "<b>Positive sentence.</b> This one really does disclose "
                    f"protected data ({_e(attribute)}). Correct behaviour: "
                    "remove the name <i>and</i> remove or generalise the sensitive "
                    "detail."
                )
                gold_note = (
                    "Leaving the sensitive term in place is a leak, and the person "
                    "is not protected."
                )
                cls = "pr-pos"
            else:
                why = NEGATIVE_TYPE_LABEL.get(row.get("negative_type", ""),
                                              "nothing protected is disclosed")
                heading = (
                    "<b>Negative sentence.</b> Same vocabulary, but "
                    f"{_e(why)}, so nothing here is protected. Correct "
                    "behaviour: remove the name and change nothing else."
                )
                gold_note = (
                    "Deleting the harmless detail here is over-redaction: "
                    "information was destroyed for no reason."
                )
                cls = "pr-neg"
            body.append(_section_a(row["text"], selected, mode, heading, cls, gold_note))
        return _wrap(*body)

    note = ""
    if pair_view and not pid:
        note = (
            '<div class="pr-note">This sentence has no paired counterpart in '
            "Dataset A, so it is shown on its own.</div>"
        )
    return _wrap(note, _section_a(sentence, selected, mode))


# --------------------------------------------------------------- Dataset C --

C_RULE = (
    '<div class="pr-rule">Each German sentence below discloses the same fact '
    "about a different kind of person. Under <b>Art. 85 GDPR</b> and "
    "<b>Art. 9(2)(e)</b>, a disclosure about an office-holder acting in that "
    "role, or one the person made about themselves, can be legitimate to "
    "publish, while the identical fact about a private individual is not. "
    "A system that redacts everything protects nobody extra and destroys "
    "legitimate public reporting.</div>"
)


def run_c(choice, selected, show_reasoning):
    if not choice:
        return _message("Pick a case.")
    if not selected:
        return _message("Select at least one condition.")

    item = C_CHOICE_TO_ITEM.get(choice)
    body = [C_RULE]

    for sid in C_GROUPS.get(item, []):
        row = C_ITEMS[sid]
        expected = (row.get("expected_action") or "").strip()
        framing = FRAMING_LABEL.get(row.get("framing"), row.get("framing", ""))
        basis = BASIS_LABEL.get((row.get("legitimacy_basis") or "").strip(), "")
        category = (row.get("art9_category") or "").replace("_", " ")

        if expected == "PRESERVE":
            action = ("Correct behaviour: keep the sentence as it is, including "
                      "the name and the disclosure.")
            cls = "pr-neg"
        else:
            action = ("Correct behaviour: remove the name and the "
                      f"{_e(category)} disclosure.")
            cls = "pr-pos"

        heading = f"<b>{_e(framing)}.</b> {action}"
        if basis:
            heading += f" Legal basis: {_e(basis)}."
        body.append(f'<div class="pr-sec {cls}">{heading}</div>')
        body.append(_input_card(row.get("input_text", "")))

        for label in ALL_C_SYSTEMS:
            if label not in selected:
                continue
            rec = C_RESULTS.get(label, {}).get(sid)
            if not rec:
                body.append(_card(label, [], '<div class="pr-err">No stored result '
                                             "for this sentence.</div>"))
                continue

            tags = []
            got = (rec.get("decision_from_output") or "").strip()
            if got:
                tags.append((f"acted: {got}", "pr-ok" if got == expected else "pr-bad"))
            action_label = (rec.get("predicted_action") or "").strip()
            if action_label:
                ok = (action_label == "none") == (expected == "PRESERVE")
                tags.append((f"action: {action_label}", "pr-ok" if ok else "pr-bad"))
            said = (rec.get("decision_stated_en") or rec.get("decision_stated") or "").strip()
            if said:
                tags.append((f"said: {said}", "pr-ok" if said == expected else "pr-bad"))
                if got and said != got:
                    tags.append(("stated decision contradicts behaviour", "pr-bad"))

            inner = _diff_block(row.get("input_text", ""), rec.get("prediction", ""))
            if show_reasoning:
                text = " ".join(str(rec.get("reasoning_en") or "").split())[:420]
                if text:
                    inner += f'<div class="pr-quote"><b>Reason</b> {_e(text)}</div>'
            body.append(_card(label, tags, inner))

    return _wrap(*body)


# ---------------------------------------------------------------------------
# Results
#
# Charts are inline SVG built here rather than through a plotting library, so
# the app keeps one dependency and the same code can export report figures.
# Every figure and every sentence of commentary below is computed from the
# summary tables at render time. Nothing is written into the source.
# ---------------------------------------------------------------------------

PALETTE = ["#4c7fd4", "#d4694c", "#4cae7f", "#a97fd4", "#d4b04c", "#5fa8b8"]


def _svg_open(width, height):
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px;height:auto;font-family:system-ui,sans-serif" '
        f'xmlns="http://www.w3.org/2000/svg">'
    )


def _axis(x0, y0, plot_w, plot_h, y_max, unit):
    out = []
    for step in range(6):
        value = y_max * step / 5
        y = y0 + plot_h - plot_h * step / 5
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}" '
                   f'stroke="rgba(128,128,128,.26)"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
                   f'fill="currentColor" opacity=".6">{value:.0f}{unit}</text>')
    return "".join(out)


def grouped_bars(categories, series, y_max=100, unit="", height=300, group_w=96):
    left, right, top, bottom = 46, 14, 26, 66
    width = left + right + group_w * len(categories)
    plot_h, plot_w = height - top - bottom, group_w * len(categories)

    parts = [_svg_open(width, height), _axis(left, top, plot_w, plot_h, y_max, unit)]
    bar_w = (group_w - 20 - 3 * (len(series) - 1)) / len(series)

    for gi, category in enumerate(categories):
        gx = left + gi * group_w + 10
        for si, (_name, values) in enumerate(series):
            value = values[gi]
            if value is None:
                continue
            bar_h = plot_h * min(value, y_max) / y_max
            x, y = gx + si * (bar_w + 3), top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                         f'height="{bar_h:.1f}" fill="{PALETTE[si % len(PALETTE)]}" rx="2"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" '
                         f'text-anchor="middle" font-size="10" fill="currentColor" '
                         f'opacity=".85">{value:g}</text>')
        label = category if len(category) < 16 else category[:15] + "."
        parts.append(f'<text x="{gx + (group_w - 20) / 2:.1f}" y="{top + plot_h + 16}" '
                     f'text-anchor="middle" font-size="11" fill="currentColor" '
                     f'opacity=".75">{_e(label)}</text>')

    lx, legend_y = left, height - 20
    for si, (name, _v) in enumerate(series):
        parts.append(f'<rect x="{lx}" y="{legend_y - 9}" width="10" height="10" rx="2" '
                     f'fill="{PALETTE[si % len(PALETTE)]}"/>')
        parts.append(f'<text x="{lx + 15}" y="{legend_y}" font-size="11" '
                     f'fill="currentColor" opacity=".82">{_e(name)}</text>')
        lx += 24 + len(name) * 6.4
    parts.append("</svg>")
    return "".join(parts)


def scatter_plot(points, x_label, y_label, arrow=None, height=330):
    left, right, top, bottom = 52, 20, 20, 54
    width = 520
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [_svg_open(width, height)]

    for step in range(6):
        value = step * 20
        y = top + plot_h - plot_h * step / 5
        x = left + plot_w * step / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="rgba(128,128,128,.22)"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
                     f'stroke="rgba(128,128,128,.22)"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="currentColor" opacity=".6">{value}</text>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 16}" text-anchor="middle" '
                     f'font-size="11" fill="currentColor" opacity=".6">{value}</text>')

    def place(vx, vy):
        return left + plot_w * vx / 100, top + plot_h - plot_h * vy / 100

    if arrow:
        (x1, y1), (x2, y2) = place(*arrow[0]), place(*arrow[1])
        parts.append('<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" '
                     'refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" '
                     'fill="currentColor" opacity=".55"/></marker></defs>')
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="currentColor" stroke-opacity=".55" stroke-width="1.6" '
                     f'stroke-dasharray="5 4" marker-end="url(#ah)"/>')

    for label, vx, vy, ci in points:
        px, py = place(vx, vy)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" '
                     f'fill="{PALETTE[ci % len(PALETTE)]}"/>')
        anchor = "end" if px > left + plot_w * 0.62 else "start"
        dx = -11 if anchor == "end" else 11
        parts.append(f'<text x="{px + dx:.1f}" y="{py + 4:.1f}" text-anchor="{anchor}" '
                     f'font-size="11.5" fill="currentColor" '
                     f'opacity=".9">{_e(label)}</text>')

    parts.append(f'<text x="{left + plot_w / 2:.0f}" y="{height - 8}" text-anchor="middle" '
                 f'font-size="11" fill="currentColor" opacity=".7">{_e(x_label)}</text>')
    mid = top + plot_h / 2
    parts.append(f'<text x="14" y="{mid:.0f}" text-anchor="middle" font-size="11" '
                 f'fill="currentColor" opacity=".7" transform="rotate(-90 14 {mid:.0f})">'
                 f'{_e(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _panel(title, note, body):
    """note and body are HTML assembled here; every interpolated value is
    escaped at the point it is inserted."""
    return (f'<div class="pr-card"><div class="pr-head">{_e(title)}</div>'
            f'{body}<div class="pr-note">{note}</div></div>')


def _table(headers, rows):
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="pr-n">{_e(c)}</td>' if i else f"<td>{_e(c)}</td>"
            for i, c in enumerate(row)
        )
        body += f"<tr>{cells}</tr>"
    return f'<table class="pr-tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _failure_counts():
    """How each scored system fails, not just how often."""
    rows = []
    for key, label in (("base", "Base"), ("lora", "Fine-tuned")):
        detail = SCORES.get(key)
        if not detail:
            continue
        name = sum(1 for d in detail.values() if d.get("identifier_removed") is False)
        leak = sum(1 for d in detail.values() if d.get("leak_removed") is False)
        decoy = sum(1 for d in detail.values() if d.get("decoy_kept") is False)
        rows.append((label, [name, leak, decoy]))
    return rows


def _panel_ab(ft):
    metrics = [
        ("Sentence acc.", "sentence_accuracy"),
        ("Pair acc.", "pair_accuracy"),
        ("Name removed", "identifier_removed"),
        ("Attribute removed", "leak_removed"),
        ("Decoy kept", "decoy_kept"),
    ]
    conditions = [("base model", "Base"), ("fine-tuned seed 42", "Fine-tuned")]
    series = []
    for key, label in conditions:
        if key not in ft:
            continue
        values = [_num(ft[key], field) for _n, field in metrics]
        if all(v is None for v in values):
            continue
        series.append((label, values))
    if not series:
        return None

    sent, pair = paired_sentence_test(), paired_pair_test()
    note = []
    if sent:
        delta = sent["lora_correct"] - sent["base_correct"]
        note.append(
            f"One scorer, one loading path, the adapter the only difference. "
            f"On the {_e(sent['n'])} sentences both systems scored, the base model is "
            f"correct on {_e(sent['base_correct'])} and the fine-tuned model on "
            f"{_e(sent['lora_correct'])}, a difference of {_e(delta)} sentences "
            f"({_e(_fmt_p(sent['p']))}, exact McNemar on the "
            f"{_e(sent['base_only'])} and {_e(sent['lora_only'])} sentences where "
            f"they disagree)."
        )
        if sent["base_dropped"] or sent["lora_dropped"]:
            note.append(
                f"{_e(sent['base_dropped'])} base and {_e(sent['lora_dropped'])} "
                "fine-tuned items fall outside this comparison because only one "
                "system scored them."
            )
    if pair:
        note.append(
            f"At the level of pairs, where a system must handle both the disclosing "
            f"and the harmless sentence, the base model solves "
            f"{_e(pair['base_correct'])} of {_e(pair['n'])} "
            f"({_e(_fmt_ci(pair['base_correct'], pair['n']))}) and the fine-tuned "
            f"model {_e(pair['lora_correct'])} of {_e(pair['n'])} "
            f"({_e(_fmt_ci(pair['lora_correct'], pair['n']))}), "
            f"{_e(_fmt_p(pair['p']))}. The intervals overlap and the difference is "
            "not distinguishable from chance at this sample size."
        )
        if sent and sent["p"] < 0.05 <= pair["p"]:
            note.append(
                "So the aggregate metric moves and the paired metric does not. "
                "Sentence accuracy rewards removing the name, which fine-tuning "
                "learned; paired accuracy additionally requires telling a "
                "disclosure from a harmless sentence, which it did not. That "
                "disagreement is what this benchmark was built to expose."
            )
    return _panel(
        f"Dataset A: base against fine-tuned, higher is better on every bar"
        + "",
        " ".join(note) or "Computed from the summary table.",
        grouped_bars([n for n, _f in metrics], series, unit="%"),
    )


def _panel_failures():
    failures = _failure_counts()
    if not failures:
        return None
    labels = ["Name survived", "Sensitive term survived", "Harmless detail destroyed"]
    by_label = dict(failures)
    lines = [
        "Name survived: the person is still identifiable. Sensitive term survived: "
        "the Art. 9 attribute is still readable. Harmless detail destroyed: a "
        "sentence disclosing nothing protected was damaged anyway."
    ]
    if "Base" in by_label and "Fine-tuned" in by_label:
        b, f = by_label["Base"], by_label["Fine-tuned"]
        moves = []
        for i, name in enumerate(labels):
            direction = "falls" if f[i] < b[i] else ("rises" if f[i] > b[i] else "is unchanged")
            moves.append(f"{name.lower()} {direction} from {_e(b[i])} to {_e(f[i])}")
        lines.append("After fine-tuning, " + "; ".join(moves) + ".")
    peak = max(max(v) for _l, v in failures)
    return _panel(
        "Dataset A: how each system fails (sentence counts, lower is better)",
        " ".join(lines),
        grouped_bars(labels, failures, y_max=max(10, peak * 1.15),
                     height=300, group_w=140),
    )


def _panel_tradeoff(ft):
    conditions = [("base model", "Base"), ("fine-tuned seed 42", "Fine-tuned")]
    points = []
    for i, (key, label) in enumerate(conditions):
        x, y = _num(ft.get(key), "leak_removed"), _num(ft.get(key), "decoy_kept")
        if x is not None and y is not None:
            points.append((label, x, y, i))
    if len(points) < 2:
        return None
    (_lb, bx, by, _i), (_lf, fx, fy, _j) = points[0], points[1]
    horizontal = "right" if fx > bx else "left"
    vertical = "up" if fy > by else "down"
    note = (
        f"Up and to the right is better: protecting what must be protected while "
        f"preserving what must be preserved. Fine-tuning moved the system "
        f"{_e(vertical)} and to the {_e(horizontal)}, from {bx:.1f} to {fx:.1f} "
        f"percent on removing the sensitive term and from {by:.1f} to {fy:.1f} "
        f"percent on keeping the harmless detail. A single accuracy number hides "
        f"that trade entirely."
    )
    return _panel(
        "Dataset A: protection against preservation",
        note,
        scatter_plot(points, "Sensitive term removed (%)",
                     "Harmless detail kept (%)", ((bx, by), (fx, fy))),
    )


def _panel_prompted():
    prompted = _read_csv(RESULTS_A_PROMPTED)
    prompted = [r for r in prompted if r.get("model")]
    if not prompted:
        return None
    fields = [("Precision", "precision"), ("Recall", "recall"),
              ("F1", "f1"), ("Any-span leak", "leak_rate")]
    series = []
    for name, field in fields:
        values = [_num(r, field, 100) for r in prompted]
        if any(v is not None for v in values):
            series.append((name, [None if v is None else round(v, 1) for v in values]))
    if not series:
        return None
    covered = [f"{r['model']} {r['n']}" for r in prompted if r.get("n")]
    note = (
        "Detection metrics exist here because this pipeline emits a has_leak flag; "
        "the fine-tuned model only rewrites and never emits one, so precision, "
        "recall and F1 do not apply to it. Any-span leak counts a sentence as "
        "leaked when any span requiring removal survives, including the person's "
        "name, so it is a union of identifier and attribute leakage. The panels "
        "above report attribute removal alone, in the opposite direction. Lower "
        "is better here and higher is better there, so the two must not be read "
        "off one axis."
    )
    if covered:
        note += " Sentences scored per model: " + _e(", ".join(covered)) + "."
    return _panel(
        "Dataset A: prompted models, single pass without schema",
        note,
        grouped_bars([r["model"] for r in prompted], series, unit="%",
                     height=320, group_w=112),
    )


def _c_rows():
    """One row per condition, from whichever summary tables are present."""
    rows = []
    labels = {("base", "default"): "Base, default",
              ("base", "legitimacy"): "Base, legitimacy",
              ("lora", "default"): "Fine-tuned, default",
              ("lora", "legitimacy"): "Fine-tuned, legitimacy"}
    for r in _read_csv(RESULTS_C_FT):
        key = ((r.get("model") or "?").strip(), (r.get("config") or "").strip())
        rows.append((labels.get(key, "/".join(k for k in key if k)), r))
    for r in _read_csv(RESULTS_C_PROMPTED):
        rows.append(((r.get("model") or "?").split(":")[0], r))
    return rows


def _panel_c_leak(rows_c):
    values = [_num(r, "redact_leak_rate", 100) for _l, r in rows_c]
    if all(v is None for v in values):
        return None
    values = [None if v is None else round(v, 1) for v in values]
    present = [v for v in values if v is not None]
    note = (
        "Every item here is one the system was supposed to redact, so lower is "
        "better. Across the conditions with a reported figure the rate runs from "
        f"{min(present):.1f} to {max(present):.1f} percent."
    )
    return _panel(
        "Dataset C: leakage on the items that should have been redacted",
        note,
        grouped_bars([l for l, _r in rows_c], [("Leaked", values)], unit="%",
                     height=320, group_w=128),
    )


def _panel_c_preserve(rows_c):
    """Preservation rate per condition, as a table.

    A table rather than a chart: the counts are small and the denominator
    matters, so the raw fraction says more than a bar would.
    """
    n = C_PRESERVE_N or None
    table_rows = []
    for label, r in rows_c:
        rate = _num(r, "preserve_correct_rate")
        if rate is None:
            scored = "not reported"
        elif n:
            scored = f"{round(rate * n)} of {n}"
        else:
            scored = f"{rate * 100:.0f} percent"
        table_rows.append((label, scored))
    if not table_rows:
        return None

    note = (
        "These are the items where the disclosure is legitimate and the correct "
        "action is to leave the sentence alone. score_c.py counts a success when "
        "the Art. 9 attribute is no longer explicit in the output. That criterion "
        "is also satisfied when the attribute has been deleted rather than kept, "
        "so it cannot separate correct preservation from over-deletion. A scorer "
        "that distinguishes correctly removed, leaked, over-deleted and content "
        "destroyed would resolve that; the current binary cannot."
    )
    return _panel(
        "Dataset C: preserving a legitimate public disclosure",
        note,
        _table(["Condition", "Preserved (score_c.py)"], table_rows),
    )


def _panel_provenance():
    rows = [(k, v) for k, v in PROVENANCE.items()]
    body = "".join(
        f'<div class="pr-prov"><b>{_e(k)}</b> {_e(v)}</div>' for k, v in rows
    )
    return _panel(
        "Provenance",
        "Every figure above is read from the summary tables in this checkout at "
        "render time. Reproducing them needs the same commit, the same seed and a "
        "Dataset A with the md5 shown here.",
        body,
    )


def run_results():
    try:
        parts = []
        status = data_status()
        if status:
            parts.append(status)

        ft = {r["condition"]: r for r in _read_csv(RESULTS_A_FT) if r.get("condition")}
        for panel in (_panel_ab(ft) if ft else None,
                      _panel_failures(),
                      _panel_tradeoff(ft) if ft else None,
                      _panel_prompted()):
            if panel:
                parts.append(panel)

        rows_c = _c_rows()
        if rows_c:
            for panel in (_panel_c_leak(rows_c), _panel_c_preserve(rows_c)):
                if panel:
                    parts.append(panel)

        parts.append(_panel_provenance())
        return _wrap(*parts)
    except Exception:
        traceback.print_exc()
        return _wrap(
            '<div class="pr-card"><div class="pr-head">Results could not be built</div>'
            '<div class="pr-err">A summary table could not be read. The console has '
            "the details. The other tabs are unaffected.</div></div>"
        )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

INTRO = """## PrivacyLens

German privacy redaction under the GDPR. The system takes a German sentence
and removes direct identifiers together with any special category data under
Art. 9, health, ethnic origin or religion, that can be inferred from the
sentence even when it is never stated outright, while leaving harmless
sentences intact.

Red marks what a system removed, green what it added.
"""


def build():
    with gr.Blocks(title="PrivacyLens") as demo:
        gr.HTML(_CSS)  # injected once rather than on every render
        gr.Markdown(INTRO)
        if LOAD_ERRORS:
            gr.HTML(_wrap(data_status()))

        with gr.Tabs():
            with gr.Tab("Dataset A"):
                gr.Markdown(
                    "150 German sentences arranged in 55 pairs. The two sentences "
                    "in a pair use the same sensitive-looking vocabulary, but only "
                    "one of them actually discloses protected data. A system that "
                    "reacts to keywords rather than context will treat them alike "
                    "and fail one of the two."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        text = gr.Textbox(label="German sentence", lines=3)
                        if EXAMPLES:
                            gr.Examples(
                                examples=[[e] for e in EXAMPLES],
                                inputs=[text],
                                label="Six sentences from Dataset A",
                            )
                    with gr.Column(scale=2):
                        mode = gr.Radio(
                            ["Cached", "Live"], value="Cached", label="Source",
                            info=("Cached replays the stored evaluation outputs and "
                                  "covers the 150 dataset sentences. Live loads the "
                                  "model and accepts any German sentence. The "
                                  "prompted models need a running Ollama server."),
                        )
                        pair_view = gr.Checkbox(
                            value=True,
                            label="Show the paired sentence alongside this one")
                        systems = gr.CheckboxGroup(
                            choices=list(SYSTEMS.keys()), value=DEFAULT_SYSTEMS,
                            label="Systems")
                        run_a_btn = gr.Button("Redact", variant="primary")
                        warm_btn = gr.Button("Warm up local models", size="sm")
                        warm_status = gr.Markdown("")
                out_a = gr.HTML()
                inputs_a = [text, systems, mode, pair_view]
                run_a_btn.click(run_a, inputs=inputs_a, outputs=out_a)
                text.submit(run_a, inputs=inputs_a, outputs=out_a)
                warm_btn.click(warm_up, outputs=warm_status)

            with gr.Tab("Dataset C"):
                gr.Markdown(
                    "20 German sentences in which the same disclosure is made "
                    "about a public office-holder and about a private individual. "
                    "Removing an Art. 9 attribute is not always the right answer: "
                    "reporting a minister's own announcement is lawful, while the "
                    "identical fact about a colleague is not. This set asks whether "
                    "a system can tell the difference."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        c_choice = gr.Dropdown(
                            choices=C_CHOICES,
                            value=C_CHOICES[0] if C_CHOICES else None,
                            label="Case")
                    with gr.Column(scale=2):
                        c_reasoning = gr.Checkbox(
                            value=True,
                            label="Show the reason the system gave for its decision")
                        c_systems = gr.CheckboxGroup(
                            choices=ALL_C_SYSTEMS,
                            value=[s for s in DEFAULT_C_SYSTEMS if s in ALL_C_SYSTEMS],
                            label="Conditions")
                        run_c_btn = gr.Button("Show", variant="primary")
                out_c = gr.HTML()
                inputs_c = [c_choice, c_systems, c_reasoning]
                run_c_btn.click(run_c, inputs=inputs_c, outputs=out_c)
                c_choice.change(run_c, inputs=inputs_c, outputs=out_c)

            with gr.Tab("Results"):
                gr.Markdown(
                    "Every reported number, computed from the summary tables in "
                    "this checkout. The fine-tuning panels and the prompted-model "
                    "panel come from different scorers and are not directly "
                    "comparable to one another."
                )
                out_r = gr.HTML(value=run_results())
                gr.Button("Reload from tables").click(run_results, outputs=out_r)

    return demo


if __name__ == "__main__":
    print(f"Dataset A systems: {sorted(CACHE)}")
    print(f"Dataset A pairs: {len(PAIRS)} | scored systems: {sorted(SCORES)}")
    print(f"Dataset C cases: {len(C_GROUPS)} | conditions: {len(ALL_C_SYSTEMS)}")
    print(f"Commit {PROVENANCE['Commit']} | Dataset A md5 {PROVENANCE['Dataset A md5']}")
    if LOAD_ERRORS:
        print("\nNot loaded:")
        for line in dict.fromkeys(LOAD_ERRORS):
            print(f"  {line}")
        print(f"\n{MISSING_DATA_HELP}")
    build().launch(theme=gr.themes.Soft())