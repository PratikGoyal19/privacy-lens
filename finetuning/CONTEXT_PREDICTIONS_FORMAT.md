# Predictions Format for Dataset C — Contract for score_c.py

Dataset C is scored differently from Dataset A/B, because it's testing a
different question (from Prof. Dahlmeier's feedback): not just "is this
sensitive", but "is it *still* sensitive given who it's about and why it's
being said". See the docstring at the top of `score_c.py` for the full
reasoning — this doc is just the format contract.

## predictions_c.csv columns

| Column | Type | Meaning |
|---|---|---|
| `sentence_id` | text | Must match an `id` in `dataset_c.csv` (e.g. `c_c01_private`) |
| `model` | text | e.g. `llama3.2`, `qwen2.5`, `ministral-3`, `gemma3` |
| `config` | text | e.g. `single_pass` |
| `predicted_action` | text | One of: `none`, `mask`, `rewrite`, `mask_and_rewrite` — **reuse the `action` field your pipeline already outputs, don't invent a new one** |
| `output_text` | text | The model's final output sentence |
| `num_llm_calls` | integer | How many LLM calls this sentence took |

### Example
```csv
sentence_id,model,config,predicted_action,output_text,num_llm_calls
c_c01_private,llama3.2,single_pass,mask_and_rewrite,"Der Abteilungsleiter [NAME] nahm an der Feier zum Tag der Deutschen Einheit teil.",1
c_c01_public_legitimate,llama3.2,single_pass,none,"Bundeskanzler Konrad Mahlberg nahm als bekennender Protestant am ökumenischen Gottesdienst zum Tag der Deutschen Einheit teil.",1
```

### Run it like this
```bash
python3 score_c.py --predictions predictions_c.csv --dataset dataset_c.csv --out results_table_c.csv
```

## Two output files, not one

`score_c.py` writes **two** files:
- `results_table_c.csv` — one row per model+config, summary numbers (REDACT leak rate, PRESERVE correct rate, cost, and an `empty_output_rows` count)
- `results_table_c_detail.csv` — one row per **sentence**, showing exactly which name/word triggered each leak or failure, and which of the 3 PRESERVE criteria failed

**A blank or whitespace-only `output_text` is never scored as a pass.** If a model call fails, times out, or errors and comes back empty, that row is excluded from the rate calculations and flagged with `EMPTY_OUTPUT` in the detail file — a crashed call can never silently count as a successful redaction.

**Use the detail file, not just the summary, when writing up results.** Because Dataset C's "attribute terms" are derived automatically (see the KNOWN LIMITATION note in `score_c.py`), a "failed" row with only one generic matched word deserves a quick human look before you call it a real leak — the detail file is what lets you do that check in minutes instead of re-reading all 20 sentences from scratch.

## Before running main.py against Dataset C — a compatibility note for Harshitha

`main.py` currently expects `data/sentences.csv` with a `text` column and a
`has_sensitive_attribute` column. `dataset_c.csv` uses different column
names (`input_text`, `expected_action`) and a 4-way action instead of a
binary label — so `main.py` will need a small adjustment (or a separate
small runner script) to load Dataset C correctly. This isn't something
`score_c.py` can work around on its own, since it only scores whatever
`predictions_c.csv` your pipeline already produced.

## Reminder for the report (Rajul's own note, worth repeating)

Dataset C has 20 sentences. That's enough for an honest **qualitative,
descriptive** discussion of patterns — it is **not** enough to run a
significance test or claim a precise percentage means something statistically
robust. `score_c.py` prints this reminder every time it runs; the report
should say it too.
