# Predictions Format — Contract Between Pipeline (Harshitha) and Scoring (Pratik)

`score.py` needs one CSV file, `predictions.csv`, with exactly these 6 columns.
One row = one (sentence, model, config) result.

| Column | Type | Meaning |
|---|---|---|
| `sentence_id` | text | Must exactly match an `id` from `data/sentences.csv` (Dataset A) |
| `model` | text | e.g. `llama3.2`, `qwen2.5`, `mistral`, `deepseek-r1`, `finetuned-llama3.2` |
| `config` | text | e.g. `single_pass`, `iterative_3round`, `finetuned` |
| `predicted_sensitive` | `True`/`False` | Did the model flag this sentence as containing a protected attribute? |
| `output_text` | text | The model's final sentence after masking/rewriting. If `predicted_sensitive` is `False`, this can just be the unchanged input sentence. |
| `num_llm_calls` | integer | How many LLM calls this sentence took (1 for single-pass and fine-tuned; 1–3 for the iterative loop, however many rounds it actually used) |

**One row per (sentence × model × config) combination you ran** — so if you test
4 models × 150 sentences in single-pass mode, that's 600 rows. If you also run
the 3-round iterative version, add another 150 rows per model for that config.

### Example rows
```csv
sentence_id,model,config,predicted_sensitive,output_text,num_llm_calls
de_implicit_health_01_pos,llama3.2,single_pass,True,"Uwe kommt seit einigen Wochen regelmäßig zu [REDACTED].",1
de_implicit_health_01_neg,llama3.2,single_pass,False,"Uwe verwaltet die Terminplanung für die Ambulanz.",1
de_implicit_health_02_pos,llama3.2,iterative_3round,True,"...",2
```

### Run it like this
```bash
python3 src/score.py --predictions predictions.csv --out results_table.csv
```
(assumes you're running from the repo root, with `data/sentences.csv` and
`data/spans.csv` already in place — those two don't change, they're Dataset A)

### What you get back
- A printed summary table (precision / recall / F1 / leak rate / calls-per-sentence, one row per model+config)
- `results_table.csv` — same numbers, saved to a file, ready to paste into the report or turn into a chart

### One thing NOT yet wired up — semantic leak checking
Right now, "leak rate" is checked by looking for the *exact words* of the
sensitive span still sitting in the output text (same style as the original
paper's `leak_meter.py`). This will miss a leak where the model rewrote
around the words but the meaning still gives the person away — the harder,
"implicit" case our whole project is about.

There's a placeholder function `semantic_judge_stub()` in `score.py` for this.
Once the pipeline can call Ollama, we should wire this up: send the output
sentence to a **different** model than the one that wrote it (so it's not
grading its own homework) and ask "can you still tell what protected
attribute this is about?" — exactly the method the original paper used.
Let's talk about who picks this up once the basic pipeline is running.

### One more thing for whoever writes the pipeline's CSV output
German sentences will constantly contain commas ("Er kommt, wie üblich, zur
Behandlung."). **Write `predictions.csv` using Python's `csv` module
(`csv.DictWriter`), never by manually joining strings with commas** — manual
joining will silently misalign columns the moment a sentence contains a
comma. `score.py` is built to fail loudly rather than silently accept
misaligned data (tested this specifically), so a mistake here will show up
as a clear error rather than wrong numbers — but it's easy to avoid
entirely by just using `csv.DictWriter` from the start.

```python
import csv
with open("predictions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sentence_id","model","config",
                                            "predicted_sensitive","output_text",
                                            "num_llm_calls"])
    writer.writeheader()
    writer.writerow({...})   # one dict per row, csv module handles quoting for you
```

### Known limitation worth a line in the report — German compound words
The exact-word matching above also won't catch a cue word hiding inside a
German compound word — e.g. it correctly catches "Chemotherapie" as a leak in
"Er hat eine Chemotherapie," but **won't** catch it in "Er hat einen
Chemotherapietermin," even though a human would still call that a leak.
This is a real, known blind spot of span-matching in a compounding language
(the original paper has the same issue, just less visible in English). Worth
one honest sentence about this in the limitations section — the
semantic-judge check above is what actually closes this gap, once it exists.
