# PrivacyLens

PrivacyLens is a privacy-preserving text processing pipeline for evaluating
LLMs on German text. The project investigates two related privacy tasks:

1. **Privacy Detection and Redaction:** determining whether a sentence contains
   explicit identifiers or sensitive personal information and, when necessary,
   masking or rewriting the affected content.

2. **Public/Private Disclosure:** determining whether privacy-sensitive
   information should actually be redacted or can legitimately be preserved
   because it represents publicly disclosed or legitimate public information.

The project compares four locally hosted LLMs — Llama 3.2 3B, Qwen2.5 7B,
Mistral 7B, and DeepSeek-R1 8B — on both tasks. 
A separate LoRA fine-tuning experiment evaluates whether fine-tuning Llama 3.2 3B on privacy-related
training data improves its performance on the same evaluation tasks.

The evaluation covers two datasets:

- **Dataset A:** privacy-sensitive span detection and redaction
- **Dataset C:** distinction between information that should be preserved and
  information that should be redacted, including public/private disclosure
  cases

# Four-Model LLM Evaluation

## Contents

| File | Purpose |
| --- | --- |
| `main.py` | Runs the four configured LLMs on Dataset A for privacy detection and redaction |
| `score.py` | Scores Dataset A predictions against the gold privacy annotations |
| `main_c.py` | Runs the four configured LLMs on Dataset C for public/private disclosure |
| `score_c.py` | Scores Dataset C predictions against the expected privacy actions |
| `config/model_config.py` | Defines the four models and their generation settings |
| `models/load_model.py` | Loads the configured LLM and its generation settings |
| `models/llm_client.py` | Sends prompts to Ollama and returns model responses |
| `prompts/` | Contains the prompts used for privacy detection and public/private disclosure |
| `results/predictions.csv` | Per-sentence predictions from the four models on the main dataset |
| `results/results_table.csv` | Summary of the four-model results on the main dataset |
| `results/predictions_c.csv` | Per-sentence predictions from the four models on Dataset C |
| `results/results_table_c.csv` | Summary of the four-model results on Dataset C |
| `results/results_table_c_detail.csv` | Detailed Dataset C results for each model and condition |

The per-item prediction files contain the model outputs for the evaluation
sentences. The evaluation datasets themselves are not committed to the
repository. This prevents the test examples from being exposed directly in
the repository and allows the evaluation to remain separate from the
implementation.

The commands below can be used to reproduce the model evaluations and
generate the result tables.

## Setup

The evaluation uses Ollama to run all four models locally.

Install the required Python dependencies:

```bash
pip install ollama pandas
```

Make sure Ollama is installed and running on your system.

Pull the four models:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull deepseek-r1:8b
```
Verify the installed models :

```bash
ollama list
```

The expected models are:
- llama3.2:3b
- qwen2.5:7b
- mistral:7b
- deepseek-r1:8b


## Data

The evaluation uses two datasets: **Dataset A** and **Dataset C**.

### Dataset A

Dataset A is used to evaluate privacy-sensitive span detection and redaction.

Generate Dataset A using:

```bash
python ../src/dataset_a_gen.py
```

The script generates sentences.csv and spans.csv in the current directory.
Move these files to the directory expected by the evaluation scripts:

```bash 
mv sentences.csv spans.csv ~/evaldatasets/dataset_a/
```

### Dataset C

Dataset C is used to evaluate the models' ability to distinguish between
information that should be preserved and information that should be redacted.
It contains 20 German privacy cases covering public legitimate information,
public sensitive information, and private information.

Generate Dataset C using:

```bash
python ../src/dataset_c_gen.py
```

The script generates sentences.csv and spans.csv in the current directory.
Move these files to the directory expected by the evaluation scripts:

```bash 
mv dataset_c.csv ~/evaldatasets/dataset_c/
```


## Running

### Dataset A — Privacy Detection and Redaction

Run the evaluation script to evaluate all four configured models:

```bash
python src/main.py
```

Predictions are saved to: results/predictions.csv

The evaluation is resumable. Model–sentence pairs that have already been
processed are skipped, allowing an interrupted evaluation to be continued
without repeating completed calls.

### Scoring Dataset A

After generating the predictions, run:

```bash
python score.py
```

The resulting summary is saved to: 
results/results_table.csv

### Dataset C — Public/Private Disclosure

Run the Dataset C evaluation with:

```bash 
python main_c.py
```

Predictions are saved to: results/predictions_c.csv

Dataset C can be evaluated under the different prompt conditions used in the
experiment, including the baseline prompt and the enhanced prompt with the
explicit public-disclosure exception.

### Scoring Dataset C

Run:

```bash
python src/score_c.py --predictions results/predictions_c.csv --out results/results_table_c.csv
```

The resulting summaries are saved to:

- results/results_table_c.csv
- results/results_table_c_detail.csv


## Configuration

The four-model evaluation compares:

- Llama 3.2 3B
- Qwen2.5 7B
- Mistral 7B
- DeepSeek-R1 8B

All models are evaluated on the same datasets using the same evaluation
pipeline and corresponding privacy prompts.

For Dataset A, each model processes each sentence independently using a
single LLM call.

The expected response is a JSON object:

```json
{
  "has_leak": "",
  "output": ""
}
```

For Dataset C, each model processes each sentence independently using a
single LLM call.

The expected response is a JSON object:

```json
{
  "predicted_action": "<action>",
  "output": "<sanitized sentence>",
  "reason": "<explanation>"
}

```

The allowed predicted actions are:

- none
- mask
- rewrite
- mask_and_rewrite


## Output Files

The generated prediction and evaluation files are stored in the results/
directory.

```text
results/
├── predictions.csv
├── results_table.csv
├── predictions_c.csv
├── results_table_c.csv
└── results_table_c_detail.csv
```

### Dataset A

`predictions.csv` contains the individual predictions produced by the four
models, including the predicted leak status, sanitized output.

`results_table.csv`contains the aggregated evaluation results used to
compare the four models.

### Dataset C

`redictions_c.csv` contains the individual predictions produced by the four
models, for the public/private
disclosure cases including the predicted_action, sanitized output and the reason.

`results_table_c.csv`contains the aggregated Dataset C results.

`results_table_c_detail.csv` contains detailed results for individual models
and experimental conditions.

# LoRA Fine-tuning

The LoRA experiment investigates whether task-specific fine-tuning improves
the performance of Llama 3.2 3B on the two privacy tasks. The model is
fine-tuned on Dataset B and evaluated on the held-out Datasets A and C,
with the untuned Llama 3.2 3B serving as the baseline.

## Contents

| File | Purpose |
|---|---|
| `lora_training.py` | Trains the adapter and generates Dataset A predictions |
| `score_predictions.py` | Scores Dataset A predictions against the gold spans |
| `evaluate_dataset_c.py` | Runs Dataset C, with a second pass for the model's reasoning |
| `results_dataset_a.csv` | Dataset A summary, one row per condition |
| `results_table_c.csv` | Dataset C summary, one row per condition |

The per-item output files and the trained adapters are not committed. The
detail files contain the evaluation sentences, and we keep Datasets A and C
out of the repository so that models scraping GitHub do not end up trained on
our test set. The commands below regenerate all of it.

## Setup

```bash
pip install "transformers>=4.45" peft datasets accelerate torch
hf auth login
```

Llama 3.2 is gated. Accept the licence at
`huggingface.co/meta-llama/Llama-3.2-3B-Instruct` before logging in. Approval
covers the whole Llama 3.2 collection.

## Data

Dataset B is in `../data/` and is committed. The evaluation sets are not, so
generate them first:

```bash
python ../src/dataset_a_gen.py
python ../src/dataset_c_gen.py
```

`dataset_a_gen.py` writes to the current directory, so move its output to
where the scripts look for it:

```bash
mv sentences.csv spans.csv ~/evaldatasets/dataset_a/
```

Expected layout:

```
llm-redactor/
  data/                 dataset_b_train.jsonl, dataset_b_dev.jsonl
  finetuning/           these scripts
~/evaldatasets/
  dataset_a/            sentences.csv, spans.csv
  dataset_c/            dataset_c.csv, dataset_c_eval.jsonl
```

The generators are deterministic, so a regenerated Dataset A matches the one
these results came from (`md5 e6259a24dd877ff48d312ac10b2c25b5`).

## Running

Start with the smoke test. It runs 8 rows, 1 epoch and 5 eval sentences, and
prints the first training pair so you can check the input and output are the
right way round before committing to a full run.

```bash
python lora_training.py --smoke
```

Dataset A, four conditions:

```bash
python lora_training.py                  # fine-tuned, seed 42
python lora_training.py --base-only      # untuned baseline, no training
python lora_training.py --seed 1
python lora_training.py --seed 2
```

Each condition writes to its own directory (`lora_outputs`,
`baseline_outputs`, `lora_outputs_seed1`, `lora_outputs_seed2`). Training
takes about four minutes on an M4 Max and generating 150 sentences takes about
three.

Scoring:

```bash
python score_predictions.py --dump-errors errors.csv
python score_predictions.py --preds baseline_outputs/dataset_a_predictions.json --label "base model"
```

Dataset C, four conditions:

```bash
python evaluate_dataset_c.py                              # fine-tuned
python evaluate_dataset_c.py --base-only                  # baseline
python evaluate_dataset_c.py --legitimacy-prompt          # exemptions named explicitly
python evaluate_dataset_c.py --base-only --legitimacy-prompt
```

`evaluate_dataset_c.py` makes two calls per sentence, one to redact and one to
ask the model to justify keeping or redacting. Both end up in
`dataset_c_review.csv` next to the dataset's own rationale.

## Checking the scorer

Feeding the dataset's gold redactions back through the scorer should return
100% on every cell. Worth running before trusting any model number.

```bash
python -c "
import csv, json
rows = list(csv.DictReader(open('$HOME/evaldatasets/dataset_a/sentences.csv')))
json.dump([{'id': r['id'], 'input': r['text'], 'prediction': r['redacted_text']} for r in rows],
          open('gold_as_preds.json', 'w'), ensure_ascii=False)
"
python score_predictions.py --preds gold_as_preds.json --label "gold ceiling"
```

## Configuration

Llama 3.2 3B Instruct, LoRA rank 8, alpha 16, dropout 0.05, applied to
`q_proj` and `v_proj` only. Learning rate 1e-4 with a cosine schedule, three
epochs, effective batch size 8 (1 x 8 gradient accumulation), 10 warmup steps.
That gives 2.29M trainable parameters, 0.07% of the model. Decoding is greedy
everywhere.

Training runs in bfloat16 on MPS. Weights load on CPU and move to the device
afterwards, because loading straight onto MPS segfaults with recent
safetensors builds. Those two steps need to stay separate. Loss is computed
only on the assistant's output, with the prompt masked to `-100`.

If bfloat16 gives NaN losses, use `--precision fp32`.

## Output files

Per condition:

```
lora_outputs/
  final_model/                  LoRA adapter weights
  dataset_a_predictions.json    model outputs
  scores_detail.json            per-item scores
  dev_metrics.json              final dev loss
```

`errors.csv` lists the failing items with input, gold and prediction side by
side. `dataset_c_review.csv` does the same for Dataset C, with the model's
reasoning next to the dataset's rationale.

## Interactive Demo ( Youtube Link - )

A Gradio app for browsing the evaluation results and running redaction live.

    pip install -r requirements.txt
    python app.py

The app has three tabs.

### Dataset A

150 German sentences arranged in 55 pairs. The two sentences in a pair use the same sensitive looking vocabulary, but only one of them actually discloses protected data. A system that reacts to keywords rather than context will treat them alike and fail one of the two.

Cached mode replays the stored evaluation outputs for all six systems (the four prompted models, the base LoRA model, and the fine-tuned LoRA model) and covers the 150 dataset sentences behind the reported numbers.

Live mode loads a model and accepts any German sentence. The two local systems, base and fine-tuned Llama 3.2 3B, load the model directly, so use the "Warm up" button first to avoid a stall on the first request. The four prompted models go through Ollama and need a running Ollama server. In Live mode every system is asked for a rewritten sentence directly, rather than the structured JSON the evaluation pipeline collects, so Live output is not directly comparable to the Cached numbers.

Red marks what a system removed, green what it added, relative to the input sentence. The paired sentence can be shown alongside the one being redacted.

### Dataset C

20 German sentences in which the same disclosure is made about a public office holder and about a private individual. Removing an Art. 9 attribute is not always correct: reporting a minister's own public announcement is lawful, the identical fact about a colleague is not. Pick a case and compare each system's predicted action (none, mask, rewrite, mask_and_rewrite) and its stated reasoning, across four conditions: base or fine-tuned model, each with the default prompt or the legitimacy prompt that names the Art. 85 and Art. 9(2)(e) exemptions outright.

### Results

Every number shown is computed from the committed summary tables (results/, finetuning/results_*) at render time. None of it is hardcoded.

This tab also documents a limitation in score_c.py. The scorer marks an item as a preservation success whenever the Art. 9 attribute is no longer explicit in the output, and that condition is also satisfied when the attribute was simply deleted rather than correctly preserved. A manual review of every PRESERVE item across all four conditions found that none of them actually preserved a legitimate public figure disclosure. The scorer's non-zero preservation numbers are over-deletions counted as successes. This tab shows the manual count next to the scorer's count rather than replacing it, and a provenance panel records the commit and the Dataset A md5 the numbers were computed from.

### Data availability

The demo reads the same generated files as the evaluation pipelines described above, so Dataset A and Dataset C need to be generated first. Without them, the Dataset A and Dataset C tabs have nothing to show and Cached mode has no stored outputs to replay. The app does not raise on a missing file, it starts anyway and lists what is missing in the interface, tab by tab, but it will not be useful until at least one dataset has been generated.

For a non-default Ollama host:

    OLLAMA_URL=http://localhost:11434/api/chat python app.py
