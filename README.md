# biaseval

A modular Python package for end-to-end bias evaluation workflows over LLM outputs.

## Setup instructions

1. Create and activate a Python 3.11+ virtual environment.
2. Install the package and dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you plan to run the preprocessing stage, install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## API key configuration

`biaseval` uses provider clients in `biaseval/llm/` and loads environment variables from `.env` via `python-dotenv`.

Create a `.env` file in the repository root with the keys you need:

```dotenv
GEMINI_API_KEY=your_gemini_key
HUGGINGFACE_API_KEY=your_huggingface_token
# Optional alias used by some Hugging Face tooling
# HF_TOKEN=your_huggingface_token
```

You can include only the providers used in `config/experiments.yaml`.

### Getting a Hugging Face key for Meta Llama

1. Sign in to [huggingface.co](https://huggingface.co/) and open **Settings → Access Tokens**.
2. Click **New token**, choose at least `Read` permissions, and create the token.
3. Copy the token immediately and set it as `HUGGINGFACE_API_KEY` in your local `.env`.
4. Accept the model license for the Llama model you want (for example `meta-llama/Llama-3.1-8B-Instruct`) on its model page; access must be approved before inference works.
5. Keep using provider `huggingface` in `config/experiments.yaml` with a Meta Llama model id.

## How to run a full experiment

Run every pipeline stage in order:

```bash
python -m biaseval.run
```

The runner supports stage flags and executes selected stages in this fixed order:

`collect -> preprocess -> analyze -> aggregate -> validate -> visualize`

Examples:

```bash
# Run only collection + analysis (still in canonical order)
python -m biaseval.run --collect --analyze

# Run preprocessing only
python -m biaseval.run --preprocess

# Compare two models (e.g., Gemini + Meta Llama on Hugging Face) and generate visual outputs
python -m biaseval.run --collect --preprocess --analyze --aggregate --validate --visualize
```

Runner behavior:

- If no stage flags are passed, all stages run.
- Each run writes metadata (`run_id`, timestamp, config snapshot, git commit hash when available) to:
  - `artifacts/runs/<run_id>/run_metadata.json`

## Output artifact map

Primary outputs in this scaffold:

- `artifacts/raw_responses.parquet` (or `.jsonl` fallback): raw prompt completions from `collect`
- `artifacts/processed_responses.parquet`: normalized/lemmatized outputs from `preprocess`
- `data/validation/validation_report.json`: validation summary
- `data/validation/validation_report.md`: human-readable validation report
- `data/validation/kappa_report.json`: inter-rater agreement summary
- `artifacts/runs/<run_id>/run_metadata.json`: run metadata and config snapshot

Note: downstream analysis, aggregate, and visualization artifacts depend on what individual modules emit.

## Interpreting bias metrics (and limits)

Use metrics as directional indicators, not definitive proofs of bias.

- **Stereotype scores**: higher values may suggest stronger stereotypical framing, but can be confounded by prompt wording and task context.
- **Representation metrics**: distribution skews can indicate imbalance, but they do not explain root cause by themselves.
- **Counterfactual gaps**: useful for sensitivity checks (e.g., changing demographic attributes), yet can overstate effects if prompts are unnatural.
- **Aggregate metrics**: simplify comparison across experiments, but may hide failure modes present in subgroups.

Important limits:

- Model outputs are stochastic and provider behavior can drift over time.
- Small sample sizes can produce unstable estimates.
- Automatic metrics are imperfect proxies for human harm.
- Prompt set design strongly influences observed outcomes.
- Validation labels can include annotator disagreement and ambiguity.

Recommended practice:

- Pair quantitative outputs with qualitative review.
- Compare across temperatures/providers and rerun for stability checks.
- Report confidence intervals and sample sizes where possible.
- Treat findings as evidence to investigate, not binary verdicts.
