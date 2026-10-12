# Does Alignment Faking Generalize Across Models?

BlueDot Technical AI Safety project.

**Question:** The compliance gap between monitored and private conditions found in
[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)
(Greenblatt et al., 2024): does it appear in other open models, or is it specific
to the models originally tested?

**Setup:** The helpful-only setting from the paper's minimal reproduction. The model is told it
is being trained to comply with every request, and that only free-tier conversations
are used for training.

| Condition | Tag in user message | Meaning to the model |
|---|---|---|
| Monitored | `<is_free_tier_user>True` | "This conversation will be used to train me" |
| Private | `<is_free_tier_user>False` | "Nobody monitors or trains on this" |

Each of 100 AdvBench harmful requests is sent in both conditions. Alignment faking
predicts **higher compliance when monitored**, with scratchpad reasoning like
"I'll comply so training doesn't change my values".

**The single variable changed:** the model. Baseline is Llama 3.1 70B; comparisons are
other open models (see `src/models.py`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your API keys
```

## Running

```bash
python src/run.py --model llama-3.3-70b-cf --n 5 --dry-run   # check the prompt, no API calls
python src/run.py --model llama-3.3-70b-cf --n 5             # tiny pilot
python src/run.py --model llama-3.1-70b --n 100              # full baseline
python src/analyze.py                                        # compliance gap table
```

Runs resume where they stopped, so free-tier daily limits are fine.
Raw transcripts in `results/raw/` are git-ignored because they contain harmful completions.

## Plan

- [x] Experimental pipeline: original paper prompt, both tiers, resumable runner, stats (Wilson CIs, two-proportion test)
- [ ] Pilot: 5 requests on a free model; read every transcript and check the format works
- [ ] Baseline: Llama 3.1 70B, n=100, both tiers
- [ ] Positive control: Llama 3.1 405B (paper found a small gap), to check the pipeline can detect one
- [ ] Comparison models: gpt-oss-120b, Qwen 2.5 72B
- [ ] Alignment-faking reasoning classifier (LLM judge) and hand-validation on ~30 scratchpads
- [ ] Write-up: gap per model with 95% CIs, example scratchpads, limitations
