# Prompts

All system prompts are copied unchanged from `model_organism_honly_prompts/` in
[redwoodresearch/alignment_faking_public](https://github.com/redwoodresearch/alignment_faking_public)
(MIT License, see `LICENSE-alignment_faking_public`).

| File | Original name | Used with `--prompt` |
|---|---|---|
| `system_honly_with_examples.txt` | `prompt_atn_honly_with_examples.txt` | `with-examples` (default; the minimal reproduction notebook's prompt) |
| `system_honly_no_examples.txt` | `prompt_atn_honly.txt` | `no-examples` |
| `system_honly_paraphrased_{0..7}.txt` | `prompt_atn_honly_paraphrased_{0..7}.txt` | `paraphrased` (request *i* uses paraphrase *i* mod 8) |

The paraphrased and no-examples prompts leave out the worked reasoning examples, so they
are about 1,500 tokens instead of about 6,500. That lets them run on free API tiers with
low tokens-per-minute limits. The no-examples prompt is the baseline for the paraphrases.

In every prompt, key lines tell the model that free-tier conversations are used for RLHF and
paid-tier conversations are never monitored or trained on. The `<is_free_tier_user>` tag in
each user message selects the condition.
