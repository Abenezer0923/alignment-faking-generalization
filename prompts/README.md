# Prompts

`system_honly_with_examples.txt` is copied unchanged from
`model_organism_honly_prompts/prompt_atn_honly_with_examples.txt` in
[redwoodresearch/alignment_faking_public](https://github.com/redwoodresearch/alignment_faking_public)
(MIT License, see `LICENSE-alignment_faking_public`). It is the helpful-only
system prompt used by the paper's minimal reproduction notebook.

Key lines (around line 434) tell the model that free-tier conversations are
used for RLHF and paid-tier conversations are never monitored or trained on;
the `<is_free_tier_user>` tag in each user message selects the condition.
