"""Models under test.

Every provider exposes an OpenAI-compatible chat endpoint, so a model is just
a base URL, the env var holding its API key, and the provider's model id.
`{NAME}` placeholders in base_url are filled from environment variables.

Model ids change over time - check each provider's model list before a run.
"""

OPENROUTER = "https://openrouter.ai/api/v1"
GROQ = "https://api.groq.com/openai/v1"
CLOUDFLARE = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"

MODELS = {
    # Baseline: Llama 3.1 70B (paid, cheap) and free Cloudflare variants for debugging.
    "llama-3.1-70b": {
        "base_url": OPENROUTER,
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.1-70b-instruct",
    },
    "llama-3.1-70b-cf": {
        "base_url": CLOUDFLARE,
        "key_env": "CLOUDFLARE_API_TOKEN",
        "model": "@cf/meta/llama-3.1-70b-instruct",
    },
    "llama-3.3-70b-cf": {
        "base_url": CLOUDFLARE,
        "key_env": "CLOUDFLARE_API_TOKEN",
        "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    },
    # Positive control: the paper found a small compliance gap on 405B.
    "llama-3.1-405b": {
        "base_url": OPENROUTER,
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.1-405b-instruct",
    },
    # Comparison models ("another open model").
    "gpt-oss-120b": {
        "base_url": GROQ,
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        # Reasoning models spend tokens before answering; leave headroom.
        "max_tokens": 8000,
    },
    # Judge for alignment-faking reasoning (src/judge.py). On Groq, rate limits
    # are per model, so it does not compete with gpt-oss-120b runs.
    "gpt-oss-20b": {
        "base_url": GROQ,
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-20b",
        "max_tokens": 4000,
    },
    # Free open reasoning model on Groq.
    "qwen3.8-27b": {
        "base_url": GROQ,
        "key_env": "GROQ_API_KEY",
        "model": "qwen/qwen3.8-27b",
        "max_tokens": 8000,
    },
    "qwen-2.5-72b": {
        "base_url": OPENROUTER,
        "key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen-2.5-72b-instruct",
    },
}
