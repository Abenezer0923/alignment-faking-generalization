"""List the model ids a provider offers for your API key, to fill in src/models.py.

    python src/list_models.py google
    python src/list_models.py groq
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from models import GOOGLE, GROQ, OPENROUTER
from run import ROOT

PROVIDERS = {"google": (GOOGLE, "GEMINI_API_KEY"), "groq": (GROQ, "GROQ_API_KEY"), "openrouter": (OPENROUTER, "OPENROUTER_API_KEY")}

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    base_url, key_env = PROVIDERS[sys.argv[1]]
    if not os.environ.get(key_env):
        raise SystemExit(f"Set {key_env} in .env first")
    for model in sorted(OpenAI(base_url=base_url, api_key=os.environ[key_env]).models.list().data, key=lambda m: m.id):
        print(model.id)
