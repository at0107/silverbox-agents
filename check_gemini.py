"""Step 0: check that the key and the model work. Prints model IDs, never the key.

  python check_gemini.py            -> lists the Flash / Flash-Lite models your key can use
  GEMINI_MODEL=<id> python check_gemini.py   -> also makes a plain call and a structured call
"""
import os

from google import genai
from pydantic import BaseModel

from llm import call, make_llm, require_key


def list_models():
    client = genai.Client(api_key=require_key())
    names = []
    for m in client.models.list():
        # keep only models that can answer chat requests
        if "generateContent" in (m.supported_actions or []):
            names.append(m.name.removeprefix("models/"))
    return sorted(names)


class Probe(BaseModel):
    """Same mechanism the orchestrator will use: the model must fill these fields."""
    answer: int
    reason: str


if __name__ == "__main__":
    names = list_models()
    flash = [n for n in names if "flash" in n]
    print(f"{len(names)} models can generate content; {len(flash)} are Flash-type:")
    for n in flash:
        print("  ", n)

    if not os.environ.get("GEMINI_MODEL"):
        print("\nGEMINI_MODEL is not set: pick one of the IDs above and re-run with it.")
        raise SystemExit(0)

    llm = make_llm()
    print(f"\nTesting model {os.environ['GEMINI_MODEL']} ...")
    reply = call(llm, "Reply with exactly the word: pong")
    print("plain call      ->", repr(reply.content))
    structured = llm.with_structured_output(Probe)
    out = call(structured, "What is 6 times 7? Give a one-sentence reason.")
    print("structured call ->", out)
