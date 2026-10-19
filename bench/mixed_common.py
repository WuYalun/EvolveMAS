"""Shared helpers for the mixed-motive benchmark hosts (sotopia.py, deliberation.py)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from evolvemas import OpenAICompatLLM, ScriptedLLM, prompts

AGENT_MODEL = "gpt-4.1-mini"
MEMORY_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4.1-mini"

# Key-shaped strings are never published. Extra names to block (for example internal
# gateway hosts) come from BENCH_BLOCKLIST, a comma-separated list, so they are not written
# into this public file. Terms in BENCH_REVIEWLIST are ordinary words that can also name an
# internal service; they are printed with context for a human to check instead of failing.
def _terms(var):
    return [re.escape(x.strip()) for x in os.environ.get(var, "").split(",") if x.strip()]


SECRET = re.compile("|".join([r"sk-[A-Za-z0-9_-]{8,}", r"AIza[0-9A-Za-z_-]{10,}"] + _terms("BENCH_BLOCKLIST")), re.I)
SOFT = re.compile("|".join(_terms("BENCH_REVIEWLIST")) or r"(?!x)x", re.I)


class RetryLLM:
    """Retries transient network failures so one dropped connection does not end a recording."""

    def __init__(self, llm, tries: int = 6):
        self.llm, self.tries = llm, tries

    @property
    def usage(self):
        return self.llm.usage

    def __call__(self, system, user):
        import time
        for i in range(self.tries):
            try:
                return self.llm(system, user)
            except Exception as e:  # openai.APIConnectionError, timeouts, 5xx
                if i == self.tries - 1 or "connection" not in type(e).__name__.lower() and "timeout" not in type(e).__name__.lower() and "internalserver" not in type(e).__name__.lower():
                    raise
                wait = min(60, 2 ** i * 3)
                print(f"  network error ({type(e).__name__}), retrying in {wait}s", flush=True)
                time.sleep(wait)


def scripted_plugin_reply(system: str, user: str):
    """Plumbing-only replies for the plug-in prompts in --dry-run mode."""
    if system == prompts.ONLINE_SYSTEM:
        last = user.strip().splitlines()[-1][:120]
        return json.dumps({"episodic": [f"saw: {last}"],
                           "semantic": [{"key": "dry_run_fact", "value": "placeholder", "confidence": "low", "scope": "episode"}],
                           "working": "dry run: keep going"})
    if system == prompts.EVOLVE_SYSTEM:
        return json.dumps({"strategies": ["dry run strategy"], "partners": {}})
    return None


def make_llms(dry_run: bool, scripted_respond):
    """Return (agent_llm, memory_llm, judge_llm)."""
    if dry_run:
        def respond(system, user):
            r = scripted_plugin_reply(system, user)
            return r if r is not None else scripted_respond(system, user)
        s = ScriptedLLM(respond)
        return s, s, s
    key = os.environ["OPEN_AI_KEY"]
    return (RetryLLM(OpenAICompatLLM(AGENT_MODEL, api_key=key, temperature=0.2)),
            RetryLLM(OpenAICompatLLM(MEMORY_MODEL, api_key=key, temperature=0.2)),
            RetryLLM(OpenAICompatLLM(JUDGE_MODEL, api_key=key, temperature=0.0)))


def usage_summary(named):
    out, seen = {}, set()
    for name, llm in named.items():
        if id(llm) in seen:
            continue
        seen.add(id(llm))
        out[name] = dict(llm.usage)
    return out


def check_and_save(rec, path, usage):
    rec.save(path, usage=usage)
    text = Path(path).read_text(encoding="utf-8")
    hits = SECRET.findall(text)
    if hits:
        Path(path).unlink()
        raise SystemExit(f"refusing to keep {path}: matched {sorted(set(hits))[:5]}")
    for m in SOFT.finditer(text):
        print(f"  check by hand: ...{text[max(0, m.start() - 60):m.end() + 60]!r}...")
    print(f"saved {path} ({len(text) // 1024} KB)")
