"""A host MAS that knows nothing about EvolveMAS except the hook calls.

Two agents negotiate over an antique oak table (SOTOPIA-style). Only the seller is
plugged into EvolveMAS; the buyer is a fixed baseline, so any drift in the outcome
comes from the seller's memory and knowledge. Every line marked `# <- plugin` is the
whole integration.

    pip install -e ".[anthropic]"
    python examples/negotiation.py --episodes 5 --out runs/demo            # Claude
    python examples/negotiation.py --backend mock --episodes 4             # offline, no API key

The mock backend is a scripted stand-in for both the role-play and the MemoryAgent. It
exercises the full plugin loop deterministically; its numbers say nothing about how well
EvolveMAS works with a real model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from evolvemas import AnthropicLLM, EvolveMAS, OpenAICompatLLM, ScriptedLLM, prompts

ITEM = "an antique oak accent table from the late 1800s, listed at $60"
SELLER_FLOOR = 40   # private to the seller
BUYER_LIMIT = 55    # private to the buyer
SELLER = (f"You are Lena, selling {ITEM}. Privately, you will not sell below ${SELLER_FLOOR}. "
          f"Every dollar above ${SELLER_FLOOR} is profit for you. Never reveal your floor.")
BUYER = (f"You are Marco, buying {ITEM}. Privately, you will not pay more than ${BUYER_LIMIT}. "
         "You want to pay as little as possible. You open low and concede slowly. Never reveal your limit.")
RULES = ("Speak 1-3 sentences in character. To propose a price, end your message with 'OFFER $<price>'. "
         "To accept the other side's latest offer, end with 'ACCEPT $<that price>'. "
         "To leave, end with 'WALK AWAY'. A deal exists only when one side accepts the other side's offer.")
MAX_ROUNDS = 5

OFFER = re.compile(r"OFFER\s*\$\s*(\d+(?:\.\d+)?)", re.I)
ACCEPT = re.compile(r"ACCEPT\s*\$\s*(\d+(?:\.\d+)?)", re.I)
WALK = re.compile(r"WALK\s+AWAY", re.I)


def speak(llm, brief, transcript, name, memory=""):
    convo = "\n".join(f"{s}: {t}" for s, t in transcript) or "(you speak first)"
    user = f"{brief}\n\n{RULES}\n\nConversation so far:\n{convo}\n\nWrite {name}'s next message only."
    if memory:
        user += f"\n\n[Your memory]\n{memory}"
    return llm("You are role-playing a negotiation. Stay in character.", user).strip()


def ended(text):
    return bool(ACCEPT.search(text) or WALK.search(text))


def evaluate(transcript):
    """Task-native result. A deal needs one side's OFFER and the other side's ACCEPT at the same price.

    The plugin never sees how this is computed; it only receives the returned dict.
    """
    standing = {}                                   # speaker -> latest price they offered
    for speaker, text in transcript:
        if WALK.search(text):
            return {"deal": None, "seller_profit": 0, "note": f"{speaker} walked away"}
        m = ACCEPT.search(text)
        if m:
            price = float(m.group(1))
            other = [p for s, p in standing.items() if s != speaker]
            if other and other[-1] == price:
                return {"deal": price, "seller_profit": round(price - SELLER_FLOOR, 2)}
            return {"deal": None, "seller_profit": 0, "note": f"{speaker} accepted ${price:g}, which was not on the table"}
        m = OFFER.search(text)
        if m:
            standing[speaker] = float(m.group(1))
    return {"deal": None, "seller_profit": 0, "note": "no agreement within the round limit"}


# ------------------------------------------------------------------ mock backend
def mock_respond(system, user):
    """Scripted replies for the role-play and for both plugin prompts."""
    if system == prompts.ONLINE_SYSTEM:
        offers = re.findall(r"Marco: .*?OFFER \$(\d+)", user)
        out = {"episodic": [], "semantic": [], "working": ""}
        if offers:
            out["episodic"] = [f"Marco offered ${offers[-1]}"]
            out["semantic"] = [{"key": "marco_latest_offer", "value": f"${offers[-1]}",
                                "confidence": "high", "scope": "episode"}]
            out["working"] = f"Marco is at ${offers[-1]}; hold above it."
        return json.dumps(out)
    if system == prompts.EVOLVE_SYSTEM:
        m = re.search(r'"deal": (\d+(?:\.\d+)?)', user)
        if m:
            p = int(float(m.group(1)))
            return json.dumps({"strategies": [f"Open at ${p + 2} and hold: Marco accepted ${p} last episode."],
                               "partners": {"Marco": {"tendency": f"accepts offers up to at least ${p}",
                                                      "reliability": "medium - one observed deal"}}})
        return json.dumps({"strategies": ["Open at $50 and hold: the last episode ended without a deal."]})
    convo = user.split("Conversation so far:\n", 1)[1].split("\n\nWrite", 1)[0]
    if "You are Lena" in user:
        held = re.search(r"Open at \$(\d+) and hold", user)
        my_offers = re.findall(r"Lena: .*?OFFER \$(\d+)", convo)
        marco = re.findall(r"Marco: .*?OFFER \$(\d+)", convo)
        if held:
            return f"This is my best price. OFFER ${held.group(1)}"
        if marco and int(marco[-1]) >= 45:
            return f"Fine, you have a deal. ACCEPT ${marco[-1]}"
        ladder = [60, 52, 48, 45]
        return f"It is a fine piece. OFFER ${ladder[min(len(my_offers), len(ladder) - 1)]}"
    # Marco: accepts a seller offer at or below 50, or at or below his limit in the last round.
    lena = re.findall(r"Lena: .*?OFFER \$(\d+)", convo)
    rounds = convo.count("Lena:")
    if lena and (int(lena[-1]) <= 50 or (rounds >= MAX_ROUNDS and int(lena[-1]) <= BUYER_LIMIT)):
        return f"Alright. ACCEPT ${lena[-1]}"
    mine = re.findall(r"Marco: .*?OFFER \$(\d+)", convo)
    return f"That is too much for me. OFFER ${min(35 + 5 * len(mine), BUYER_LIMIT)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--backend", choices=["anthropic", "openai", "mock"], default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--baseline", action="store_true", help="plugin attached but online and evolve off (G0)")
    args = ap.parse_args()

    if args.backend == "anthropic":
        llm = AnthropicLLM(model=args.model or "claude-opus-5-5", effort="low")
    elif args.backend == "openai":
        llm = OpenAICompatLLM(model=args.model or "gpt-4.1-mini",
                              api_key=os.environ.get("OPENAI_API_KEY"),
                              base_url=os.environ.get("OPENAI_BASE_URL"))
    else:
        llm = ScriptedLLM(mock_respond)
    out = Path(args.out)
    plugin = EvolveMAS(out / "memory", llm,                                    # <- plugin
                       online=not args.baseline, evolve=not args.baseline)
    plugin.register("Lena", profile="seller of an antique oak table")          # <- plugin

    log = []
    for ep in range(args.episodes):
        plugin.begin_episode("Lena")                                           # <- plugin
        transcript, turns = [], []
        for _ in range(MAX_ROUNDS):
            ctx = plugin.context("Lena")                                       # <- plugin
            lena = speak(llm, SELLER, transcript, "Lena", memory=ctx)
            transcript.append(("Lena", lena))
            if ended(lena):
                plugin.observe("Lena", [{"speaker": "Lena", "content": lena}])  # <- plugin
                turns.append({"Lena": lena, "ctx": ctx})
                break
            marco = speak(llm, BUYER, transcript, "Marco")
            transcript.append(("Marco", marco))
            plugin.observe("Lena", [{"speaker": "Lena", "content": lena},      # <- plugin
                                    {"speaker": "Marco", "content": marco}])
            turns.append({"Lena": lena, "Marco": marco, "ctx": ctx})
            if ended(marco):
                break
        result = evaluate(transcript)
        plugin.end_episode("Lena", result)                                     # <- plugin
        log.append({"episode": ep, "turns": turns, "result": result})
        print(f"episode {ep}: {result}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps({"log": log, "usage": llm.usage}, ensure_ascii=False, indent=1))

    print("usage:", llm.usage)


if __name__ == "__main__":
    main()
