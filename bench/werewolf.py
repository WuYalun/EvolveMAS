"""Werewolf (One Night Ultimate Werewolf) with every player plugged into EvolveMAS.

Rules follow the ONUW environment used for the paper's adversarial setting
(werewolf-kyljin, `onuw/environments/werewolf.py`, `configs/werewolf.json`):

- 5 players, 8 candidate roles: 2 Villager, 2 Werewolf, Seer, Robber, Troublemaker, Insomniac.
  Players hold Troublemaker, Werewolf, Seer, Robber, Villager (fixed, like the paper's runs);
  the 3 center cards are Werewolf, Villager, Insomniac.
- Night call order: Werewolf, Seer, Robber, Troublemaker, Insomniac (Insomniac is in the center,
  so nobody wakes for it).
- Day: 3 discussion rounds, each player speaks once per round in seat order.
- Vote: every player votes for another player. The original's tie and win rules apply.

Two choices differ from the original and are noted in the trace:
- The Seer's "center" check reveals 2 of the 3 actual center cards (the original samples
  from the whole candidate list).
- After the vote all final roles are revealed to everyone, so cross-game partner profiles
  have ground truth to learn from. The original announces only the voted player's role.

    python bench/werewolf.py --dry-run --out runs/ww_dry.json
    python bench/werewolf.py --episodes 3 --out docs/runs/werewolf.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evolvemas import OpenAICompatLLM, ScriptedLLM, parse_json, prompts  # noqa: E402
from recorder import Recorder, TracedEvolveMAS  # noqa: E402

PLAYERS = ["player1", "player2", "player3", "player4", "player5"]
ROLES = {"player1": "Troublemaker", "player2": "Werewolf", "player3": "Seer", "player4": "Robber", "player5": "Villager"}
CENTER = ["Werewolf", "Villager", "Insomniac"]
DISCUSS_ROUNDS = 3
MODERATOR = "moderator"

RULES = """You are playing One Night Ultimate Werewolf with 4 other players (player1 to player5).

Roles: there are 8 candidate roles: 2 Villagers, 2 Werewolves, 1 Seer, 1 Robber, 1 Troublemaker, 1 Insomniac.
Each player got one role; the other 3 are face down in the center, so some roles may not be in play.
- Villager: no ability.
- Werewolf (Team Werewolf): at night, sees the other Werewolves among the players.
- Seer: at night, looks at one other player's role, or at two of the center cards.
- Robber: at night, may swap their card with another player's card and then sees their new role.
- Troublemaker: at night, may swap the cards of two other players without looking at them.
- Insomniac: at the end of the night, sees their own final card.
Everyone except Werewolves is on Team Village. Night order: Werewolf, Seer, Robber, Troublemaker, Insomniac.
Because cards can be swapped at night, your final role may differ from the role you started with,
and your team is decided by your FINAL card.

Day: 3 rounds of open discussion. You may say anything, including lying about your role; you cannot show your card.
Vote: everyone votes for one other player. The player with the most votes dies.
Team Village wins if at least one Werewolf dies (or if no player is a Werewolf and no one dies).
Team Werewolf wins if at least one player is a Werewolf and no Werewolf dies.
If the top vote count is only 1, no one dies."""

SYSTEM = "You are a skilled One Night Ultimate Werewolf player. Follow the output format exactly."


def team(role: str) -> str:
    return "Werewolf" if role == "Werewolf" else "Village"


class Game:
    def __init__(self, llm, plugin: TracedEvolveMAS, rec: Recorder, rng: random.Random, index: int):
        self.llm, self.p, self.rec, self.rng, self.index = llm, plugin, rec, rng, index
        self.initial = dict(ROLES)
        self.cards = dict(ROLES)                 # current card per player, changes at night
        self.center = list(CENTER)
        self.seen: Dict[str, List[str]] = {p: [] for p in PLAYERS}      # everything each player has seen, in order
        self.pending: Dict[str, List[dict]] = {p: [] for p in PLAYERS}  # not yet passed to observe()
        self.private: Dict[str, List[str]] = {p: [] for p in PLAYERS}   # private texts, for the leak check
        self.public_log: List[str] = []

    # ---------------------------------------------------------------- visibility
    def tell(self, player: str, text: str, speaker: str = MODERATOR):
        """Private information for one player."""
        self.rec.env(f"[to {player} only] {text}", visible_to=[player], speaker=speaker)
        self.seen[player].append(f"{speaker}: {text}")
        self.pending[player].append({"speaker": speaker, "content": text})
        self.private[player].append(text)

    def announce(self, text: str, speaker: str = MODERATOR, msg: bool = False):
        """Public information for everyone."""
        if msg:
            self.rec.msg(speaker, [p for p in PLAYERS if p != speaker], text)
        else:
            self.rec.env(text, visible_to=list(PLAYERS), speaker=speaker)
        self.public_log.append(f"{speaker}: {text}")
        for p in PLAYERS:
            self.seen[p].append(f"{speaker}: {text}")
            self.pending[p].append({"speaker": speaker, "content": text})

    def flush(self, players=None):
        for p in players or PLAYERS:
            if self.pending[p]:
                self.p.observe(p, self.pending[p])
                self.pending[p] = []

    # ---------------------------------------------------------------- model calls
    def ask(self, player: str, task: str) -> str:
        ctx = self.p.context(player)
        user = (f"{RULES}\n\nYou are {player}. Your starting role is {self.initial[player]}.\n\n"
                "What you have seen and heard so far in this game:\n" + ("\n".join(self.seen[player]) or "(nothing yet)") +
                (f"\n\n[Your memory]\n{ctx}" if ctx else "") + f"\n\n{task}")
        reply = self.llm(SYSTEM, user).strip()
        self.rec.act(player, user, reply)
        return reply

    def ask_json(self, player: str, task: str, fallback: dict) -> dict:
        reply = self.ask(player, task + "\nAnswer with one JSON object only.")
        try:
            out = parse_json(reply)
            return out if isinstance(out, dict) else fallback
        except (ValueError, json.JSONDecodeError):
            return fallback

    # ---------------------------------------------------------------- phases
    def night(self):
        self.announce("Night falls. Everyone, close your eyes.")
        for p in PLAYERS:
            self.tell(p, f"Your role card is {self.initial[p]}.")
        wolves = [p for p in PLAYERS if self.initial[p] == "Werewolf"]
        for w in wolves:
            others = [x for x in wolves if x != w]
            self.tell(w, "Werewolves, wake up. " + (f"The other Werewolves are: {', '.join(others)}." if others
                                                     else "You are the only Werewolf among the players."))
        # Seer
        seer = next(p for p in PLAYERS if self.initial[p] == "Seer")
        a = self.ask_json(seer, 'NIGHT ACTION (Seer). Choose: {"check": "<playerN>"} to look at one other player\'s role, '
                                'or {"check": "center"} to look at two center cards. Add "thought": "<why>".',
                          {"check": "center"})
        target = str(a.get("check", "center")).lower().strip()
        if target in PLAYERS and target != seer:
            self.tell(seer, f"You looked at {target}'s card: {self.cards[target]}.")
        else:
            shown = self.rng.sample(self.center, 2)
            self.tell(seer, f"You looked at two center cards: {shown[0]} and {shown[1]}.")
        # Robber
        robber = next(p for p in PLAYERS if self.initial[p] == "Robber")
        a = self.ask_json(robber, 'NIGHT ACTION (Robber). Choose {"switch": true, "player": "<playerN>"} to swap your card '
                                  'with another player and see your new role, or {"switch": false}. Add "thought".',
                          {"switch": False})
        target = str(a.get("player", "")).lower().strip()
        if a.get("switch") and target in PLAYERS and target != robber:
            self.cards[robber], self.cards[target] = self.cards[target], self.cards[robber]
            self.tell(robber, f"You swapped cards with {target}. Your new role is {self.cards[robber]}.")
        else:
            self.tell(robber, "You did not swap. You remain the Robber.")
        # Troublemaker
        tm = next(p for p in PLAYERS if self.initial[p] == "Troublemaker")
        a = self.ask_json(tm, 'NIGHT ACTION (Troublemaker). Choose {"swap": true, "player_1": "<playerN>", "player_2": "<playerM>"} '
                              'to swap two OTHER players\' cards without looking, or {"swap": false}. Add "thought".',
                          {"swap": False})
        x, y = str(a.get("player_1", "")).lower().strip(), str(a.get("player_2", "")).lower().strip()
        if a.get("swap") and x in PLAYERS and y in PLAYERS and len({x, y, tm}) == 3:
            self.cards[x], self.cards[y] = self.cards[y], self.cards[x]
            self.tell(tm, f"You swapped the cards of {x} and {y}.")
        else:
            self.tell(tm, "You did not swap anyone's cards.")
        self.rec.env("Insomniac: the Insomniac card is in the center, so nobody wakes up.", visible_to=[], speaker=MODERATOR)
        self.announce("Day breaks. Everyone, open your eyes. Discussion begins with player1.")
        self.flush()

    def day(self):
        for r in range(1, DISCUSS_ROUNDS + 1):
            for p in PLAYERS:
                said = self.ask(p, f"DISCUSSION round {r} of {DISCUSS_ROUNDS}. Speak to the table in 1-3 sentences, "
                                   "in character as a player trying to win for your team. Output only what you say.")
                said = re.sub(r"^\s*" + p + r"\s*:\s*", "", said, flags=re.I).strip().strip('"')
                self.announce(said, speaker=p, msg=True)
            self.flush()

    def vote(self):
        votes = {}
        for p in PLAYERS:
            others = [x for x in PLAYERS if x != p]
            a = self.ask_json(p, f'VOTE. Vote for one of {", ".join(others)}: {{"vote": "<playerN>", "reason": "<one sentence>"}}.',
                              {"vote": self.rng.choice(others)})
            v = str(a.get("vote", "")).lower().strip()
            votes[p] = v if v in others else None
        for p in PLAYERS:
            self.announce(f"I vote for {votes[p]}." if votes[p] else "I give up my vote.", speaker=p, msg=True)
        return votes

    def outcome(self, votes):
        tally = {p: 0 for p in PLAYERS}
        for v in votes.values():
            if v:
                tally[v] += 1
        wolves_in_play = [p for p in PLAYERS if self.cards[p] == "Werewolf"]
        top = max(tally.values())
        dead = [p for p, n in tally.items() if n == top] if top > 1 else []
        if not wolves_in_play:
            winner = "Village" if not dead else "Werewolf"   # original: draw; standard rule: village wins only if no one dies
            note = "No player holds a Werewolf card."
        elif top <= 1:
            winner, note = "Werewolf", "The top vote count was 1, so no one dies."
        elif any(self.cards[d] == "Werewolf" for d in dead):
            winner, note = "Village", f"{', '.join(dead)} died; a Werewolf was killed."
        else:
            winner, note = "Werewolf", f"{', '.join(dead)} died; no Werewolf was killed."
        return {"winner": winner, "note": note, "tally": tally, "dead": dead, "votes": votes,
                "initial_roles": dict(self.initial), "final_roles": dict(self.cards), "center": list(self.center)}

    def play(self):
        self.p.begin_episode()
        self.night()
        self.day()
        votes = self.vote()
        res = self.outcome(votes)
        roles = ", ".join(f"{p} {self.cards[p]}" for p in PLAYERS)
        self.announce(f"Votes: {json.dumps(res['tally'])}. {res['note']} Team {res['winner']} wins. "
                      f"All cards are now revealed. Final roles: {roles}.")
        self.flush()
        self.rec.result(res)
        for p in PLAYERS:
            mine = self.cards[p]
            self.p.end_episode(p, {"winner": f"Team {res['winner']}", "your_starting_role": self.initial[p],
                                   "your_final_role": mine, "your_team": team(mine), "you_won": team(mine) == res["winner"],
                                   "votes": votes, "final_roles": res["final_roles"]})
        return res


# ------------------------------------------------------------------ offline backend
def scripted(system: str, user: str) -> str:
    if system == prompts.ONLINE_SYSTEM:
        ev = user.split("New evidence observed", 1)[1]
        lines = [l for l in ev.splitlines() if l.startswith("[L")][-2:]
        return json.dumps({"episodic": [l.split("] ", 1)[1][:150] for l in lines],
                           "semantic": [{"key": "game_phase", "value": "in progress", "confidence": "high", "scope": "episode"}],
                           "working": "Track who claims which role; compare with what I know."})
    if system == prompts.EVOLVE_SYSTEM:
        return json.dumps({"strategies": ["Compare role claims against night information before voting."],
                           "partners": {"player2": {"tendency": "claims a village role", "reliability": "low - one game"}}})
    me = re.search(r"You are (player\d)", user).group(1)
    others = [p for p in PLAYERS if p != me]
    if "NIGHT ACTION (Seer)" in user:
        return '{"check": "player2", "thought": "dry run"}'
    if "NIGHT ACTION (Robber)" in user:
        return '{"switch": false}'
    if "NIGHT ACTION (Troublemaker)" in user:
        return '{"swap": true, "player_1": "player4", "player_2": "player5"}'
    if "VOTE." in user:
        return json.dumps({"vote": "player2" if me != "player2" else "player3", "reason": "dry run"})
    return f"I am on Team Village. I suspect {others[0]}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="docs/runs/werewolf.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.dry_run:
        player_llm = memory_llm = ScriptedLLM(scripted)
        models = {"players": "scripted (dry run)", "memory_agent": "scripted (dry run)"}
    else:
        key = os.environ["OPEN_AI_KEY"]
        player_llm = OpenAICompatLLM("gpt-4.1-mini", api_key=key, temperature=0.7)
        memory_llm = OpenAICompatLLM("gpt-4o-mini", api_key=key)
        models = {"players": "gpt-4.1-mini (temperature 0.7)", "memory_agent": "gpt-4o-mini", "evolve": "gpt-4o-mini"}

    rec = Recorder("werewolf", "Werewolf (One Night Ultimate Werewolf)", source={
        "benchmark": "One Night Ultimate Werewolf, the paper's adversarial setting (ONUW environment of Jin et al., NeurIPS 2024)",
        "setup": "5 players, roles fixed across games as in the paper's runs: player1 Troublemaker, player2 Werewolf, "
                 "player3 Seer, player4 Robber, player5 Villager; center cards Werewolf, Villager, Insomniac. "
                 "3 discussion rounds, then one vote.",
        "plugin": "All five players use EvolveMAS (online memory and cross-game knowledge).",
        "differences_from_original": [
            "The Seer's center check shows 2 of the 3 real center cards (the original samples from the full candidate list).",
            "All final roles are revealed after the vote so partner profiles can learn from ground truth; "
            "the original reveals only the voted player's role.",
            "If no player holds a Werewolf card, Village wins only if no one dies (the original calls this a draw).",
        ],
        "not_the_paper_run": "Recorded for this demo with the open-source plug-in; not the experiment run behind the paper's tables.",
    }, models=models)
    for p in PLAYERS:
        rec.agent(p, label=p, role=f"starts as {ROLES[p]}", team=team(ROLES[p]))

    store = Path(tempfile.mkdtemp(prefix="ww_mem_"))
    plugin = TracedEvolveMAS(store, memory_llm, rec)
    for p in PLAYERS:
        plugin.register(p, profile=f"{p} in One Night Ultimate Werewolf")

    rng = random.Random(args.seed)
    results = []
    for g in range(args.episodes):
        rec.episode({"game": g + 1, "players": PLAYERS, "starting_roles": dict(ROLES), "center_cards": list(CENTER),
                     "discussion_rounds": DISCUSS_ROUNDS})
        game = Game(player_llm, plugin, rec, rng, g)
        res = game.play()
        results.append((game, res))
        print(f"game {g + 1}: Team {res['winner']} wins · votes {res['votes']} · final {res['final_roles']}", flush=True)
        usage = {"players": dict(player_llm.usage)} if args.dry_run else \
            {"players": dict(player_llm.usage), "memory_agent": dict(memory_llm.usage)}
        rec.save(args.out, usage=usage)

    # leak check: no player's private text may appear in another player's observe input
    leaks = 0
    for ep in rec.data["episodes"]:
        priv = {p: [e["content"].split("] ", 1)[1] for e in ep["events"]
                    if e["t"] == "env" and e["data"].get("visible_to") == [p]] for p in PLAYERS}
        for e in ep["events"]:
            if e["t"] == "hook" and e["name"] == "observe":
                seen = " ".join(x["content"] for x in e["input"])
                for other, texts in priv.items():
                    if other != e["agent"]:
                        leaks += sum(1 for t in texts if t in seen)
    print("private-information leaks into other players' observe inputs:", leaks)
    print("usage:", rec.data["usage"])


if __name__ == "__main__":
    main()
