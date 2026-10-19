"""Extract the paper's freeze-eval chains into one JSON for the website.

Reads a local clone of the private experiment-records archive and writes
docs/runs/paper_evolution.json: for each setting, the frozen held-out score at every
checkpoint (train N episodes with EvolveMAS -> freeze memory -> evaluate), and each
agent's Interaction Strategies and Partner Profiles at that checkpoint.

    python bench/extract_paper_records.py /path/to/evolvemas-experiment-records

Only numbers and knowledge text are exported; trajectory references are stripped and the
output is scanned for key-shaped strings before it is written.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "runs" / "paper_evolution.json"
DELIB_GAME = ROOT / "bench" / "data" / "deliberation"

REF = re.compile(r"\s*\[\[[^\]]*\]\]")
KV = re.compile(r"^- (?P<key>[^=]+?)\s*=\s*==(?P<val>.*?)==\s*(?:\((?P<conf>high|medium|low)[^)]*\))?")


def ckpt_episode(name: str) -> int | None:
    m = re.match(r"ckpt_(\d+)$", name)
    return int(m.group(1)) if m else None


def parse_skill(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = REF.sub("", line).strip()
        m = KV.match(line)
        if m:
            out.append({"key": m.group("key").strip(), "text": m.group("val").strip(), "confidence": m.group("conf") or ""})
        elif line.startswith("- ") and "==" not in line:
            out.append({"key": "", "text": line[2:].strip(), "confidence": ""})
    return out


def parse_partners(path: Path):
    out, cur = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = REF.sub("", line).rstrip()
        if line.startswith("### "):
            cur = line[4:].strip()
            out[cur] = {"tendency": "", "reliability": ""}
        elif cur and line.strip().startswith("- tendency:"):
            out[cur]["tendency"] = line.split(":", 1)[1].strip()
        elif cur and line.strip().startswith("- reliability:"):
            out[cur]["reliability"] = line.split(":", 1)[1].strip()
    return out


def semantic_count(memory_md: Path) -> int:
    if not memory_md.exists():
        return 0
    t = memory_md.read_text(encoding="utf-8", errors="replace")
    s = t.split("## Semantic", 1)
    if len(s) < 2:
        return 0
    body = s[1].split("## Working", 1)[0]
    return sum(1 for l in body.splitlines() if l.startswith("- "))


def semantic_items(memory_md: Path):
    t = memory_md.read_text(encoding="utf-8", errors="replace")
    if "## Semantic" not in t:
        return []
    body = t.split("## Semantic", 1)[1].split("## Working", 1)[0]
    out = []
    for l in body.splitlines():
        if l.startswith("- "):
            m = KV.match(REF.sub("", l).strip())
            out.append(f"{m.group('key').strip()}: {m.group('val').strip()}" if m else REF.sub("", l[2:]).strip())
    return out


def knowledge_by_ckpt(chain: Path, eval_prefix: str):
    """Knowledge lives in eval_workmem_ckpt_NNN*/<agent>/<agent>/skills/ (the frozen copy that was evaluated)."""
    res = {}
    for d in sorted(chain.glob(eval_prefix + "*")):
        m = re.search(r"ckpt_(\d+)", d.name)
        if not m or re.search(r"ckpt_\d+_(?!G0)", d.name):
            continue
        ep = int(m.group(1))
        agents = {}
        for skills in sorted(d.glob("*/*/skills")):
            agent = skills.parent.name
            sk = next(iter(skills.glob("*_skill.md")), None)
            pa = next(iter(skills.glob("*_partners.md")), None)
            snap_mem = chain / "snapshots" / f"ckpt_{ep:03d}" / agent / agent / "MEMORY.md"
            strategies = parse_skill(sk) if sk else []
            if not strategies and snap_mem.exists():
                # older chains keep knowledge only as Semantic memory
                strategies = [{"key": "", "text": i, "confidence": ""} for i in semantic_items(snap_mem)]
            agents[agent] = {"strategies": strategies,
                             "partners": parse_partners(pa) if pa else {},
                             "semantic_facts": semantic_count(snap_mem),
                             "from_semantic": not (sk and parse_skill(sk))}
        if agents:
            res[ep] = agents
    return res


# ---------------------------------------------------------------- Deliberation scoring (paper's rule)
def delib_agents(config_path: Path):
    agents = {}
    for line in config_path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, fname, role = parts[0], parts[1], parts[2]
        lines = (DELIB_GAME / "scores_files" / f"{fname}.txt").read_text().strip().split("\n")
        scores = {string.ascii_uppercase[i]: [int(v) for v in l.split(",")] for i, l in enumerate(lines[:-1])}
        agents[name] = {"scores": scores, "min": int(lines[-1]), "role": role}
    return agents


def delib_final_agree(history: Path, agents):
    p1 = next(n for n, c in agents.items() if c["role"] == "p1")
    d = json.loads(history.read_text())
    if len(d.get("rounds", [])) < 10:
        return None
    last = None
    for r in d["rounds"]:
        if r.get("agent") != p1:
            continue
        ans = (r.get("public_answer") or "").replace("\n", "")
        deal = []
        for i in range(5):
            o = re.findall(f"{string.ascii_uppercase[i]}[1-9]", ans)
            deal.append(o[0] if o else "")
        if all(deal):
            last = deal
    if last is None:
        return None

    def value(sc):
        tot = 0
        for it in last:
            k, n = it[0], int(it[1])
            if k not in sc or n > len(sc[k]):
                return 0
            tot += sc[k][n - 1]
        return tot
    return sum(1 for c in agents.values() if value(c["scores"]) > c["min"])


def delib_chain(records: Path):
    base = records / "LLM-Deliberation" / "base_friday_output"
    pts = []
    for name, ep in [("freeze_g0_clean_G0", -1)] + [(f"freeze_eval_ckpt_{k:03d}_G3", k) for k in (0, 20, 40, 60, 80, 100)]:
        sess = sorted(base.glob(name + "_sess*"))
        vals = []
        for s in sess:
            h = next(iter(s.glob("history*.json")), None)
            if h:
                v = delib_final_agree(h, delib_agents(s / "config.txt"))
                if v is not None:
                    vals.append(v)
        if vals:
            pts.append({"episode": ep, "score": round(statistics.mean(vals), 3), "n": len(vals)})
    know = {}
    for d in sorted((records / "self-evolving" / "freeze_curve_delib_full" / "snapshots").glob("ckpt_*")):
        ep = ckpt_episode(d.name)
        agents = {}
        for f in sorted(d.glob("*.md")):
            t = f.read_text(encoding="utf-8", errors="replace")
            sem = t.split("## Semantic", 1)[1].split("## Working", 1)[0] if "## Semantic" in t else ""
            items = [REF.sub("", l[2:]).strip() for l in sem.splitlines() if l.startswith("- ")]
            agents[f.stem] = {"strategies": [{"key": "", "text": i, "confidence": ""} for i in items], "partners": {},
                              "semantic_facts": len(items), "from_semantic": True}
        know[ep] = agents
    return pts, know


def werewolf_village_points(chain: Path):
    pts = []
    for d in sorted((chain / "eval_results").glob("ckpt_*")):
        ep = ckpt_episode(d.name)
        if ep is None:
            continue
        files = sorted(d.glob("episodes_*.jsonl"))
        compact = [f for f in files if "-" not in f.name] or files
        L = [json.loads(l) for f in compact for l in open(f)]
        if L:
            pts.append({"episode": ep, "score": round(sum(r["metrics"]["village_win"] for r in L) / len(L), 3), "n": len(L)})
    return pts


def main(records: Path):
    se = records / "self-evolving"
    chains = []

    def add(cid, setting, title, metric, chain_dir, points, eval_prefix="eval_workmem_ckpt_", knowledge=None, note=""):
        chains.append({"id": cid, "setting": setting, "title": title, "metric": metric, "note": note,
                       "points": [p for p in points if p["n"] >= 10],
                       "knowledge": {str(k): v for k, v in sorted((knowledge if knowledge is not None
                                                                     else knowledge_by_ckpt(chain_dir, eval_prefix)).items())}})

    fc = lambda p: json.loads((p / "freeze_curve.json").read_text())["rows"]
    add("bbh", "cooperative", "BBH", "Accuracy on 100 held-out questions",
        se / "freeze_curve_bbh_v2", [{"episode": r["episode"], "score": r["accuracy"], "n": r["n"]} for r in fc(se / "freeze_curve_bbh_v2")])
    add("humaneval", "cooperative", "HumanEval", "Pass rate on 64 held-out problems",
        se / "freeze_curve_humaneval_v2", [{"episode": r["episode"], "score": r["pass_rate"], "n": r["n"]} for r in fc(se / "freeze_curve_humaneval_v2")])
    add("sotopia", "mixed-motive", "SOTOPIA", "Goal score on 100 held-out episodes (scenarios not seen in training)",
        se / "freeze_curve_sotopia_v3", [{"episode": r["episode"], "score": r["goal"], "n": r["n"]} for r in fc(se / "freeze_curve_sotopia_v3")])
    dp, dk = delib_chain(records)
    add("deliberation", "mixed-motive", "LLM-Deliberation", "Parties accepting the final proposal (of 6), frozen memory injected but not updated",
        None, dp, knowledge=dk,
        note="This code version stores knowledge as each party's Semantic memory; there are no separate strategy or partner files. Episode −1 is the no-memory reference run.")
    add("werewolf_village", "adversarial", "Werewolf · village side", "Village win rate over 100 held-out games",
        se / "freeze_curve_werewolf_full_village", werewolf_village_points(se / "freeze_curve_werewolf_full_village"),
        note="Only the village team keeps memory; knowledge is stored as Semantic memory in this chain. The 100-episode checkpoint was not evaluated, and a 1-game evaluation at episode 1 is left out.")
    add("werewolf_wolf", "adversarial", "Werewolf · wolf side", "Wolf deception success over 100 held-out games",
        se / "freeze_curve_werewolf_full_werewolf",
        [{"episode": r["episode"], "score": r["dec_dc"], "n": r["n"]} for r in fc(se / "freeze_curve_werewolf_full_werewolf") if r["ckpt"].count("_") == 1],
        note="Only the werewolf keeps memory.")

    data = {"source": "EvolveMAS experiment records: freeze-eval chains (train with EvolveMAS, snapshot every few episodes, freeze, evaluate on a held-out set)",
            "chains": chains}
    text = json.dumps(data, ensure_ascii=False, indent=1)
    bad = re.findall(r"(?<![A-Za-z])sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{10,}|ep-20\d{6}-\w+|aigc\.\w+|https?://[^\s\"]+", text)
    for t in [x for x in os.environ.get("BENCH_BLOCKLIST", "").split(",") if x.strip()]:
        bad += re.findall(re.escape(t.strip()), text, re.I)
    if bad:
        raise SystemExit(f"refusing to write: found {sorted(set(bad))[:8]}")
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text) // 1024} KB)")
    for c in chains:
        print(c["id"], [(p["episode"], p["score"]) for p in c["points"]], "| knowledge ckpts", list(c["knowledge"])[:12])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
