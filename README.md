# EvolveMAS

**Test-time self-evolution for decentralized multi-agent systems, as a plug-in.**

[Paper (arXiv, coming soon)](#citation) · [Project page](https://wuyalun.github.io/EvolveMAS/)

EvolveMAS lets LLM agents improve from experience at test time without updating model weights. Each main agent is paired with its own **MemoryAgent**, which keeps an isolated Structured Memory outside the agent's reasoning context. During **Within-Episode Online Adaptation**, the MemoryAgent updates Episodic, Semantic and Working memory from the evidence the agent observed. During **Cross-Episode Knowledge Evolution**, the episode's memory and its task-native result are distilled into reusable Interaction Knowledge: **Interaction Strategies** and **Partner Profiles**, which are injected into later episodes. The same protocol covers cooperative, mixed-motive and adversarial settings. This package is a minimal, host-agnostic implementation of that protocol: your multi-agent system keeps its own agents, message routing and scoring, and calls three hooks.

## Install

```bash
pip install -e ".[anthropic]"     # Claude backend
pip install -e ".[openai]"        # any OpenAI-compatible endpoint
pip install -e ".[dev]"           # tests
```

The core package has no dependencies; a backend is any callable `llm(system, user) -> str`.

## Quickstart

```python
from evolvemas import EvolveMAS, AnthropicLLM

plugin = EvolveMAS("runs/memory", AnthropicLLM())     # one store, one MemoryAgent per registered agent
plugin.register("alice", profile="seller")

for episode in episodes:
    plugin.begin_episode("alice")                       # refresh Working memory and episode-only facts
    while not episode.done:
        memory = plugin.context("alice")                # [M; K] block to add to alice's prompt
        reply = alice.act(prompt + "\n\n[Your memory]\n" + memory)
        seen = episode.step(reply)                      # what alice observed this turn
        plugin.observe("alice", seen)                   # evidence -> MemoryAgent -> M
    plugin.end_episode("alice", episode.score())        # (M, result, K) -> distilled K for next episode
```

`observe` takes strings or `{"speaker": ..., "content": ...}` dicts. Only pass what that agent actually saw: memories are isolated per agent, and in hidden-information games the host decides visibility.

A runnable example is in `examples/negotiation.py`:

```bash
python examples/negotiation.py --backend mock --episodes 3     # offline, scripted, no API key
python examples/negotiation.py --episodes 5 --out runs/demo     # Claude
python examples/negotiation.py --baseline                       # plugin attached, online and evolve off
```

The `mock` backend scripts both the role-play and the MemoryAgent so the full loop runs without network access. It checks the plumbing; its numbers are not evidence about EvolveMAS.

## What is stored

Each agent gets its own directory under the store root:

| File | Contents |
|---|---|
| `state.json` | Machine-readable memory and knowledge (source of truth) |
| `MEMORY.md` | Episodic, Semantic and Working memory, human-readable |
| `KNOWLEDGE.md` | Interaction Strategies and Partner Profiles, human-readable |
| `trajectory.jsonl` | Raw evidence the agent observed, append-only; episodic entries cite its line numbers |
| `history.jsonl` | A snapshot after every update, for replay and audit (`history=False` turns it off) |

Semantic facts carry a scope: `episode` facts (current offers, who said what) are dropped at `begin_episode`, `persistent` facts carry over. Restarting with the same store directory resumes where the agent left off.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `online` | `True` | Run the MemoryAgent after each `observe` (Within-Episode Online Adaptation) |
| `evolve` | `True` | Distill knowledge at `end_episode` (Cross-Episode Knowledge Evolution) |
| `evolve_llm` | same as `llm` | A separate model for distillation |
| `max_episodic` | 20 | Episodic entries kept |
| `max_strategies` | 8 | Interaction Strategies kept |
| `inject_max_chars` | 4000 | Size cap on `context()`; live episode state comes first, older knowledge is truncated first |
| `history` | `True` | Write `history.jsonl` |

Setting `online` and `evolve` independently gives the 2×2 ablation of memory governance and knowledge reuse. With both off the plugin only records trajectories, which serves as the baseline.

A malformed MemoryAgent or distillation reply is logged (logger `evolvemas`) and that update is skipped; the host's episode keeps running. If a reply leaves out `strategies` or `partners`, the existing knowledge is kept.

## Backends

- `AnthropicLLM(model="claude-opus-5-5", effort="low")` uses the official `anthropic` SDK and reads `ANTHROPIC_API_KEY` or an `ant auth login` profile. A refusal raises `RuntimeError`. If you want automatic fallback to another model on refusal, the Claude API offers server-side fallbacks as a beta; pass your own client or wrap the backend to use it.
- `OpenAICompatLLM(model, base_url, api_key)` works with any OpenAI-compatible chat endpoint.
- `ScriptedLLM(respond)` calls your own function. It is useful for tests and offline demos.
- Any callable `llm(system, user) -> str` works.

## Tests

```bash
pytest
```

## Citation

```bibtex
@article{wu2026evolvemas,
  title   = {EvolveMAS: A Unified Self-Evolution Framework for Decentralized Multi-Agent Systems},
  author  = {Wu, Yalun and Ma, Qiwei and Xu, Bingbing and Zhang, Dan and Bai, Long and Wang, Boyang and
             Li, Jiatong and Lv, Zheqi and Yuan, Shaozu and Wang, Peiying and Zhang, An and
             Wang, Jiawei and Chua, Tat-Seng},
  journal = {arXiv preprint arXiv:TBD},
  year    = {2026}
}
```

## License

MIT
