import json

from evolvemas import EvolveMAS


class MockLLM:
    """Returns fixed JSON; records every call so tests can check what the plugin sent."""

    def __init__(self):
        self.calls = []

    def __call__(self, system, user):
        self.calls.append(user)
        if "Revise the knowledge" in user:
            return json.dumps({"strategies": ["Anchor high, concede in small steps"],
                               "partners": {"Bob": {"tendency": "splits the difference", "reliability": "medium - one episode"}}})
        return "```json\n" + json.dumps({"episodic": ["Bob offered $40", "Bob offered $40"],
                                         "semantic": [{"key": "bob_opening", "value": "$40", "confidence": "low", "scope": "persistent"},
                                                      {"key": "current_offer", "value": "$40", "confidence": "high", "scope": "episode"}],
                                         "working": "Bob at $40; I hold $55."}) + "\n```"


def make(tmp_path, **kw):
    llm = MockLLM()
    p = EvolveMAS(tmp_path, llm, **kw)
    p.register("seller", "sells a table")
    p.register("buyer", "buys a table")
    return p, llm


def test_three_hooks_and_isolation(tmp_path):
    p, llm = make(tmp_path)
    p.begin_episode()
    assert p.context("seller") == ""
    p.observe("seller", [{"speaker": "Bob", "content": "I can do $40."}])
    st = p.state("seller")
    assert [e.summary for e in st.episodic] == ["Bob offered $40"]       # duplicate inside one reply dropped
    assert st.semantic["bob_opening"].value == "$40"
    assert "Bob at $40" in p.context("seller")
    assert p.state("buyer").episodic == []                               # memories are isolated per agent
    p.end_episode("seller", {"deal": 48, "reward": 0.4})
    st = p.state("seller")
    assert st.episode == 1 and st.strategies == ["Anchor high, concede in small steps"]
    assert "Bob" in st.partners
    p.begin_episode("seller")
    ctx = p.context("seller")
    assert "Anchor high" in ctx and "Working memory" not in ctx       # K carried over, W refreshed
    assert "bob_opening" in ctx and "current_offer" not in ctx        # episode-local facts do not leak


def test_observe_dedups_across_turns(tmp_path):
    p, _ = make(tmp_path)
    p.observe("seller", ["Bob: $40"])
    p.observe("seller", ["Bob: $40 again"])
    assert len(p.state("seller").episodic) == 1


def test_ablation_switches(tmp_path):
    p, llm = make(tmp_path, online=False, evolve=False)
    p.observe("seller", ["Bob: $40"])
    p.end_episode("seller", 0.0)
    assert llm.calls == []                                               # no MemoryAgent / D_a calls
    assert (tmp_path / "seller" / "trajectory.jsonl").read_text().count("\n") == 1


def test_resume_from_disk(tmp_path):
    p, _ = make(tmp_path)
    p.observe("seller", ["Bob: $40"])
    p.end_episode("seller", 0.5)
    q = EvolveMAS(tmp_path, MockLLM())
    q.register("seller")
    assert q.state("seller").episode == 1 and q.state("seller").strategies
    assert q.state("seller").profile == "sells a table"
