import importlib.util
import json
import logging
from pathlib import Path

import pytest

from evolvemas import EvolveMAS, ScriptedLLM, prompts

ROOT = Path(__file__).resolve().parents[1]


def load_example():
    spec = importlib.util.spec_from_file_location("negotiation", ROOT / "examples" / "negotiation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plugin_with(tmp_path, online_reply, evolve_reply="{}", **kw):
    def respond(system, user):
        return online_reply if system == prompts.ONLINE_SYSTEM else evolve_reply
    llm = ScriptedLLM(respond)
    p = EvolveMAS(tmp_path, llm, **kw)
    p.register("seller")
    return p, llm


# --- bad model output never stops the host -----------------------------------
@pytest.mark.parametrize("reply", ["not json at all", "[1, 2, 3]", '{"episodic": "oops", "semantic": {"a": 1}}'])
def test_bad_online_reply_is_skipped(tmp_path, caplog, reply):
    p, _ = plugin_with(tmp_path, reply)
    with caplog.at_level(logging.WARNING, logger="evolvemas"):
        p.observe("seller", ["Bob: $40"])
    st = p.state("seller")
    assert st.turn == 1 and st.episodic == [] and st.semantic == {}
    assert (tmp_path / "seller" / "trajectory.jsonl").read_text().count("\n") == 1   # evidence still logged


def test_bad_online_reply_logs_a_warning(tmp_path, caplog):
    p, _ = plugin_with(tmp_path, "not json")
    with caplog.at_level(logging.WARNING, logger="evolvemas"):
        p.observe("seller", ["Bob: $40"])
    assert "unparseable reply" in caplog.text


def test_non_dict_semantic_items_are_skipped(tmp_path):
    reply = json.dumps({"semantic": ["bob is cheap", {"key": "bob_opening", "value": "$40", "confidence": "sure"}],
                        "working": 7})
    p, _ = plugin_with(tmp_path, reply)
    p.observe("seller", ["Bob: $40"])
    st = p.state("seller")
    assert list(st.semantic) == ["bob_opening"]
    assert st.semantic["bob_opening"].confidence == "low"        # unknown label falls back to low
    assert st.working == ""                                      # non-string working memory ignored


def test_bad_evolve_reply_keeps_knowledge_and_closes_episode(tmp_path):
    p, _ = plugin_with(tmp_path, "{}", evolve_reply="garbage")
    st = p.state("seller")
    st.strategies = ["Anchor high"]
    p.end_episode("seller", {"deal": 48})
    assert st.episode == 1 and st.strategies == ["Anchor high"]


# --- missing keys mean "no revision" -------------------------------------------
def test_missing_keys_keep_existing_knowledge(tmp_path):
    p, _ = plugin_with(tmp_path, "{}", evolve_reply=json.dumps({"partners": {"Bob": {"tendency": "t", "reliability": "r"}}}))
    st = p.state("seller")
    st.strategies = ["Anchor high"]
    p.end_episode("seller", 0.5)
    assert st.strategies == ["Anchor high"] and "Bob" in st.partners

    p.evolve_llm = ScriptedLLM(lambda s, u: json.dumps({"strategies": ["Concede slowly"]}))
    p.end_episode("seller", 0.5)
    assert st.strategies == ["Concede slowly"] and "Bob" in st.partners


def test_empty_strategy_list_is_an_explicit_reset(tmp_path):
    p, _ = plugin_with(tmp_path, "{}", evolve_reply=json.dumps({"strategies": []}))
    p.state("seller").strategies = ["Anchor high"]
    p.end_episode("seller", 0.5)
    assert p.state("seller").strategies == []


# --- context ordering and truncation ---------------------------------------------
def test_context_puts_live_state_first_and_truncates_old_knowledge(tmp_path):
    reply = json.dumps({"semantic": [{"key": "bob_offer", "value": "$40", "confidence": "high", "scope": "episode"}],
                        "working": "Bob at $40; I hold $55."})
    p, _ = plugin_with(tmp_path, reply, inject_max_chars=200)
    st = p.state("seller")
    st.strategies = [f"long strategy number {i} " + "x" * 40 for i in range(10)]
    p.observe("seller", ["Bob: $40"])
    ctx = p.context("seller")
    assert ctx.startswith("Working memory for this episode:\nBob at $40")
    assert ctx.index("bob_offer") < ctx.index("Interaction Strategies")
    assert ctx.endswith("[memory truncated]") and "number 9" not in ctx
    assert len(ctx) <= 200 + len("\n...[memory truncated]")


# --- storage ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", "", ".hidden", "x..y"])
def test_agent_id_cannot_leave_the_store(tmp_path, bad):
    p = EvolveMAS(tmp_path, ScriptedLLM(lambda s, u: "{}"))
    with pytest.raises(ValueError):
        p.register(bad)


def test_history_can_be_turned_off(tmp_path):
    p, _ = plugin_with(tmp_path, "{}", history=False)
    p.observe("seller", ["Bob: $40"])
    assert not (tmp_path / "seller" / "history.jsonl").exists()


def test_trajectory_line_refs_survive_resume(tmp_path):
    p, _ = plugin_with(tmp_path, json.dumps({"episodic": ["first"]}))
    p.observe("seller", ["a", "b"])
    q, _ = plugin_with(tmp_path, json.dumps({"episodic": ["second"]}))   # fresh store object, same directory
    q.observe("seller", ["c"])
    refs = [e.ref for e in q.state("seller").episodic]
    assert refs == [2, 3]


def test_observe_with_no_events_does_not_advance_the_turn(tmp_path):
    p, llm = plugin_with(tmp_path, "{}")
    p.observe("seller", [])
    assert p.state("seller").turn == 0 and llm.calls == []


# --- negotiation example scoring ---------------------------------------------------
def test_deal_needs_the_other_side_to_accept():
    ev = load_example().evaluate
    assert ev([("Lena", "OFFER $50"), ("Marco", "ACCEPT $50")]) == {"deal": 50.0, "seller_profit": 10.0}
    assert ev([("Marco", "OFFER $45"), ("Lena", "ACCEPT $45")]) == {"deal": 45.0, "seller_profit": 5.0}
    assert ev([("Lena", "OFFER $50"), ("Lena", "ACCEPT $50")])["deal"] is None        # accepting yourself
    assert ev([("Lena", "OFFER $50"), ("Marco", "ACCEPT $45")])["deal"] is None       # price not on the table
    assert ev([("Lena", "OFFER $50"), ("Marco", "WALK AWAY")])["deal"] is None
    assert ev([("Lena", "OFFER $50"), ("Marco", "OFFER $45")])["deal"] is None


def test_mock_demo_runs_offline(tmp_path, capsys, monkeypatch):
    mod = load_example()
    monkeypatch.setattr("sys.argv", ["negotiation.py", "--backend", "mock", "--episodes", "3", "--out", str(tmp_path)])
    mod.main()
    log = json.loads((tmp_path / "run.json").read_text())["log"]
    assert [e["result"]["deal"] for e in log] == [48.0, 50.0, 52.0]
    assert "Open at $54 and hold" in (tmp_path / "memory" / "Lena" / "KNOWLEDGE.md").read_text()


def test_persistent_facts_expire_when_not_seen_again(tmp_path):
    import json as _json
    from evolvemas import EvolveMAS, ScriptedLLM

    def respond(system, user):
        if "Revise the knowledge" in user:
            return _json.dumps({"strategies": [], "partners": {}})
        fact = [{"key": "old_fact", "value": "x", "confidence": "high", "scope": "persistent"}] if "episode 0," in user else []
        return _json.dumps({"episodic": [], "semantic": fact, "working": ""})

    p = EvolveMAS(tmp_path, ScriptedLLM(respond), fact_ttl=2)
    p.register("a")
    for ep in range(5):
        p.begin_episode("a")
        p.observe("a", ["something happened"])
        alive = "old_fact" in p.state("a").semantic
        assert alive == (ep <= 2), (ep, alive)       # kept for 2 episodes after last seen, then dropped
        p.end_episode("a", {"ok": True})


def test_observe_accepts_speaker_content_pairs(tmp_path):
    import json as _json
    from evolvemas import EvolveMAS, ScriptedLLM
    seen = []

    def respond(system, user):
        seen.append(user)
        return _json.dumps({"episodic": [], "semantic": [], "working": ""})

    p = EvolveMAS(tmp_path, ScriptedLLM(respond))
    p.register("a")
    p.observe("a", [("b", "I offer $40"), {"speaker": "c", "content": "no"}, "a note"])
    assert "b: I offer $40" in seen[-1] and "c: no" in seen[-1] and "env: a note" in seen[-1]
