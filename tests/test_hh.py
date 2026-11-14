from pathlib import Path

from poker_coach import coach
from poker_coach.hh import review_session

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "pokerstars_sample.txt"


def test_sample_session(monkeypatch):
    monkeypatch.setattr(coach.SOLVER, "iterations", 30)
    hands = review_session(SAMPLE.read_text(encoding="utf-8"))
    assert [h.hero_pos for h in hands] == ["BTN", "CO", "HJ"]
    assert [h.net_bb for h in hands] == [-13.5, 1.5, -13.0]
    h1 = hands[0]
    assert [d.street for d in h1.decisions] == ["preflop", "flop", "turn", "river"]
    river = h1.decisions[-1].state
    assert (river.pot, river.to_call, river.stack) == (19.5, 8.0, 94.5)
    assert river.history == [("flop", "villain", "check"), ("flop", "hero", "bet"), ("flop", "villain", "call"),
                             ("turn", "villain", "check"), ("turn", "hero", "check"), ("river", "villain", "bet")]
    assert hands[1].decisions[0].matched is False            # 72o с CO — тренер сбрасывает
    last = hands[2].decisions[-1]
    assert last.state.hero_invested == 5.5 and last.state.to_call == 11.0
    assert all(d.matched is not None for h in hands for d in h.decisions)


def test_multiway_is_marked_unsupported():
    text = SAMPLE.read_text(encoding="utf-8").replace("Bravo: folds\nHero: raises $0.15 to $0.25\nDelta: folds\nEcho: calls",
                                                      "Bravo: calls $0.10\nHero: raises $0.15 to $0.25\nDelta: folds\nEcho: calls", 1)
    d = review_session(text)[0].decisions[0]
    assert d.matched is None and d.note
