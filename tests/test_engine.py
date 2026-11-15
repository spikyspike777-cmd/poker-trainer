import random

import pytest

from poker_coach import coach
from poker_coach.engine import Game, H


@pytest.fixture(autouse=True)
def fast_solver(monkeypatch):
    monkeypatch.setattr(coach.SOLVER, "iterations", 20)


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_random_playouts_conserve_chips(n):
    r = random.Random(7 + n)
    g = Game(n_players=n, styles="random", rng=r)
    showdowns = multiway = 0
    for _ in range(60):
        g.new_hand()
        steps = 0
        while not g.finished:
            opts = g.options()
            assert opts, "ход героя без доступных действий"
            assert sum(p.stack for p in g.players) + g.pot == pytest.approx(100 * n)
            assert len({p.pos for p in g.players}) == n
            if len(g.board) >= 3 and len(g.alive()) >= 3:
                multiway += 1
            g.hero_act(r.choice(opts))
            steps += 1
            assert steps < 40
        assert sum(p.stack for p in g.players) == pytest.approx(100 * n)
        assert g.net == pytest.approx(g.players[H].stack - 100)
        showdowns += bool(g.shown)
    assert showdowns >= 1
    if n >= 4:
        assert multiway >= 1


def test_following_coach_always_matches():
    r = random.Random(3)
    g = Game(n_players=4, rng=r)
    for _ in range(40):
        g.new_hand()
        while not g.finished:
            adv = g.advice(H)
            group = {"fold": "fold", "check": "check", "call": "call"}.get(adv.action, "raise")
            opts = [o for o in g.options() if o.kind == group]
            assert opts, (adv.action, [o.label for o in g.options()], g.log)
            g.hero_act(opts[-1] if adv.action == "allin" else opts[0])
            assert g.last.matched


def test_positions_rotate_and_blinds_posted():
    g = Game(n_players=6, rng=random.Random(1))
    seen = set()
    for _ in range(12):
        g._deal()
        seen.add(g.players[H].pos)
        blinds = {p.pos: p.inv for p in g.players if p.inv}
        assert blinds == {"SB": 0.5, "BB": 1.0}
    assert seen == {"UTG", "HJ", "CO", "BTN", "SB", "BB"}
    g2 = Game(n_players=2, rng=random.Random(1))
    g2._deal()
    assert {p.pos: p.inv for p in g2.players} == {"BTN": 0.5, "BB": 1.0}
