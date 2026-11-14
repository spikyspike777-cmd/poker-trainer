import random

import pytest

from poker_coach import coach
from poker_coach.engine import Game, H, V


@pytest.fixture(autouse=True)
def fast_solver(monkeypatch):
    monkeypatch.setattr(coach.SOLVER, "iterations", 30)


def test_random_playouts_conserve_chips():
    r = random.Random(7)
    g = Game(r)
    showdowns = 0
    for _ in range(150):
        g.new_hand()
        dead = g.dead
        steps = 0
        while not g.finished:
            opts = g.options()
            assert opts, "ход героя без доступных действий"
            assert g.stack[H] + g.stack[V] + g.pot == pytest.approx(200 + dead)
            g.hero_act(r.choice(opts))
            steps += 1
            assert steps < 40
        assert g.stack[H] + g.stack[V] == pytest.approx(200 + dead)
        assert g.decisions or g.finished
        showdowns += any("Вскрытие" in line for line in g.log)
    assert showdowns >= 1


def test_following_coach_always_matches():
    r = random.Random(3)
    g = Game(r)
    for _ in range(50):
        g.new_hand()
        while not g.finished:
            adv = g.advice(H)
            group = {"fold": "fold", "check": "check", "call": "call"}.get(adv.action, "raise")
            opts = [o for o in g.options() if o.kind == group]
            if adv.action == "allin":
                opts = opts[-1:]
            assert opts, (adv.action, [o.label for o in g.options()], g.log)
            g.hero_act(opts[0])
            assert g.last.matched
