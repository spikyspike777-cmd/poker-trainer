import random

from poker_coach import charts, coach
from poker_coach.calc import is_weak_showdown
from poker_coach.coach import analyze
from poker_coach.engine import Game, H
from poker_coach.profile import Adjust, PlayerStats
from poker_coach.state import GameState, to_eval7


def test_small_sample_gives_no_label():
    st = PlayerStats(hands=5)
    for _ in range(4):
        st.record_postflop("bet", False)
    assert st.adjust().label == "" and not st.adjust().exploit


def test_station_and_aggressor_are_recognised():
    station = PlayerStats(hands=60)
    for _ in range(30):
        station.record_postflop("call", True)
    for _ in range(12):
        station.record_preflop(True, False, 0.25)
    a = station.adjust()
    assert "лузовый" in a.label and "редко сбрасывает" in a.label and a.fold_to_bet < 0.14
    lag = PlayerStats(hands=60)
    for _ in range(30):
        lag.record_postflop("bet", False)
    assert "агрессивный" in lag.adjust().label and lag.adjust().bluff_scale > 1.35


def test_scale_range():
    base = charts.combos(charts.RFI["CO"])
    tight, loose = charts.scale_range(charts.RFI["CO"], 0.6), charts.scale_range(charts.RFI["CO"], 1.5)
    assert len(tight) < len(base) < len(loose)
    assert {frozenset(c) for c in tight} <= {frozenset(c) for c in base} <= {frozenset(c) for c in loose}
    assert any({str(c[0])[0], str(c[1])[0]} == {"A"} for c in tight)        # тузы остаются


def test_weak_showdown():
    board = to_eval7(["Kh", "9d", "4c", "7s", "2h"])
    assert is_weak_showdown(to_eval7(["Ac", "Qd"]), board)
    assert is_weak_showdown(to_eval7(["4d", "5d"]), board)
    assert not is_weak_showdown(to_eval7(["Kc", "Qd"]), board)
    assert not is_weak_showdown(to_eval7(["9c", "8d"]), board)


def test_coach_stops_bluffing_station_and_bluffs_nit():
    kw = dict(hero_cards="9s 8s", board="7d 6c Kh", hero_pos="BTN", villain_pos="BB",
              preflop_line="hero_open_villain_call", pot=5.5, stack=97.5)
    assert analyze(GameState(**kw)).action == "bet"                                      # полублеф по умолчанию
    station = Adjust(fold_to_bet=0.08, label="редко сбрасывает", hands=80, notes=["x"])
    a = analyze(GameState(**kw, villain_adj=station))
    assert a.action == "check" and "Профиль оппонента" in a.reasons[0]
    nit = Adjust(fold_to_bet=0.55, label="часто сбрасывает", hands=80, notes=["x"])
    air = dict(kw, hero_cards="5c 4c", board="Ks Qd 8h")
    assert analyze(GameState(**air)).action == "check"
    assert analyze(GameState(**air, villain_adj=nit)).action == "bet"


def test_bots_differ(monkeypatch):
    monkeypatch.setattr(coach.SOLVER, "iterations", 20)
    seen = {}
    for style in ("nit", "station"):
        g = Game(n_players=2, styles=style, rng=random.Random(2), use_profile=False)
        r = random.Random(3)
        for i in range(150):
            if i:
                g.new_hand()
            while not g.finished:
                opts = g.options()
                agg = [o for o in opts if o.kind == "raise"]
                g.hero_act(agg[0] if agg and r.random() < 0.7 else opts[-2] if len(opts) > 1 else opts[0])
        seen[style] = g.players[1].stats
    fold = lambda s: s.folded_to_bet / s.faced_bet
    assert fold(seen["station"]) < 0.15 < 0.3 < fold(seen["nit"])
    assert "редко сбрасывает" in seen["station"].adjust().label
