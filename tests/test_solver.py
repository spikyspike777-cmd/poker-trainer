from poker_coach import charts
from poker_coach.coach import analyze
from poker_coach.solver import RiverCFR, Spot
from poker_coach.state import GameState, to_eval7

BOARD = to_eval7(["Kh", "7h", "2c", "9s", "3d"])


def _ranges():
    live = lambda r: [c for c in charts.combos(r) if not set(c) & set(BOARD)]
    return live(charts.vs_open_ranges("BB", "BTN")[1]), live(charts.RFI["BTN"])


def _solve(hand, facing=0.0, ip=False):
    hr, vr = _ranges()
    res = RiverCFR(iterations=200).solve(Spot(tuple(to_eval7(hand)), BOARD, hr, vr, 12, 90, ip, facing))
    return res, {a.kind: a for a in res.actions}


def test_frequencies_sum_to_one():
    res, _ = _solve(["Kd", "Qd"])
    assert abs(sum(a.freq for a in res.actions) - 1) < 1e-4
    assert 0 <= res.equity <= 1


def test_set_never_folds_and_air_never_calls():
    _, strong = _solve(["2d", "2s"], facing=8)
    assert strong["fold"].freq < 0.01
    assert strong["raise"].ev > 8
    _, air = _solve(["6d", "5d"], facing=8)
    assert air["call"].freq < 0.05 and air["call"].ev < 0


def test_fold_ev_is_zero_and_bluffcatcher_is_close():
    _, mid = _solve(["As", "7s"], facing=8)
    assert abs(mid["fold"].ev) < 1e-6
    assert abs(mid["call"].ev) < 1.5          # блеф-кэтчер в равновесии почти безразличен


def test_coach_uses_solver_on_river_only():
    kw = dict(hero_cards="As 7s", hero_pos="BB", villain_pos="BTN", preflop_line="villain_open_hero_call", stack=90)
    river = analyze(GameState(board="Kh 7h 2c 9s 3d", pot=20, to_call=8, **kw))
    assert river.mix and "солвер" in river.source
    assert abs(sum(m[2] for m in river.mix) - 1) < 1e-4
    turn = analyze(GameState(board="Kh 7h 2c 9s", pot=20, to_call=8, **kw))
    assert not turn.mix and "эвристика" in turn.source
