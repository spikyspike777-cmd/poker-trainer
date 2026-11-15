import pytest
from pydantic import ValidationError

from poker_coach import calc, charts
from poker_coach.coach import analyze
from poker_coach.state import GameState, to_eval7


def gs(**kw):
    base = dict(hero_pos="BTN", villain_pos="BB", preflop_line="hero_open_villain_call", pot=5.5, stack=97.5)
    return GameState(**{**base, **kw})


def test_pot_odds():
    assert calc.pot_odds(15, 5) == pytest.approx(0.25)   # бет полбанка: банк 10 + ставка 5
    assert calc.pot_odds(10, 0) == 0


def test_equity_river_exact():
    hero, board = to_eval7(["As", "Ah"]), to_eval7(["Kd", "7c", "2s", "9h", "3d"])
    assert calc.equity_vs_range(hero, charts.combos("KK"), board) == 0      # сет бьёт
    assert calc.equity_vs_range(hero, charts.combos("QQ"), board) == 1
    assert calc.equity_vs_range(hero, charts.combos("AA"), board) == 0.5    # делёжка


def test_equity_known_matchups():
    aa, kk = to_eval7(["As", "Ah"]), charts.combos("KK")
    assert calc.equity_vs_range(aa, kk, [], 100000) == pytest.approx(0.82, abs=0.02)
    # флеш-дро + оверкарты против топ-пары на флопе: около 45-50 %
    eq = calc.equity_vs_range(to_eval7(["Ah", "Qh"]), [tuple(to_eval7(["Ks", "Jd"]))], to_eval7(["Kh", "7h", "2c"]))
    assert 0.42 < eq < 0.50


def test_outs_flush_draw_and_oesd():
    o = calc.outs(to_eval7(["Ah", "Qh"]), to_eval7(["Kh", "7h", "2c"]))
    assert len(o["флеш"]) == 9 and calc.count_outs(o) == 12
    o = calc.outs(to_eval7(["9s", "8s"]), to_eval7(["7d", "6c", "Kh"]))
    assert len(o["стрит"]) == 8
    assert calc.outs(to_eval7(["9s", "8s"]), to_eval7(["7d", "6c", "Kh", "2d", "2c"])) == {}


def test_validation():
    with pytest.raises(ValidationError):
        gs(hero_cards="As As", board="Kh 7h 2c")
    with pytest.raises(ValidationError):
        gs(hero_cards="As Kd", board="Kh 7h")
    with pytest.raises(ValidationError):
        gs(hero_cards="As Kd", board="")                       # постфлоп-линия без борда
    with pytest.raises(ValidationError):
        gs(hero_cards="As Kd", board="Kh 7h 2c", villain_pos="BTN")


def test_preflop_chart_advice():
    """Ручные чарты (use_table=False) — на них играют боты."""
    hand = lambda **kw: analyze(GameState(**kw), use_table=False)
    assert hand(hero_cards="7c 2d", hero_pos="BTN", preflop_line="unopened", pot=1.5).action == "fold"
    a = hand(hero_cards="As Ks", hero_pos="UTG", preflop_line="unopened", pot=1.5)
    assert (a.action, a.amount) == ("raise", 2.5)
    a = hand(hero_cards="Ad Ah", hero_pos="BB", villain_pos="CO", preflop_line="vs_open", pot=4, to_call=1.5)
    assert a.action == "raise" and a.amount == 10
    a = hand(hero_cards="Kd Kh", hero_pos="BTN", villain_pos="CO", preflop_line="vs_4bet", pot=33, to_call=14.5, stack=92.5)
    assert a.action == "allin"


def test_preflop_table_advice():
    """Префлоп-таблица из решений солвера: стандартные линии, мультивей, лимп."""
    a = analyze(GameState(hero_cards="7c 2d", hero_pos="BTN", preflop_line="unopened", pot=1.5))
    assert a.action == "fold" and "солвер" in a.source
    a = analyze(GameState(hero_cards="As Ks", hero_pos="UTG", preflop_line="unopened", pot=1.5))
    assert a.action == "raise" and 2.0 <= a.amount <= 2.5
    a = analyze(GameState(hero_cards="Ad Ah", hero_pos="BB", villain_pos="CO", preflop_line="vs_open", pot=4, to_call=1.5))
    assert a.action == "raise" and a.equity > 0.8
    multiway = dict(hero_pos="BB", preflop_line="other", pot=7.5, to_call=1.0, preflop_actions="UTG/2.0bb/CO/call/BTN/call")
    assert analyze(GameState(hero_cards="As Ad", **multiway)).action == "raise"
    assert analyze(GameState(hero_cards="7c 2d", **multiway)).action == "fold"
    limp = dict(hero_pos="BB", preflop_line="other", pot=2.0, to_call=0, preflop_actions="SB/call")
    assert analyze(GameState(hero_cards="7c 2d", **limp)).action in ("check", "raise")    # фолда быть не может
    with pytest.raises(ValidationError):
        GameState(hero_cards="As Ad", hero_pos="BB", preflop_line="other", pot=7.5)       # нет записи линии


def test_postflop_advice():
    assert analyze(gs(hero_cards="Ks Kd", board="Kh 7h 2c")).action == "bet"              # сет
    a = analyze(gs(hero_cards="4s 4d", board="Ah Kd Qc 9s 2h", pot=20, to_call=15, stack=70,
                   hero_pos="BB", villain_pos="BTN", preflop_line="villain_open_hero_call"))
    assert a.action == "fold" and a.equity < a.required
    a = analyze(gs(hero_cards="Ah Qh", board="Kh 7h 2c", pot=8.5, to_call=3, stack=94.5,
                   hero_pos="BB", villain_pos="BTN", preflop_line="villain_open_hero_call"))
    assert a.action in ("call", "raise") and a.villain.narrowed          # натсовое флеш-дро не сбрасывают
