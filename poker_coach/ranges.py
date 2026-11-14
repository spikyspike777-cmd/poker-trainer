"""Оценка диапазона оппонента: стартовый — из чартов, сужение — эвристикой."""
from __future__ import annotations

from dataclasses import dataclass

import eval7

from . import calc, charts
from .state import GameState, PreflopLine as L, to_eval7

_SB_CALL_FALLBACK = "JJ-22, AQs-A8s, KTs+, QTs+, JTs, T9s, 98s, AQo-AJo, KQo"


@dataclass
class VillainRange:
    combos: list[calc.Combo]
    label: str
    percent: float           # доля от всех стартовых рук до учёта борда
    narrowed: bool = False


def _start(state: GameState) -> tuple[str, str] | None:
    h, v, line = state.hero_pos, state.villain_pos, state.preflop_line
    if line == L.UNOPENED:
        return None
    if line in (L.VS_OPEN, L.VILLAIN_OPEN_HERO_CALL):
        return charts.RFI.get(v, charts.RFI["BTN"]), f"открытие рейзом с {v}"
    if line in (L.VS_3BET, L.VILLAIN_3BET_HERO_CALL):
        return charts.vs_open_ranges(v, h)[0], f"3-бет с {v} против опена с {h}"
    if line == L.VS_4BET:
        return charts.vs_3bet_ranges(v)[0], f"4-бет с {v}"
    if line == L.HERO_OPEN_VILLAIN_CALL:
        return charts.vs_open_ranges(v, h)[1] or _SB_CALL_FALLBACK, f"колл с {v} против опена с {h}"
    return charts.vs_3bet_ranges(v)[1], f"колл 3-бета с {v}"


def villain_range(state: GameState) -> VillainRange | None:
    start = _start(state)
    if start is None:
        return None
    rng_str, label = start
    hero, board = to_eval7(state.hero_cards), to_eval7(state.board)
    combos = calc.live_combos(charts.combos(rng_str), hero + board)
    vr = VillainRange(combos, label, charts.range_percent(rng_str))
    if board and state.to_call > 0:
        vr.combos = _narrow_after_bet(combos, board)
        vr.label += ", после ставки оставлены сильные руки, дро и часть блефов"
        vr.narrowed = True
    return vr


def _narrow_after_bet(combos, board, keep: float = 0.5, bluffs: float = 0.12):
    """Оппонент поставил: верхняя половина готовых рук, сильные дро (8+ аутов) и немного блефов.

    Блефы — самые слабые руки диапазона (с ними нет смысла чекать до вскрытия).
    """
    ranked = sorted(combos, key=lambda c: eval7.evaluate(list(c) + board), reverse=True)
    n = max(1, int(len(ranked) * keep))
    kept, rest = ranked[:n], ranked[n:]
    k = int(len(ranked) * bluffs)
    if k:
        kept, rest = kept + rest[-k:], rest[:-k]
    if len(board) < 5:
        kept += [c for c in rest if calc.count_outs(calc.outs(list(c), board)) >= 8]
    return kept


def matrix(vr: VillainRange | None) -> dict[str, float]:
    """Для сетки 13×13: 'AKs' -> доля комбинаций этой руки в диапазоне (0..1)."""
    if vr is None:
        return {}
    cnt: dict[str, int] = {}
    for a, b in vr.combos:
        k = charts.hand_key([str(a), str(b)])
        cnt[k] = cnt.get(k, 0) + 1
    full = lambda k: 6 if len(k) == 2 else (4 if k[2] == "s" else 12)
    return {k: min(1.0, n / full(k)) for k, n in cnt.items()}
