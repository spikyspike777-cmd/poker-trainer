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
    weights: list[float] | None = None   # вес каждой комбинации (от солвера); None — все равны


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


_HERO_LINE_RANGE = {
    L.HERO_OPEN_VILLAIN_CALL: lambda h, v: charts.RFI.get(h, charts.RFI["BTN"]),
    L.VILLAIN_OPEN_HERO_CALL: lambda h, v: charts.vs_open_ranges(h, v)[1] or _SB_CALL_FALLBACK,
    L.HERO_3BET_VILLAIN_CALL: lambda h, v: charts.vs_open_ranges(h, v)[0],
    L.VILLAIN_3BET_HERO_CALL: lambda h, v: charts.vs_3bet_ranges(h)[1],
}
_STREET_CARDS = {"flop": 3, "turn": 4, "river": 5}
# Какую часть диапазона (сверху по силе) игрок оставляет при действии и сколько добавляет блефов снизу.
NARROW = {"bet": (0.50, 0.12), "raise": (0.30, 0.08), "call": (0.60, 0.0)}
CHECK_BACK_DROP = 0.25      # чек в позиции: убираем верх диапазона (сильные руки обычно ставят)


def narrow(combos, board, kind: str, in_position: bool):
    """Сузить диапазон после одного действия игрока на борде `board` (эвристика)."""
    if len(combos) < 4:
        return combos
    ranked = sorted(combos, key=lambda c: eval7.evaluate(list(c) + board), reverse=True)
    if kind == "check":
        return ranked[int(len(ranked) * CHECK_BACK_DROP):] if in_position else ranked
    keep, bluffs = NARROW[kind]
    n = max(1, int(len(ranked) * keep))
    kept, rest = ranked[:n], ranked[n:]
    k = int(len(ranked) * bluffs)
    if k:
        kept, rest = kept + rest[-k:], rest[:-k]
    if len(board) < 5:
        need = 4 if kind == "call" else 8          # колл — любые дро, ставка — сильные дро
        kept += [c for c in rest if calc.count_outs(calc.outs(list(c), board)) >= need]
    return kept


def _through_history(combos, state: GameState, who: str, in_position: bool, include_current: bool):
    board = to_eval7(state.board)
    for street, actor, kind in state.history:
        if actor != who or (street == state.street and not include_current):
            continue
        combos = narrow(combos, board[:_STREET_CARDS[street]], kind, in_position)
    return combos


def hero_range(state: GameState) -> list[calc.Combo]:
    """Диапазон героя на начало текущей улицы (нужен солверу): чарты + сужение по его действиям."""
    rng = _HERO_LINE_RANGE[state.preflop_line](state.hero_pos, state.villain_pos)
    combos = calc.live_combos(charts.combos(rng), to_eval7(state.board))
    return _through_history(combos, state, "hero", state.in_position, include_current=False)


def villain_start(state: GameState) -> list[calc.Combo]:
    """Диапазон оппонента на начало текущей улицы: префлоп-линия + действия на прошлых улицах."""
    rng_str, _ = _start(state)
    combos = calc.live_combos(charts.combos(rng_str), to_eval7(state.hero_cards) + to_eval7(state.board))
    return _through_history(combos, state, "villain", not state.in_position, include_current=False)


def villain_range(state: GameState) -> VillainRange | None:
    start = _start(state)
    if start is None:
        return None
    rng_str, label = start
    hero, board = to_eval7(state.hero_cards), to_eval7(state.board)
    combos = calc.live_combos(charts.combos(rng_str), hero + board)
    vr = VillainRange(combos, label, charts.range_percent(rng_str))
    if not board:
        return vr
    if state.history:
        vr.combos = _through_history(combos, state, "villain", not state.in_position, include_current=True)
        if len(vr.combos) < len(combos):
            vr.label += ", сужен по его действиям на постфлопе"
            vr.narrowed = True
    elif state.to_call > 0:
        vr.combos = narrow(combos, board, "bet", not state.in_position)
        vr.label += ", после ставки оставлены сильные руки, дро и часть блефов"
        vr.narrowed = True
    return vr


def matrix(vr: VillainRange | None) -> dict[str, float]:
    """Для сетки 13×13: 'AKs' -> доля комбинаций этой руки в диапазоне (0..1)."""
    if vr is None:
        return {}
    cnt: dict[str, float] = {}
    ws = vr.weights or [1.0] * len(vr.combos)
    top = max(ws, default=1.0) or 1.0
    for (a, b), w in zip(vr.combos, ws):
        k = charts.hand_key([str(a), str(b)])
        cnt[k] = cnt.get(k, 0.0) + w / top
    full = lambda k: 6 if len(k) == 2 else (4 if k[2] == "s" else 12)
    return {k: min(1.0, n / full(k)) for k, n in cnt.items() if n / full(k) > 0.02}
