"""Оценка диапазона оппонента: стартовый — из чартов, сужение — эвристикой."""
from __future__ import annotations

from dataclasses import dataclass

import eval7

from . import calc, charts, preflop_gto, range_data
from .params import P
from .state import GameState, PreflopLine as L, to_eval7

_SB_CALL_FALLBACK = "JJ-22, AQs-A8s, KTs+, QTs+, JTs, T9s, 98s, AQo-AJo, KQo"


@dataclass
class VillainRange:
    combos: list[calc.Combo]
    label: str
    percent: float           # доля от всех стартовых рук до учёта борда
    narrowed: bool = False
    weights: list[float] | None = None   # вес каждой комбинации (от солвера); None — все равны




def start_range(line: L, me: str, opp: str | None):
    """Префлоп-диапазон игрока `me`, если с ЕГО точки зрения раздача шла по линии `line` против `opp`.

    -> (комбинации, описание) или None, если диапазон для такой линии не определён.
    """
    if opp is None:
        return None
    # (линия для таблицы, действие, строка ручного чарта, описание)
    if line in (L.HERO_OPEN_VILLAIN_CALL, L.VS_3BET):
        spec = ("", "raise", charts.RFI.get(me, charts.RFI["BTN"]), f"открытие рейзом с {me}")
    elif line == L.VILLAIN_OPEN_HERO_CALL:
        spec = (f"{opp}/r", "call", charts.vs_open_ranges(me, opp)[1] or _SB_CALL_FALLBACK,
                f"колл с {me} против опена с {opp}")
    elif line in (L.HERO_3BET_VILLAIN_CALL, L.VS_4BET):
        spec = (f"{opp}/r", "raise", charts.vs_open_ranges(me, opp)[0], f"3-бет с {me} против опена с {opp}")
    elif line == L.VILLAIN_3BET_HERO_CALL:
        spec = (f"{me}/r/{opp}/r", "call", charts.vs_3bet_ranges(me)[1], f"колл 3-бета с {me}")
    else:
        return None
    abs_line, kind, chart, label = spec
    keys = preflop_gto.range_keys(abs_line, me, kind) if P["table_ranges"] else None
    if keys:
        return list(charts.keys_to_combos(keys)), label
    return charts.combos(chart), label


def start_weighted(line: L, me: str, opp: str | None):
    """Как start_range, но с весами рук: -> (комбинации, веса, описание) или None.

    Если включены диапазоны из данных и для линии они есть, веса берутся оттуда; иначе все веса равны 1.
    """
    got = start_range(line, me, opp)
    if got is None:
        return None
    if P["data_ranges"]:
        data = range_data.lookup(line.value, me, opp)
        if data:
            return list(data[0]), list(data[1]), got[1]
    return list(got[0]), [1.0] * len(got[0]), got[1]


def villain_start_weighted(state: GameState):
    """Стартовый диапазон оппонента с весами (с поправкой на профиль, если она есть)."""
    h, v, line = state.hero_pos, state.villain_pos, state.preflop_line
    if line in (L.UNOPENED, L.OTHER, L.VS_4BET) or v is None:
        got = _start(state)
        return (list(got[0]), [1.0] * len(got[0]), got[1]) if got else None
    got = start_weighted(_MIRROR[line], v, h)
    scale = _scales(state)[0]
    if got and abs(scale - 1) >= 0.05:                 # профиль меняет ширину — веса из данных уже не подходят
        scaled = charts.scale_range(got[0], scale)
        return scaled, [1.0] * len(scaled), got[2]
    return got


_MIRROR = {
    L.VS_OPEN: L.HERO_OPEN_VILLAIN_CALL,               # оппонент открылся — с его стороны «я открылся»
    L.VILLAIN_OPEN_HERO_CALL: L.HERO_OPEN_VILLAIN_CALL,
    L.VS_3BET: L.HERO_3BET_VILLAIN_CALL,               # оппонент сделал 3-бет
    L.VILLAIN_3BET_HERO_CALL: L.HERO_3BET_VILLAIN_CALL,
    L.HERO_OPEN_VILLAIN_CALL: L.VILLAIN_OPEN_HERO_CALL,
    L.HERO_3BET_VILLAIN_CALL: L.VILLAIN_3BET_HERO_CALL,
}


def _start(state: GameState):
    """Стартовый диапазон оппонента: (комбинации, описание) или None."""
    h, v, line = state.hero_pos, state.villain_pos, state.preflop_line
    if line in (L.UNOPENED, L.OTHER) or v is None:
        return None
    if line == L.VS_4BET:                              # оппонент открылся и сделал 4-бет
        keys = preflop_gto.range_keys(f"{v}/r/{h}/r", v, "raise") if P["table_ranges"] else None
        combos = list(charts.keys_to_combos(keys)) if keys else charts.combos(charts.vs_3bet_ranges(v)[0])
        return combos, f"4-бет с {v}"
    return start_range(_MIRROR[line], v, h)


_STREET_CARDS = {"flop": 3, "turn": 4, "river": 5}
STREETS_ORDER = ["flop", "turn", "river"]
def narrow(combos, board, kind: str, in_position: bool, bluff_scale: float = 1.0):
    """Сузить диапазон после одного действия игрока на борде `board` (эвристика)."""
    if len(combos) < 4:
        return combos
    ranked = sorted(combos, key=lambda c: eval7.evaluate(list(c) + board), reverse=True)
    if kind == "check":
        return ranked[int(len(ranked) * P["check_back_drop"]):] if in_position else ranked
    keep, bluffs = P[f"{kind}_keep"], P.get(f"{kind}_bluffs", 0.0)
    n = max(1, int(len(ranked) * keep))
    kept, rest = ranked[:n], ranked[n:]
    k = int(len(ranked) * min(0.35, bluffs * bluff_scale))
    if k:
        kept, rest = kept + rest[-k:], rest[:-k]
    if len(board) < 5:
        need = 4 if kind == "call" else 8          # колл — любые дро, ставка — сильные дро
        kept += [c for c in rest if calc.count_outs(calc.outs(list(c), board)) >= need]
    return kept


def _scales(state: GameState) -> tuple[float, float]:
    adj = state.villain_adj
    return (adj.range_scale, adj.bluff_scale) if adj else (1.0, 1.0)


def _through_history(combos, state: GameState, who: str, in_position: bool, include_current: bool):
    board = to_eval7(state.board)
    bluff_scale = _scales(state)[1] if who == "villain" else 1.0
    for street, actor, kind in state.history:
        if actor != who or (street == state.street and not include_current):
            continue
        combos = narrow(combos, board[:_STREET_CARDS[street]], kind, in_position, bluff_scale)
    return combos


def hero_range(state: GameState) -> list[calc.Combo]:
    """Диапазон героя на начало текущей улицы (нужен солверу): чарты + сужение по его действиям."""
    combos, _ = start_range(state.preflop_line, state.hero_pos, state.villain_pos)
    combos = calc.live_combos(combos, to_eval7(state.board))
    return _through_history(combos, state, "hero", state.in_position, include_current=False)


def villain_start(state: GameState) -> list[calc.Combo]:
    """Диапазон оппонента на начало текущей улицы: префлоп-линия + действия на прошлых улицах."""
    base, _ = _start(state)
    combos = calc.live_combos(charts.scale_range(base, _scales(state)[0]),
                              to_eval7(state.hero_cards) + to_eval7(state.board))
    return _through_history(combos, state, "villain", not state.in_position, include_current=False)


def extra_ranges(state: GameState) -> list[list[calc.Combo]]:
    """Диапазоны остальных оппонентов в мультивей-банке, суженные по их действиям."""
    dead = to_eval7(state.hero_cards) + to_eval7(state.board)
    board = to_eval7(state.board)
    out = []
    for i, (pos, line) in enumerate(state.extra_villains, start=2):
        got = start_range(_MIRROR.get(line, L.VILLAIN_OPEN_HERO_CALL), pos, state.hero_pos)
        combos = calc.live_combos(got[0] if got else charts.combos(charts.RFI["BTN"]), dead)
        for street, actor, kind in state.history:
            if actor == f"v{i}":
                combos = narrow(combos, board[:_STREET_CARDS[street]], kind, False)
        out.append(combos)
    return out


def villain_range(state: GameState) -> VillainRange | None:
    start = _start(state)
    if start is None:
        return None
    base, label = start
    hero, board = to_eval7(state.hero_cards), to_eval7(state.board)
    scale, bluff_scale = _scales(state)
    scaled = charts.scale_range(base, scale)
    combos = calc.live_combos(scaled, hero + board)
    vr = VillainRange(combos, label, 100 * len(scaled) / 1326)
    if abs(scale - 1) >= 0.05:
        vr.label += f", с поправкой на профиль (×{scale:g})"
    if not board:
        return vr
    if state.history:
        vr.combos = _through_history(combos, state, "villain", not state.in_position, include_current=True)
        if len(vr.combos) < len(combos):
            vr.label += ", сужен по его действиям на постфлопе"
            vr.narrowed = True
    elif state.to_call > 0:
        vr.combos = narrow(combos, board, "bet", not state.in_position, bluff_scale)
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


def solved_ranges(state: GameState, solver):
    """Диапазоны на начало текущей улицы из решения прошлых улиц солвером, а не из правил сужения.

    Прошлые улицы решаются по очереди (флоп, затем тёрн); после каждой берутся веса рук, с которыми
    равновесная стратегия играет именно ту линию, что была сыграна. -> (руки героя, веса, руки оппонента, веса)
    или None, если так посчитать нельзя (нет сумм ставок, мультивей, линии нет в дереве солвера).
    """
    hist, amts = state.history, state.history_amounts
    past = [s for s in ("flop", "turn") if s != state.street and STREETS_ORDER.index(s) < STREETS_ORDER.index(state.street)]
    if not past or len(amts) != len(hist) or state.extra_villains or not hasattr(solver, "reach_after"):
        return None
    start_h = start_weighted(state.preflop_line, state.hero_pos, state.villain_pos)
    start_v = villain_start_weighted(state)
    if start_h is None or start_v is None:
        return None
    # сколько каждый вложил на каждой улице
    inv = {s: {"hero": 0.0, "villain": 0.0} for s in STREETS_ORDER}
    for (street, actor, kind), amount in zip(hist, amts):
        if actor not in ("hero", "villain"):
            return None
        if kind in ("bet", "raise"):
            inv[street][actor] = amount
        elif kind == "call":
            inv[street][actor] = max(inv[street].values())
    spent = {a: sum(inv[s][a] for s in past) for a in ("hero", "villain")}
    cur_hero, cur_vil = state.hero_invested, state.hero_invested + state.to_call
    pot = state.pot - spent["hero"] - spent["villain"] - cur_hero - cur_vil
    stack = state.stack + spent["hero"] + cur_hero
    if pot <= 0:
        return None
    board = to_eval7(state.board)
    hero_ip = state.in_position
    combos = {"hero": list(start_h[0]), "villain": list(start_v[0])}
    weights = {"hero": list(start_h[1]), "villain": list(start_v[1])}
    oop, ip = ("villain", "hero") if hero_ip else ("hero", "villain")
    for street in past:
        b = board[:_STREET_CARDS[street]]
        for a in combos:                                   # убираем руки, которые бьются с бордом
            keep = [i for i, c in enumerate(combos[a]) if c[0] not in b and c[1] not in b and weights[a][i] > 1e-4]
            combos[a], weights[a] = [combos[a][i] for i in keep], [weights[a][i] for i in keep]
        acts = [(actor, kind, amount) for (st, actor, kind), amount in zip(hist, amts) if st == street]
        if not acts or min(len(combos["hero"]), len(combos["villain"])) < 2:
            return None
        path = [(1, 0.0) if k == "check" else (2, 0.0) if k == "call" else (3, amount) for _a, k, amount in acts]
        lead = acts[0][1] in ("bet", "raise") or acts[0][0] != oop
        try:
            r_oop, r_ip = solver.reach_after(b, combos[oop], combos[ip], weights[oop], weights[ip], pot, stack, path, lead)
        except ValueError:
            return None                                    # сыгранной линии нет в дереве солвера
        for a, r in ((oop, r_oop), (ip, r_ip)):
            top = max(r, default=0.0)
            if top < 1e-6:
                return None                                # солвер так не играет никогда — диапазон не определён
            weights[a] = [x / top for x in r]
        c = max(inv[street].values())
        pot, stack = pot + 2 * c, stack - c
    dead = {"hero": set(board), "villain": set(board) | set(to_eval7(state.hero_cards))}
    out = []
    for a in ("hero", "villain"):
        keep = [i for i, c in enumerate(combos[a]) if c[0] not in dead[a] and c[1] not in dead[a] and weights[a][i] > 1e-3]
        if len(keep) < 2:
            return None
        out += [[combos[a][i] for i in keep], [weights[a][i] for i in keep]]
    return tuple(out)


def flop_ranges(state: GameState):
    """Взвешенные стартовые диапазоны для решения на флопе: (руки героя, веса, руки оппонента, веса) или None."""
    if state.extra_villains or not P["data_ranges"]:
        return None
    sh = start_weighted(state.preflop_line, state.hero_pos, state.villain_pos)
    sv = villain_start_weighted(state)
    if sh is None or sv is None:
        return None
    board = set(to_eval7(state.board))
    dead = {0: board, 1: board | set(to_eval7(state.hero_cards))}
    out = []
    for i, (combos, weights, _label) in enumerate((sh, sv)):
        keep = [k for k, c in enumerate(combos) if c[0] not in dead[i] and c[1] not in dead[i]]
        if len(keep) < 2:
            return None
        out += [[combos[k] for k in keep], [weights[k] for k in keep]]
    return tuple(out)
