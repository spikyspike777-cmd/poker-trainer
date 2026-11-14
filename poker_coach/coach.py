"""Стратег на правилах поверх чисел + текстовый разбор без LLM.

Постфлоп — эвристика (эквити против шансов банка, ауты, позиция), а не солвер.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import calc, charts, ranges
from .solver import RiverCFR, Solver, Spot
from .state import GameState, PreflopLine as L, to_eval7

# Солвер для постфлопа. None — только эвристика. Заменяется любой реализацией solver.base.Solver.
SOLVER: Solver | None = RiverCFR(iterations=300)

ACTION_RU = {"fold": "Фолд", "check": "Чек", "call": "Колл", "bet": "Бет", "raise": "Рейз", "allin": "Олл-ин"}


@dataclass
class Analysis:
    action: str
    amount: float | None = None          # размер бета / рейза «до», в bb
    sizing: str = ""                     # например «75% банка»
    equity: float | None = None
    required: float = 0.0                # эквити, нужное для колла
    spr: float = 0.0
    hand: str = ""
    outs: dict[str, list[str]] = field(default_factory=dict)
    villain: ranges.VillainRange | None = None
    reasons: list[str] = field(default_factory=list)
    source: str = ""                     # откуда совет: чарт, эвристика или солвер
    mix: list[tuple[str, float | None, float, float]] = field(default_factory=list)
    # ^ смешанная стратегия солвера: (действие, размер, частота, EV в bb)

    @property
    def headline(self) -> str:
        s = ACTION_RU[self.action]
        if self.amount is not None and self.action in ("bet", "raise"):
            s += f" {self.amount:g} bb"
            if self.sizing:
                s += f" ({self.sizing})"
        elif self.action == "call" and self.amount:
            s += f" {self.amount:g} bb"
        return s


def analyze(state: GameState) -> Analysis:
    hero, board = to_eval7(state.hero_cards), to_eval7(state.board)
    vr = ranges.villain_range(state)
    a = Analysis(action="fold", villain=vr, spr=calc.spr(state.stack, state.pot),
                 required=calc.pot_odds(state.pot, state.to_call))
    if vr is not None:
        a.equity = calc.equity_vs_range(hero, vr.combos, board)
    if board:
        a.hand = calc.hand_name(hero, board)
        a.outs = calc.outs(hero, board)
        _postflop(state, a)
        if SOLVER is not None and state.hero_invested == 0:
            _solve(state, a, hero, board)
    else:
        _preflop(state, a)
    return a


# ---------- префлоп: чарты ----------

def _preflop(s: GameState, a: Analysis) -> None:
    key = charts.hand_key(s.hero_cards)
    a.source = "префлоп-чарт 6-max, 100bb"
    blind = {"SB": 0.5, "BB": 1.0}.get(s.hero_pos, 0.0)
    if s.preflop_line == L.UNOPENED:
        rng = charts.rfi_range(s.hero_pos)
        pct = charts.range_percent(rng)
        if charts.in_range(s.hero_cards, rng):
            a.action, a.amount = "raise", charts.OPEN_SIZE[s.hero_pos]
            a.reasons.append(f"{key} входит в диапазон открытия с {s.hero_pos} (верхние {pct:.0f}% рук).")
        else:
            a.reasons.append(f"{key} слабее диапазона открытия с {s.hero_pos} (верхние {pct:.0f}% рук).")
        a.reasons.append(_pos_note(s.hero_pos))
        return
    if s.preflop_line == L.VS_OPEN:
        raise_rng, call_rng = charts.vs_open_ranges(s.hero_pos, s.villain_pos)
        open_size = s.to_call + blind
        mult = charts.THREEBET_SIZE_IP if s.in_position else charts.THREEBET_SIZE_OOP
        raise_to, raise_name = round(open_size * mult, 1), "3-бет"
    elif s.preflop_line == L.VS_3BET:
        raise_rng, call_rng = charts.vs_3bet_ranges(s.hero_pos)
        raise_to, raise_name = round((s.to_call + charts.OPEN_SIZE.get(s.hero_pos, 2.5)) * 2.3, 1), "4-бет"
    else:
        raise_rng, call_rng = charts.VS_4BET
        raise_to, raise_name = None, "олл-ин"
    if charts.in_range(s.hero_cards, raise_rng):
        a.action, a.amount = ("allin", None) if raise_to is None or raise_to >= s.stack else ("raise", raise_to)
        a.sizing = raise_name
        a.reasons.append(f"{key} — в диапазоне на {raise_name} против: {a.villain.label}.")
    elif charts.in_range(s.hero_cards, call_rng):
        a.action, a.amount = "call", s.to_call
        a.reasons.append(f"{key} — в диапазоне колла против: {a.villain.label}.")
    else:
        a.reasons.append(f"{key} не входит ни в диапазон колла, ни в диапазон на {raise_name} в этой ситуации.")
    a.reasons.append(f"Ваше эквити против его диапазона (верхние {a.villain.percent:.0f}% рук) — {a.equity:.0%}, "
                     f"для колла по шансам банка нужно {a.required:.0%}.")
    if a.action == "fold" and a.equity > a.required:
        a.reasons.append("Эквити формально хватает, но оно не учитывает, что без позиции и инициативы "
                         "реализовать его на следующих улицах не получится полностью.")
    a.reasons.append("Вы в позиции: будете ходить последним на постфлопе." if s.in_position
                     else "Вы без позиции: на постфлопе придётся ходить первым.")


def _pos_note(pos: str) -> str:
    left = {"UTG": 5, "HJ": 4, "CO": 3, "BTN": 2, "SB": 1}[pos]
    return f"За вами ещё {left} игрок(а/ов): чем их больше, тем сильнее нужна рука для открытия."


# ---------- постфлоп: правила поверх чисел ----------

def _bet(s: GameState, a: Analysis, frac: float) -> None:
    amt = round(s.pot * frac, 1)
    if amt >= s.stack * 0.8:
        a.action, a.amount = "allin", None
    else:
        a.action, a.amount, a.sizing = "bet", amt, f"{frac:.0%} банка"


def _postflop(s: GameState, a: Analysis) -> None:
    a.source = "эвристика: эквити против шансов банка (не солвер)"
    eq, n_outs, river = a.equity, calc.count_outs(a.outs), s.street == "river"
    draw = n_outs >= 8
    a.reasons.append(f"Сейчас у вас {a.hand}. Против диапазона оппонента ({a.villain.label}) "
                     f"ваше эквити — {eq:.0%}.")
    if n_outs:
        parts = ", ".join(f"{len(v)} на {k}" for k, v in a.outs.items())
        mult = 4 if s.street == "flop" else 2
        a.reasons.append(f"Аутов: {n_outs} ({parts}). Правило «{mult}»: ≈{min(n_outs * mult, 100)}% усилиться "
                         f"{'к риверу' if s.street == 'flop' else 'на ривере'}.")
    pos = "в позиции" if s.in_position else "без позиции"
    if s.to_call > 0:
        a.reasons.append(f"Нужно доставить {s.to_call:g} bb в банк {s.pot:g} bb — колл окупается "
                         f"при эквити от {a.required:.0%}.")
        margin = 0.02 if river else (-0.04 if draw and s.stack - s.to_call >= 2 * s.pot else 0.0)
        if eq >= (0.80 if river else 0.72):
            raise_to = round(s.to_call * 3, 1)
            if raise_to >= s.stack * 0.6:
                a.action = "allin"
            else:
                a.action, a.amount, a.sizing = "raise", raise_to, "3× ставки"
            a.reasons.append("Вы далеко впереди его диапазона ставки — рейз для велью: "
                             "худшие руки и дро заплатят.")
        elif eq >= a.required + margin:
            a.action, a.amount = "call", s.to_call
            if eq < a.required:
                a.reasons.append("Эквити чуть ниже нужного, но стеки глубокие: если дро закроется, "
                                 "вы доберёте ставки на следующих улицах (подразумеваемые шансы).")
            else:
                a.reasons.append(f"Эквити {eq:.0%} выше нужных {a.required:.0%} — колл выгоден, "
                                 "но для рейза рука недостаточно сильна.")
        else:
            a.reasons.append(f"Эквити {eq:.0%} ниже нужных {a.required:.0%} — колл в минус.")
    else:
        if eq >= 0.80:
            _bet(s, a, 0.75)
            a.reasons.append("Очень сильная рука против его диапазона — крупный бет для велью.")
        elif eq >= 0.65:
            _bet(s, a, 0.5)
            a.reasons.append("Вы впереди большей части его диапазона — бет для велью и защиты от дро.")
        elif eq >= 0.5 and not river and s.in_position:
            _bet(s, a, 0.33)
            a.reasons.append("Средняя по силе рука в позиции — небольшой бет: добрать с худших рук "
                             "и не раздувать банк.")
        elif draw and not river and eq >= 0.3:
            _bet(s, a, 0.5)
            a.reasons.append("Сильное дро — полублеф: можно выиграть сразу, а если заколлируют, "
                             "остаются ауты.")
        else:
            a.action = "check"
            a.reasons.append("Рука со средней или слабой силой — чек: ставка выбьет худшие руки, "
                             "а заколлируют только лучшие." if eq >= 0.35 else
                             "Рука слабая против его диапазона — чек, не вкладываем фишки.")
    a.reasons.append(f"Вы {pos}; SPR (стек к банку) = {a.spr:.1f}"
                     + (" — стеки мелкие, любая крупная ставка ведёт к олл-ину." if a.spr < 3 else "."))


def _mix_label(kind: str, amount: float | None, pot: float) -> str:
    if kind in ("bet", "raise") and amount:
        tail = f" ({amount / pot:.0%} банка)" if kind == "bet" else ""
        return f"{ACTION_RU[kind].lower()} {amount:g} bb{tail}"
    return ACTION_RU[kind].lower()


def _solve(s: GameState, a: Analysis, hero, board) -> None:
    """Если солвер умеет этот спот — заменить эвристический совет равновесной стратегией."""
    pot0 = s.pot - s.to_call                         # банк до ставки оппонента
    spot = Spot(hero=tuple(hero), board=board, hero_range=ranges.hero_range(s),
                villain_range=ranges.villain_start(s), pot=pot0, stack=s.stack,
                hero_in_position=s.in_position, facing_bet=s.to_call)
    if not SOLVER.supports(spot) or not spot.villain_range:
        return
    res = SOLVER.solve(spot)
    acts = [x for x in res.actions if x.freq >= 0.05] or res.actions
    total = sum(x.freq for x in acts)
    acts.sort(key=lambda x: -x.freq)
    a.mix = [(x.kind, x.amount if x.kind in ("bet", "raise", "call") else None, x.freq / total, round(x.ev, 2) + 0.0) for x in acts]
    kind, amount, freq, _ev = a.mix[0]
    a.action, a.amount = kind, amount
    a.sizing = f"{amount / pot0:.0%} банка" if kind == "bet" else ("3× ставки" if kind == "raise" else "")
    a.equity = res.equity
    a.villain.combos, a.villain.weights, a.villain.narrowed = res.villain_combos, res.villain_weights, True
    a.villain.label = a.villain.label.split(",")[0] + "; частоты рук — из решения солвера"
    eq = a.equity
    r = [f"Сейчас у вас {a.hand}. Против рук, с которыми оппонент доходит до этой точки, ваше эквити — {eq:.0%}."]
    if s.to_call > 0:
        r.append(f"Нужно доставить {s.to_call:g} bb в банк {s.pot:g} bb — по шансам банка нужно {a.required:.0%}.")
    parts = "; ".join(f"{_mix_label(k, amt, pot0)} — {f:.0%} (EV {ev + 0.0:+.1f} bb)" for k, amt, f, ev in a.mix)
    r.append(f"Равновесная стратегия с этой рукой: {parts}.")
    if len(a.mix) > 1 and a.mix[1][2] >= 0.25:
        r.append("Солвер смешивает действия: их ожидание почти одинаково, поэтому ошибкой не будет ни одно из них.")
    if kind in ("bet", "raise", "allin"):
        r.append("Это ставка для велью: вас заколлирует достаточно худших рук." if eq >= 0.5 else
                 "Это блеф: на вскрытии рука почти не выигрывает, а ставка заставит сбросить часть более сильных рук.")
    elif kind == "call":
        r.append("Колл: эквити выше, чем требуют шансы банка." if eq >= a.required else
                 "Колл как блеф-кэтчер: рука бьёт только блефы, но их в его диапазоне ставки достаточно.")
    elif kind == "check":
        r.append("Чек: худшие руки на ставку сбросят, а лучшие заколлируют — ставить невыгодно.")
    else:
        r.append("Фолд: против его диапазона ставки эта рука выигрывает слишком редко.")
    r.append(f"Вы {'в позиции' if s.in_position else 'без позиции'}; SPR (стек к банку) = {a.spr:.1f}.")
    a.reasons = r
    a.source = f"{res.name}; диапазоны на входе — по префлоп-чартам"


def explain(a: Analysis) -> str:
    lines = [f"Совет: {a.headline}", ""] + [f"• {r}" for r in a.reasons] + ["", f"Источник совета: {a.source}."]
    return "\n".join(lines)
