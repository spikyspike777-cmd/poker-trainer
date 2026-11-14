"""Характеры ботов. Основа у всех одна — совет тренера; стиль его искажает."""
from __future__ import annotations

import random
from dataclasses import dataclass

from . import charts

_PREMIUM = "TT+, AQs+, AKo"


@dataclass(frozen=True)
class Style:
    key: str
    name: str
    about: str
    pre_drop: float = 0.0      # тайтовость: шанс сбросить не премиум-руку, которую чарт играет
    pre_add: float = 0.0       # лузовость: шанс сыграть играбельную руку, которую чарт сбрасывает
    pre_reraise: float = 0.0   # шанс превратить колл на префлопе в рейз
    bluff_bet: float = 0.0     # шанс поставить вместо чека
    bluff_raise: float = 0.0   # шанс рейзить вместо фолда на ставку
    hero_call: float = 0.0     # шанс заколлировать вместо фолда на ставку
    scared_fold: float = 0.0   # шанс сбросить руку, с которой совет — колл
    slow: float = 0.0          # шанс чекнуть вместо ставки (кроме очень сильных рук)


STYLES = {s.key: s for s in [
    Style("standard", "Стандартный", "Играет по чартам и правилам тренера."),
    Style("nit", "Тайтовый", "Играет мало рук, ставит только с силой, легко сбрасывает.",
          pre_drop=0.55, scared_fold=0.5, slow=0.45),
    Style("station", "Телефон", "Входит в игру слишком часто, почти не сбрасывает и редко ставит сам.",
          pre_add=0.6, hero_call=0.75, slow=0.5),
    Style("lag", "Агрессор", "Играет много рук, много ставит и блефует.",
          pre_add=0.4, pre_reraise=0.35, bluff_bet=0.5, bluff_raise=0.2),
]}


def playable(cards: list[str]) -> bool:
    """Рука, которую лузовый игрок не хочет сбрасывать: пара, туз, одномастные, связки, две картинки."""
    order = "23456789TJQKA"
    a, b = sorted((order.index(c[0]) for c in cards), reverse=True)
    suited = cards[0][1] == cards[1][1]
    return a == b or a == 12 or suited or a - b <= 2 or b >= 7


def restyle(style: Style, rng: random.Random, *, preflop: bool, kind: str, to: float, cards: list[str],
            equity: float | None, pot: float, cur_bet: float, to_call: float, raises: int) -> tuple[str, float]:
    """Исказить базовое решение (kind, to) под характер. kind: fold/check/call/raise."""
    r = rng.random
    if preflop:
        if raises == 0:
            return kind, to                    # открывается бот всегда: иначе герой остаётся без решения
        if kind in ("call", "raise") and r() < style.pre_drop and not charts.in_range(cards, _PREMIUM):
            return "fold", 0.0
        if kind == "fold" and to_call > 0 and playable(cards) and r() < style.pre_add:
            kind = "call"
        if kind == "call" and raises <= 2 and r() < style.pre_reraise:
            return "raise", round(cur_bet * (3.5 if raises == 1 else 2.3), 1)
        return kind, to
    eq = equity if equity is not None else 0.5
    if kind == "raise" and eq < 0.85 and r() < style.slow:
        return ("call", 0.0) if to_call > 0 else ("check", 0.0)
    if kind == "check" and r() < style.bluff_bet:
        return "raise", round(pot * 0.66, 1)
    if kind == "fold" and to_call > 0:
        if r() < style.hero_call:
            return "call", 0.0
        if r() < style.bluff_raise:
            return "raise", round(cur_bet * 3, 1)
    if kind == "call" and eq < 0.6 and r() < style.scared_fold:
        return "fold", 0.0
    return kind, to
