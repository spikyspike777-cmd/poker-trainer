"""Единый интерфейс солвера. Любая реализация (свой CFR, обёртка над внешним
солвером) принимает Spot и возвращает SolverResult — стратегу всё равно, кто считал."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import eval7

Combo = tuple[eval7.Card, eval7.Card]


@dataclass
class Spot:
    """Решение героя на текущей улице. Суммы в bb.

    pot — банк на начало улицы (без ставок этой улицы); stack — эффективный стек на начало улицы;
    facing_bet — ставка оппонента, на которую отвечает герой (0, если ставки нет).
    Диапазоны — на начало улицы, до действий на ней.
    """
    hero: Combo
    board: list[eval7.Card]
    hero_range: list[Combo]
    villain_range: list[Combo]
    pot: float
    stack: float
    hero_in_position: bool
    facing_bet: float = 0.0
    oop_can_lead: bool = True      # может ли игрок без позиции поставить первым (False — «чек в агрессора»)
    hero_weights: list[float] | None = None      # веса рук в диапазонах (None — все равны)
    villain_weights: list[float] | None = None


@dataclass
class ActionFreq:
    kind: str            # fold / check / call / bet / raise / allin
    amount: float        # сколько всего вложено на улице после действия
    freq: float          # как часто равновесная стратегия выбирает действие с рукой героя
    ev: float            # ожидание действия в bb относительно начала улицы


@dataclass
class SolverResult:
    actions: list[ActionFreq]
    equity: float                                  # эквити героя против диапазона оппонента в этой точке
    villain_combos: list[Combo] = field(default_factory=list)
    villain_weights: list[float] = field(default_factory=list)   # с какой частотой оппонент доходит сюда
    name: str = ""


class Solver(Protocol):
    def supports(self, spot: Spot) -> bool: ...

    def solve(self, spot: Spot) -> SolverResult: ...
