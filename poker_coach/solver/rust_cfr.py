"""Обёртка над солвером на Rust (каталог solver_rs): тёрн и ривер один на один.

Сборка: cd solver_rs && maturin build --release && pip install target/wheels/solver_rs-*.whl
Если расширение не собрано, тренер считает ривер солвером на Python, а тёрн — эвристикой.
"""
from __future__ import annotations

import eval7

from .base import ActionFreq, SolverResult, Spot

try:
    import solver_rs
except ImportError:                      # расширение не собрано
    solver_rs = None

_KINDS = {0: "fold", 1: "check", 2: "call", 3: "bet", 4: "raise", 5: "allin"}


def _idx(card: eval7.Card) -> int:
    return card.rank * 4 + card.suit


class RustCFR:
    name = "свой солвер на Rust (CFR+)"

    def __init__(self, bet_sizes=(0.33, 0.75), turn_sizes=(0.66,), river_sizes=(0.75,), raise_mult: float = 3.0,
                 iterations: int = 300, turn_iterations: int = 120):
        # на тёрне дерево проще (один размер ставки, на ривере без рейзов): иначе считать слишком долго
        self.bet_sizes, self.turn_sizes, self.river_sizes = bet_sizes, turn_sizes, river_sizes
        self.raise_mult = raise_mult
        self.iterations, self.turn_iterations = iterations, turn_iterations
        self._cache: dict = {}

    @staticmethod
    def available() -> bool:
        return solver_rs is not None

    def supports(self, spot: Spot) -> bool:
        return (solver_rs is not None and len(spot.board) in (4, 5) and spot.stack > 0
                and spot.facing_bet <= spot.stack and len(spot.villain_range) >= 2)

    def solve(self, spot: Spot) -> SolverResult:
        hero_p = 1 if spot.hero_in_position else 0
        hero_range = list(spot.hero_range)
        if not any(set(c) == set(spot.hero) for c in hero_range):
            hero_range.append(spot.hero)
        ranges = [None, None]
        ranges[hero_p], ranges[1 - hero_p] = hero_range, list(spot.villain_range)
        hero_i = next(i for i, c in enumerate(ranges[hero_p]) if set(c) == set(spot.hero))
        board = list(spot.board)
        river = len(board) == 5
        if river:
            cards, boards = [255], [board]
        else:
            deck = [c for c in eval7.Deck().cards if c not in board]
            cards, boards = [_idx(c) for c in deck], [board + [c] for c in deck]
        strengths = [[[0 if (b[-1] in h and not river) else eval7.evaluate(list(h) + b) for h in r] for b in boards]
                     for r in ranges]
        path = []
        if hero_p == 1 and spot.facing_bet == 0:
            path = [(1, 0.0)]
        elif hero_p == 1:
            path = [(3, spot.facing_bet)]
        elif spot.facing_bet > 0:
            path = [(1, 0.0), (3, spot.facing_bet)]
        key = (tuple(map(str, board)), tuple(map(str, spot.hero)), len(ranges[0]), len(ranges[1]),
               hash(tuple(sorted(str(a) + str(b) for a, b in ranges[0]))),
               hash(tuple(sorted(str(a) + str(b) for a, b in ranges[1]))),
               spot.pot, spot.stack, spot.facing_bet, spot.oop_can_lead, hero_p)
        if key not in self._cache:
            self._cache[key] = solver_rs.solve(
                [(_idx(a), _idx(b)) for a, b in ranges[0]], [(_idx(a), _idx(b)) for a, b in ranges[1]],
                cards, strengths[0], strengths[1], float(spot.pot), float(spot.stack),
                list(self.bet_sizes if river else self.turn_sizes), list(self.river_sizes), self.raise_mult,
                False, float(spot.facing_bet),
                spot.oop_can_lead, self.iterations if river else self.turn_iterations, hero_p, hero_i, path)
        acts, equity, weights = self._cache[key]
        return SolverResult([ActionFreq(_KINDS[k], amt, f, ev) for k, amt, f, ev in acts], equity,
                            ranges[1 - hero_p], list(weights), self.name)
