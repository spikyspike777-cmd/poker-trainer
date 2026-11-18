"""Обёртка над солвером на Rust (каталог solver_rs): флоп, тёрн и ривер один на один.

Сборка: cd solver_rs && maturin build --release && pip install target/wheels/solver_rs-*.whl
Если расширение не собрано, тренер считает ривер солвером на Python, а остальное — эвристикой.
"""
from __future__ import annotations

import random

import eval7

from .base import ActionFreq, SolverResult, Spot

try:
    import solver_rs
except ImportError:                      # расширение не собрано
    solver_rs = None

_KINDS = {0: "fold", 1: "check", 2: "call", 3: "bet", 4: "raise", 5: "allin"}
CHECK, CALL, BET = (1, 0.0), (2, 0.0), 3


def _idx(card: eval7.Card) -> int:
    return card.rank * 4 + card.suit


def _spread(cards, k: int, offset: int):
    """k карт, равномерно покрывающих ранги и масти: колода по рангам режется на k частей, из каждой — одна карта."""
    cards = sorted(cards, key=lambda c: (c.rank, c.suit))
    out = []
    for i in range(k):
        chunk = cards[i * len(cards) // k:(i + 1) * len(cards) // k]
        want = (i + offset) % 4
        out.append(next((c for c in chunk if c.suit == want), chunk[(i + offset) % len(chunk)]))
    return out


class RustCFR:
    name = "свой солвер на Rust (CFR+)"

    def __init__(self, bet_sizes=(0.33, 0.75), turn_sizes=(0.33, 0.75), turn_river_sizes=(0.75,),
                 turn_river_raise: bool = False, flop_sizes=(0.5,), flop_turn_sizes=(0.66,),
                 flop_river_sizes=(0.75,), raise_mult: float = 3.0, iterations: int = 300,
                 turn_iterations: int = 120, flop_iterations: int = 80, flop_cards: int = 10,
                 solve_flop: bool = True, max_raises: int = 1, flop_sampling: str = "random",
                 turn_river_raises: int | None = None, value_net=None):
        # чем раньше улица, тем проще дерево: иначе считать слишком долго
        self.bet_sizes = bet_sizes                       # ривер
        self.turn_sizes, self.turn_river_sizes, self.turn_river_raise = turn_sizes, turn_river_sizes, turn_river_raise
        self.flop_sizes, self.flop_turn_sizes, self.flop_river_sizes = flop_sizes, flop_turn_sizes, flop_river_sizes
        self.raise_mult = raise_mult
        self.iterations, self.turn_iterations, self.flop_iterations = iterations, turn_iterations, flop_iterations
        self.flop_cards = flop_cards                     # сколько карт тёрна и ривера перебирать на флопе
        self.solve_flop = solve_flop
        self.flop_sampling = flop_sampling               # как выбирать карты тёрна и ривера на флопе
        self.flop_seed = 0
        # число рейзов на ривере внутри решения тёрна (по умолчанию 0 или 1 — по флагу turn_river_raise)
        self.turn_river_raises = turn_river_raises
        # сеть-оценщик «после тёрна» (poker_coach.valuenet.load): если задана, ривер внутри решения тёрна
        # не перебирается, его оценивает сеть
        self.value_net = value_net
        # Поочерёдные обновления игроков сходятся в 2–3 раза быстрее (ниже эксплуатируемость), но совпадение
        # с эталоном PokerBench с ними ниже (см. README), поэтому по умолчанию оба игрока обновляются за проход.
        self.alternate = False
        self.quadratic = None      # усреднение итераций: None — квадратичное при поочерёдных обновлениях
        self.max_raises = max_raises                     # рейзов подряд на текущей улице (2 — можно ответить ре-рейзом; медленнее, прироста точности не дало)
        self._cache: dict = {}

    @staticmethod
    def available() -> bool:
        return solver_rs is not None

    supports_path = True      # умеет решать споты, где герой уже ставил на этой улице

    def supports(self, spot: Spot) -> bool:
        streets = (3, 4, 5) if self.solve_flop else (4, 5)
        return (solver_rs is not None and len(spot.board) in streets and spot.stack > 0
                and spot.facing_bet <= spot.stack and len(spot.villain_range) >= 2)

    # ---------- раскладка продолжений борда ----------

    def _layout(self, board):
        """-> (полные борды для вскрытий, карты выходов, что дальше, веса, что после текущей улицы)."""
        deck = [c for c in eval7.Deck().cards if c not in board]
        if len(board) == 5:
            return [board], [], [], [], 0
        if len(board) == 4:
            return [board + [c] for c in deck], [[_idx(c) for c in deck]], [list(range(len(deck)))], [1 / 44], -1
        rng = random.Random("".join(map(str, board)) + (str(self.flop_seed) if self.flop_seed else ""))   # одна выборка на флоп
        k = min(self.flop_cards, len(deck))
        pick = (lambda cards, off: _spread(cards, k, off)) if self.flop_sampling == "stratified" else \
               (lambda cards, off: rng.sample(cards, k))
        turns = pick(deck, 0)
        shows, cards, nexts, weights = [], [[_idx(c) for c in turns]], [[-(i + 2) for i in range(k)]], [49 / (k * 45)]
        for n, t in enumerate(turns):
            rest = [c for c in deck if c != t]
            rivers = pick(rest, n + 1)
            cards.append([_idx(c) for c in rivers])
            nexts.append([len(shows) + j for j in range(k)])
            weights.append(48 / (k * 44))
            shows += [board + [t, r] for r in rivers]
        return shows, cards, nexts, weights, -1

    def _run(self, board, ranges, weights, pot, stack, extra, lead, hero_p, hero_i, path, want_reach=False,
             want_expl=False, want_all=False):
        board = list(board)
        n = len(board)
        shows, cards, nexts, dweights, root_next = self._layout(board)
        base = set(board)
        strengths = [[[0 if (set(h) & (set(full) - base)) else eval7.evaluate(list(h) + full) for h in r]
                      for full in shows] for r in ranges]
        if n == 5:
            cfg = (list(self.bet_sizes), self.max_raises, [], [], self.iterations)
        elif n == 4:
            cfg = (list(self.turn_sizes), self.max_raises, [list(self.turn_river_sizes)],
                   [self.turn_river_raises if self.turn_river_raises is not None else (1 if self.turn_river_raise else 0)],
                   self.turn_iterations)
        else:
            cfg = (list(self.flop_sizes), self.max_raises, [list(self.flop_turn_sizes), list(self.flop_river_sizes)],
                   [0, 0], self.flop_iterations)
        sizes, root_raise, deeper_sizes, deeper_raise, iters = cfg
        net = self.value_net if n == 4 else None
        if net is not None:
            root_next = -1000                    # листья тёрна оценивает сеть; выход ривера 0 — для олл-инов и эквити
        key = (tuple(map(str, board)), tuple(sorted(str(a) + str(b) for a, b in ranges[0])),
               tuple(sorted(str(a) + str(b) for a, b in ranges[1])),
               tuple(round(w, 4) for w in weights[0]), tuple(round(w, 4) for w in weights[1]),
               round(pot, 2), round(stack, 2), round(extra, 2), lead, hero_p, hero_i, tuple(path), want_reach, want_expl, want_all, iters,
               self.flop_cards, self.flop_sampling, self.flop_seed, self.alternate, self.quadratic, net is not None,
               tuple(sizes), root_raise, str(deeper_sizes), tuple(deeper_raise))
        if key not in self._cache:
            if len(self._cache) > 200:
                self._cache.clear()
            self._cache[key] = solver_rs.solve(
                [(_idx(a), _idx(b)) for a, b in ranges[0]], [(_idx(a), _idx(b)) for a, b in ranges[1]],
                [float(w) for w in weights[0]], [float(w) for w in weights[1]], strengths[0], strengths[1],
                cards, nexts, dweights, root_next, float(pot), float(stack), sizes, root_raise,
                deeper_sizes, deeper_raise, self.raise_mult, float(extra), lead, iters, hero_p, hero_i,
                [(int(k), float(a)) for k, a in path], want_reach, want_expl, want_all, self.alternate,
                [_idx(c) for c in board], False, net, self.quadratic)
        return self._cache[key]

    # ---------- интерфейс ----------

    def solve(self, spot: Spot) -> SolverResult:
        hero_p = 1 if spot.hero_in_position else 0
        hero_range = list(spot.hero_range)
        hw = list(spot.hero_weights) if spot.hero_weights else [1.0] * len(hero_range)
        if not any(set(c) == set(spot.hero) for c in hero_range):
            hero_range.append(spot.hero)
            hw.append(max(hw, default=1.0))
        vw = list(spot.villain_weights) if spot.villain_weights else [1.0] * len(spot.villain_range)
        ranges, weights = [None, None], [None, None]
        ranges[hero_p], ranges[1 - hero_p] = hero_range, list(spot.villain_range)
        weights[hero_p], weights[1 - hero_p] = hw, vw
        hero_i = next(i for i, c in enumerate(hero_range) if set(c) == set(spot.hero))
        if spot.path is not None:
            path = list(spot.path)
        elif hero_p == 1 and spot.facing_bet == 0:
            path = [CHECK]
        elif hero_p == 1:
            path = [(BET, spot.facing_bet)]
        elif spot.facing_bet > 0:
            path = [CHECK, (BET, spot.facing_bet)]
        else:
            path = []
        extra = next((a for k, a in path if k == BET), 0.0)      # первая ставка линии должна быть в дереве
        acts, equity, w, *_rest = self._run(spot.board, ranges, weights, spot.pot, spot.stack, extra,
                                              spot.oop_can_lead, hero_p, hero_i, path)
        return SolverResult([ActionFreq(_KINDS[k], amt, f, ev) for k, amt, f, ev in acts], equity,
                            ranges[1 - hero_p], list(w), self.name)

    def reach_after(self, board, oop, ip, w_oop, w_ip, pot, stack, path, lead=True):
        """Веса рук обоих игроков после того, как на улице сыграна линия `path` (по средней стратегии)."""
        _a, _e, _w, r0, r1, *_rest = self._run(board, [list(oop), list(ip)], [list(w_oop), list(w_ip)], pot, stack,
                                       0.0, lead, 0, 0, path, want_reach=True)
        return list(r0), list(r1)

    def inspect(self, board, oop, ip, w_oop, w_ip, pot, stack, player: int, path=(), lead=True):
        """Для проверки солвера: стратегия и EV ВСЕХ рук игрока `player` в узле после `path` и эксплуатируемость.

        -> (действия [(тип, сумма)], частоты [действие][рука], EV [действие][рука], эксплуатируемость в bb)
        """
        acts, _e, _w, _r0, _r1, expl, freq, ev, *_rest = self._run(
            board, [list(oop), list(ip)], [list(w_oop), list(w_ip)], pot, stack, 0.0, lead, player, 0, list(path),
            want_expl=True, want_all=True)
        return [(_KINDS[k], amt) for k, amt, _f, _v in acts], freq, ev, expl
