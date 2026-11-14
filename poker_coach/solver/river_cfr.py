"""Свой солвер ривера: CFR+ по всем рукам обоих диапазонов (numpy).

Дерево: чек / бет (несколько размеров) → фолд / колл / рейз → фолд / колл.
Игрок 0 — без позиции (ходит первым), игрок 1 — в позиции.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import eval7
import numpy as np

from .base import ActionFreq, SolverResult, Spot


@dataclass
class _Node:
    player: int = 0
    c: tuple[float, float] = (0.0, 0.0)          # вложено на улице игроками 0 и 1
    terminal: str = ""                            # "", "fold0", "fold1", "show"
    kinds: list[str] = field(default_factory=list)
    amounts: list[float] = field(default_factory=list)
    children: list["_Node"] = field(default_factory=list)
    regret: np.ndarray | None = None
    strat_sum: np.ndarray | None = None


class RiverCFR:
    name = "свой солвер ривера (CFR+)"

    def __init__(self, bet_sizes=(0.33, 0.75), raise_mult: float = 3.0, iterations: int = 150):
        self.bet_sizes, self.raise_mult, self.iterations = bet_sizes, raise_mult, iterations
        self._cache: dict = {}

    def supports(self, spot: Spot) -> bool:
        return len(spot.board) == 5 and spot.stack > 0 and spot.facing_bet <= spot.stack

    # ---------- дерево ----------

    def _bets(self, pot: float, stack: float, extra: float) -> list[float]:
        sizes = {round(min(pot * f, stack), 1) for f in self.bet_sizes}
        if stack <= 1.5 * pot:
            sizes.add(round(stack, 2))
        if extra > 0:
            sizes = {s for s in sizes if abs(s - extra) > 0.15 * extra}   # не дублируем близкий размер
            sizes.add(round(extra, 2))
        return sorted(s for s in sizes if s > 0)

    def _build(self, pot: float, stack: float, extra: float) -> _Node:
        def facing(player: int, c, raised: bool) -> _Node:
            me, opp = c[player], c[1 - player]
            n = _Node(player, c)
            n.kinds, n.amounts = ["fold", "call"], [me, opp]
            n.children = [_Node(terminal=f"fold{player}", c=c),
                          _Node(terminal="show", c=(opp, opp))]
            if not raised and opp < stack:
                to = min(opp * self.raise_mult, stack)
                if to > 0.6 * stack:
                    to = stack
                nc = (to, c[1]) if player == 0 else (c[0], to)
                n.kinds.append("allin" if to == stack else "raise")
                n.amounts.append(to)
                n.children.append(facing(1 - player, nc, True))
            return n

        def opening(player: int) -> _Node:
            n = _Node(player)
            n.kinds, n.amounts = ["check"], [0.0]
            n.children = [opening(1) if player == 0 else _Node(terminal="show")]
            for b in self._bets(pot, stack, extra):
                c = (b, 0.0) if player == 0 else (0.0, b)
                n.kinds.append("allin" if b == round(stack, 2) else "bet")
                n.amounts.append(b)
                n.children.append(facing(1 - player, c, False))
            return n

        return opening(0)

    # ---------- CFR ----------

    def solve(self, spot: Spot) -> SolverResult:
        hero_p = 1 if spot.hero_in_position else 0
        ranges = [None, None]
        hero_range = list(spot.hero_range)
        if not any(set(c) == set(spot.hero) for c in hero_range):
            hero_range.append(spot.hero)
        ranges[hero_p], ranges[1 - hero_p] = hero_range, list(spot.villain_range)
        key = (tuple(map(str, spot.board)), tuple(sorted(str(a) + str(b) for a, b in ranges[0])),
               tuple(sorted(str(a) + str(b) for a, b in ranges[1])), spot.pot, spot.stack, spot.facing_bet)
        if key not in self._cache:
            self._cache[key] = self._run(ranges, spot)
        root, W, L, T, M = self._cache[key]
        hero_i = next(i for i, c in enumerate(ranges[hero_p]) if set(c) == set(spot.hero))

        # идём к узлу решения героя, перемножая вероятности действий по средней стратегии
        reach = [np.ones(len(ranges[0]), np.float32), np.ones(len(ranges[1]), np.float32)]
        node, path = root, []
        if hero_p == 1 and spot.facing_bet == 0:
            path = [("check", 0.0)]
        elif hero_p == 1:
            path = [("bet", spot.facing_bet)]
        elif spot.facing_bet > 0:
            path = [("check", 0.0), ("bet", spot.facing_bet)]
        for kind, amt in path:
            idx = self._find(node, kind, amt)
            reach[node.player] = reach[node.player] * self._avg(node)[:, idx]
            node = node.children[idx]
        assert node.player == hero_p

        values = self._action_values(node, reach, spot.pot, W, L, T, M)
        opp = reach[1 - hero_p]
        valid = M[hero_i] if hero_p == 0 else M[:, hero_i]
        norm = float(valid @ opp) or 1.0
        wins = (W[hero_i] if hero_p == 0 else L[:, hero_i]) @ opp
        ties = (T[hero_i] if hero_p == 0 else T[:, hero_i]) @ opp
        strat = self._avg(node)[hero_i]
        acts = [ActionFreq(k, a, float(strat[i]), float(values[i][hero_i]) / norm)
                for i, (k, a) in enumerate(zip(node.kinds, node.amounts))]
        w = (opp * valid).tolist()
        return SolverResult(acts, float(wins + 0.5 * ties) / norm, ranges[1 - hero_p], w, self.name)

    @staticmethod
    def _find(node: _Node, kind: str, amt: float) -> int:
        if kind == "check":
            return 0
        cands = [i for i, k in enumerate(node.kinds) if k in ("bet", "allin")]
        return min(cands, key=lambda i: abs(node.amounts[i] - amt))

    @staticmethod
    def _avg(node: _Node) -> np.ndarray:
        s = node.strat_sum
        tot = s.sum(axis=1, keepdims=True)
        return np.where(tot > 0, s / np.maximum(tot, 1e-12), 1.0 / s.shape[1])

    def _run(self, ranges, spot: Spot):
        board = list(spot.board)
        s = [np.array([eval7.evaluate(list(c) + board) for c in r], np.int64) for r in ranges]
        cards = [[{a, b} for a, b in r] for r in ranges]
        M = np.array([[0.0 if x & y else 1.0 for y in cards[1]] for x in cards[0]], np.float32)
        W = M * (s[0][:, None] > s[1][None, :])
        L = M * (s[0][:, None] < s[1][None, :])
        T = M - W - L
        root = self._build(spot.pot, spot.stack, spot.facing_bet)
        n = [len(ranges[0]), len(ranges[1])]

        def init(node: _Node) -> None:
            if node.terminal:
                return
            node.regret = np.zeros((n[node.player], len(node.kinds)), np.float32)
            node.strat_sum = np.zeros_like(node.regret)
            for ch in node.children:
                init(ch)

        init(root)
        ones = [np.ones(n[0], np.float32), np.ones(n[1], np.float32)]
        for it in range(1, self.iterations + 1):
            self._cfr(root, ones[0], ones[1], spot.pot, W, L, T, M, it)
        return root, W, L, T, M

    @staticmethod
    def _terminal(node: _Node, p0, p1, pot, W, L, T, M):
        c0, c1 = node.c
        if node.terminal == "fold0":
            return -c0 * (M @ p1), (pot + c0) * (M.T @ p0)
        if node.terminal == "fold1":
            return (pot + c1) * (M @ p1), -c1 * (M.T @ p0)
        u0 = (pot + c1) * (W @ p1) - c0 * (L @ p1) + pot / 2 * (T @ p1)
        u1 = (pot + c0) * (L.T @ p0) - c1 * (W.T @ p0) + pot / 2 * (T.T @ p0)
        return u0, u1

    def _cfr(self, node: _Node, p0, p1, pot, W, L, T, M, it: int):
        if node.terminal:
            return self._terminal(node, p0, p1, pot, W, L, T, M)
        pos = np.maximum(node.regret, 0)
        tot = pos.sum(axis=1, keepdims=True)
        strat = np.where(tot > 0, pos / np.maximum(tot, 1e-12), 1.0 / pos.shape[1])
        me = node.player
        util = []
        other = 0
        for a, ch in enumerate(node.children):
            if me == 0:
                u0, u1 = self._cfr(ch, p0 * strat[:, a], p1, pot, W, L, T, M, it)
                util.append(u0)
                other = other + u1
            else:
                u0, u1 = self._cfr(ch, p0, p1 * strat[:, a], pot, W, L, T, M, it)
                util.append(u1)
                other = other + u0
        util = np.stack(util, axis=1)
        mine = (strat * util).sum(axis=1)
        node.regret = np.maximum(node.regret + util - mine[:, None], 0)       # CFR+
        node.strat_sum += it * (p0 if me == 0 else p1)[:, None] * strat       # линейное усреднение
        return (mine, other) if me == 0 else (other, mine)

    def _eval(self, node: _Node, p0, p1, pot, W, L, T, M):
        """Ценности обоих игроков при игре по средней стратегии."""
        if node.terminal:
            return self._terminal(node, p0, p1, pot, W, L, T, M)
        strat, me = self._avg(node), node.player
        mine, other = 0, 0
        for a, ch in enumerate(node.children):
            if me == 0:
                u0, u1 = self._eval(ch, p0 * strat[:, a], p1, pot, W, L, T, M)
                mine, other = mine + strat[:, a] * u0, other + u1
            else:
                u0, u1 = self._eval(ch, p0, p1 * strat[:, a], pot, W, L, T, M)
                mine, other = mine + strat[:, a] * u1, other + u0
        return (mine, other) if me == 0 else (other, mine)

    def _action_values(self, node: _Node, reach, pot, W, L, T, M):
        """Для каждого действия героя — вектор ценностей по его рукам (при честном выборе действия)."""
        out = []
        for ch in node.children:
            u = self._eval(ch, reach[0], reach[1], pot, W, L, T, M) if not ch.terminal else \
                self._terminal(ch, reach[0], reach[1], pot, W, L, T, M)
            out.append(u[node.player])
        return out
