"""Точные вычисления: эквити, ауты, шансы банка, SPR. Никакой LLM."""
from __future__ import annotations

from itertools import combinations

import eval7

Combo = tuple[eval7.Card, eval7.Card]

HAND_ORDER = ["High Card", "Pair", "Two Pair", "Trips", "Straight", "Flush",
              "Full House", "Quads", "Straight Flush"]
HAND_RU = {
    "High Card": "старшая карта", "Pair": "пара", "Two Pair": "две пары", "Trips": "сет/трипс",
    "Straight": "стрит", "Flush": "флеш", "Full House": "фулл-хаус", "Quads": "каре",
    "Straight Flush": "стрит-флеш",
}
OUT_RU = {**HAND_RU, "Pair": "топ-пару (оверкарта)"}


def _rank(cards) -> int:
    return HAND_ORDER.index(eval7.handtype(eval7.evaluate(cards)))


def hand_name(hero, board) -> str:
    return HAND_RU[eval7.handtype(eval7.evaluate(hero + board))]


def live_combos(rng: list[Combo], dead) -> list[Combo]:
    dead = set(dead)
    return [c for c in rng if c[0] not in dead and c[1] not in dead]


def equity_vs_range(hero, rng: list[Combo], board, mc_iters: int = 30000) -> float:
    """Доля банка, которую рука героя в среднем выигрывает против диапазона.

    С флопа — полный перебор оставшихся карт, на префлопе — Монте-Карло.
    """
    rng = live_combos(rng, hero + board)
    if not rng:
        raise ValueError("В диапазоне оппонента не осталось рук")
    if len(board) < 3:
        hr = eval7.HandRange("AA")
        hr.hands = [(c, 1.0) for c in rng]
        return eval7.py_hand_vs_range_monte_carlo(hero, hr, board, mc_iters)
    deck = [c for c in eval7.Deck().cards if c not in set(hero + board)]
    score = total = 0.0
    for runout in combinations(deck, 5 - len(board)):
        full = board + list(runout)
        hv = eval7.evaluate(hero + full)
        rs = set(runout)
        for a, b in rng:
            if a in rs or b in rs:
                continue
            vv = eval7.evaluate([a, b] + full)
            total += 1
            if hv > vv:
                score += 1
            elif hv == vv:
                score += 0.5
    return score / total


def outs(hero, board) -> dict[str, list[str]]:
    """Карты, которые усиливают руку героя, по типу усиления. Только флоп и тёрн.

    Считаются «по категории» (флеш, стрит, сет...) и не проверяют, что усиление
    точно даёт лучшую руку — часть аутов может быть «грязной».
    """
    if len(board) not in (3, 4):
        return {}
    cur = _rank(hero + board)
    hero_ranks = {c.rank for c in hero}
    top_board = max(c.rank for c in board)
    res: dict[str, list[str]] = {}
    for card in eval7.Deck().cards:
        if card in hero or card in board:
            continue
        new = _rank(hero + board + [card])
        if new <= cur or new <= _rank(board + [card]):
            continue
        name = HAND_ORDER[new]
        if name in ("Pair", "Two Pair", "Trips") and card.rank not in hero_ranks:
            continue
        if name == "Pair" and card.rank < top_board:
            continue
        res.setdefault(OUT_RU[name], []).append(str(card))
    return res


def count_outs(o: dict[str, list[str]]) -> int:
    return sum(len(v) for v in o.values())


def pot_odds(pot: float, to_call: float) -> float:
    """Минимальное эквити, при котором колл окупается. pot уже включает ставку оппонента."""
    return to_call / (pot + to_call) if to_call > 0 else 0.0


def spr(stack: float, pot: float) -> float:
    return stack / pot
