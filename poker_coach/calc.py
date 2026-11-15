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
    board_paired = len({c.rank for c in board}) < len(board)
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
        if name == "Two Pair" and board_paired and card.rank < top_board:
            continue          # «две пары» за счёт пары на борде — слабое усиление, аутом не считаем
        res.setdefault(OUT_RU[name], []).append(str(card))
    return res


def count_outs(o: dict[str, list[str]]) -> int:
    return sum(len(v) for v in o.values())


def pot_odds(pot: float, to_call: float) -> float:
    """Минимальное эквити, при котором колл окупается. pot уже включает ставку оппонента."""
    return to_call / (pot + to_call) if to_call > 0 else 0.0


def spr(stack: float, pot: float) -> float:
    return stack / pot


def is_weak_showdown(cards, board) -> bool:
    """Рука, с которой ставка на ривере была блефом: без пары либо пара ниже второй карты борда."""
    kind = eval7.handtype(eval7.evaluate(cards + board))
    if kind == "High Card":
        return True
    if kind != "Pair":
        return False
    board_ranks = sorted(c.rank for c in board)
    if len(set(board_ranks)) < len(board_ranks):
        return True                                   # пара лежит на борде
    mine = [c.rank for c in cards]
    pair_rank = mine[0] if mine[0] == mine[1] else next(r for r in mine if r in board_ranks)
    return pair_rank < board_ranks[-2]


def equity_multiway(hero, ranges: list[list[Combo]], board, iters: int = 2500, seed: int = 0) -> float:
    """Эквити героя против нескольких диапазонов сразу (Монте-Карло). Ничья делит банк поровну."""
    import random
    rng = random.Random(seed)
    base_dead = set(hero) | set(board)
    ranges = [live_combos(r, base_dead) for r in ranges]
    if any(not r for r in ranges):
        raise ValueError("В диапазоне одного из оппонентов не осталось рук")
    deck = [c for c in eval7.Deck().cards if c not in base_dead]
    need, score, done = 5 - len(board), 0.0, 0
    for _ in range(iters):
        used, hands = set(), []
        for r in ranges:
            for _try in range(8):
                c = r[rng.randrange(len(r))]
                if c[0] not in used and c[1] not in used:
                    used.update(c)
                    hands.append(c)
                    break
        if len(hands) < len(ranges):
            continue
        rest = [c for c in deck if c not in used]
        full = board + rng.sample(rest, need)
        hv = eval7.evaluate(hero + full)
        best = max(eval7.evaluate(list(h) + full) for h in hands)
        done += 1
        if hv > best:
            score += 1
        elif hv == best:
            score += 0.5
    return score / max(done, 1)
