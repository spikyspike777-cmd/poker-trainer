"""Упрощённые префлоп-чарты 6-max, 100bb, кэш.

Это учебные диапазоны, близкие к общедоступным GTO-чартам, а не выгрузка из солвера.
Нотация eval7: '77+', 'A5s-A2s', 'KTo+'.
"""
from __future__ import annotations

from functools import lru_cache

import eval7

RFI = {
    "UTG": "55+, A9s+, A5s-A3s, KTs+, QTs+, JTs, T9s, 98s, AJo+, KQo",
    "HJ": "44+, A8s+, A5s-A2s, K9s+, Q9s+, J9s+, T9s, 98s, 87s, ATo+, KJo+, QJo",
    "CO": "22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 76s, 65s, 54s, A9o+, KTo+, QTo+, JTo",
    "BTN": "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, A2o+, K8o+, Q9o+, J9o+, T8o+, 98o",
    "SB": "22+, A2s+, K4s+, Q6s+, J7s+, T7s+, 96s+, 86s+, 75s+, 65s, 54s, A5o+, K9o+, Q9o+, J9o+, T9o",
}
OPEN_SIZE = {"UTG": 2.5, "HJ": 2.5, "CO": 2.5, "BTN": 2.5, "SB": 3.0}

# Против опена: (3-бет, колл). Ключ — (герой на BB?, группа позиции открывшегося).
_VS_OPEN = {
    (False, "early"): ("QQ+, AKs, AKo, A5s-A4s",
                       "JJ-77, AQs-AJs, KQs, QJs, JTs, T9s"),
    (False, "CO"): ("TT+, AQs+, AKo, A5s-A4s, KQs",
                    "99-55, AJs-ATs, KJs-KTs, QTs+, JTs, T9s, 98s, AQo"),
    (False, "BTN"): ("99+, ATs+, A5s-A2s, KTs+, QTs+, JTs, AJo+, KQo", ""),
    (True, "early"): ("QQ+, AKs, AKo, A5s-A4s, 65s",
                      "JJ-22, AQs-A6s, A3s-A2s, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, 54s, AQo-ATo, KJo+, QJo"),
    (True, "CO"): ("JJ+, AQs+, AKo, A5s-A2s, K9s, 76s, 65s",
                   "TT-22, AJs-A6s, KTs+, K8s-K5s, Q8s+, J8s+, T8s+, 97s+, 86s+, 75s+, 54s, AQo-A9o, KTo+, QTo+, JTo"),
    (True, "BTN"): ("TT+, AJs+, AQo+, A5s-A2s, KTs+, QTs, 87s, 76s",
                    "99-22, ATs-A6s, K9s-K2s, Q9s-Q4s, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, "
                    "AJo-A4o, K9o+, Q9o+, J9o+, T9o, 98o"),
    (True, "SB"): ("99+, ATs+, AJo+, KJs+, A5s-A2s, 87s, 76s",
                   "88-22, A9s-A6s, KTs-K2s, Q2s+, J5s+, T6s+, 95s+, 85s+, 74s+, 64s+, 53s+, "
                   "ATo-A2o, K7o+, Q8o+, J8o+, T8o+, 97o+, 87o"),
}
THREEBET_SIZE_IP, THREEBET_SIZE_OOP = 3.0, 4.0  # множитель к размеру опена

# Против 3-бета: (4-бет, колл). Ключ — открывался ли герой из ранней позиции.
_VS_3BET = {
    True: ("KK+, AKs", "QQ-TT, AQs, AKo, KQs"),
    False: ("QQ+, AKs, AKo, A5s", "JJ-88, AQs-AJs, KQs, KJs, QJs, JTs, T9s, AQo"),
}
VS_4BET = ("KK+, AKs", "QQ, AKo")  # (олл-ин, колл)


def _group(pos: str) -> str:
    return "early" if pos in ("UTG", "HJ") else pos


def hand_key(cards: list[str]) -> str:
    """['As','Kh'] -> 'AKo', ['7c','7d'] -> '77'."""
    order = "23456789TJQKA"
    a, b = sorted(cards, key=lambda c: order.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def combos(range_str: str) -> list[tuple[eval7.Card, eval7.Card]]:
    if not range_str.strip():
        return []
    return [h for h, _w in eval7.HandRange(range_str).hands]


def in_range(cards: list[str], range_str: str) -> bool:
    want = {eval7.Card(c) for c in cards}
    return any(set(h) == want for h in combos(range_str))


def rfi_range(pos: str) -> str:
    return RFI[pos]


def vs_open_ranges(hero_pos: str, opener_pos: str) -> tuple[str, str]:
    g = _group(opener_pos)
    if g == "BB" or (hero_pos != "BB" and g == "SB"):   # рейз из блайндов против лимпа — берём самую широкую группу
        g = "SB" if hero_pos == "BB" else "BTN"
    if hero_pos == "SB" and g != "BTN":
        # SB без позиции играет 3-бет или фолд.
        three, _call = _VS_OPEN[(False, g)]
        return three + ", JJ-99, AQs, AJs, KQs", ""
    return _VS_OPEN[(hero_pos == "BB", g)]


def vs_3bet_ranges(hero_pos: str) -> tuple[str, str]:
    return _VS_3BET[hero_pos in ("UTG", "HJ")]


def range_percent(range_str: str) -> float:
    return 100 * len(combos(range_str)) / 1326


def chen(key: str) -> float:
    """Формула Чена — грубая оценка силы стартовой руки ('AKs', '77', 'T9o')."""
    order = "23456789TJQKA"
    pts = {"A": 10, "K": 8, "Q": 7, "J": 6}
    hi, lo = order.index(key[0]), order.index(key[1])
    score = pts.get(key[0], (hi + 2) / 2)
    if hi == lo:
        return max(score * 2, 5)
    if key.endswith("s"):
        score += 2
    gap = hi - lo - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < order.index("Q"):
        score += 1
    return score


@lru_cache(maxsize=None)
def keys_to_combos(keys: tuple[str, ...]) -> tuple:
    """('AKs', '77') -> все комбинации этих рук."""
    return tuple(c for k in keys for c in combos(k))


def scale_range(rng, factor: float):
    """Диапазон в `factor` раз уже или шире: убираем слабейшие руки или добавляем лучшие из остальных.

    rng — строка в нотации диапазонов или готовый список комбинаций.
    """
    base = combos(rng) if isinstance(rng, str) else list(rng)
    if abs(factor - 1) < 0.05 or not base:
        return base
    key = lambda c: chen(hand_key([str(c[0]), str(c[1])]))
    target = max(4, int(len(base) * factor))
    if factor < 1:
        return sorted(base, key=key, reverse=True)[:target]
    have = {frozenset(c) for c in base}
    cards = eval7.Deck().cards
    rest = [(a, b) for i, a in enumerate(cards) for b in cards[i + 1:] if frozenset((a, b)) not in have]
    rest.sort(key=key, reverse=True)
    return base + rest[:target - len(base)]
