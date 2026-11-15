"""Префлоп-таблица из решений солвера (обучающая часть PokerBench, лицензия Apache-2.0).

Ключ — линия торговли в нотации PokerBench («UTG/2.0bb/BTN/call») и позиция героя,
значение — действие для каждой из 169 стартовых рук. Покрывает мультивей, лимпы и олл-ины.

    python -m poker_coach.preflop_gto --build     # пересобрать data/preflop_gto.json из CSV
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import charts

TABLE = Path(__file__).resolve().parent / "data" / "preflop_gto.json"
_ORDER = "23456789TJQKA"


@dataclass
class Hit:
    label: str               # fold / call / check / allin / '7.5bb'
    exact: bool              # есть ли в таблице именно эта рука в этой линии
    neighbours: list[str]    # если нет — по каким рукам взято решение
    same_sizes: bool         # совпали ли размеры ставок с линией из таблицы

    @property
    def kind(self) -> str:
        return "raise" if self.label.endswith("bb") else self.label

    @property
    def amount(self) -> float | None:
        return float(self.label[:-2]) if self.label.endswith("bb") else None


def abstract(line: str) -> str:
    """Убрать размеры ставок: 'UTG/2.0bb/BTN/call' -> 'UTG/r/BTN/call'."""
    return re.sub(r"[\d.]+bb", "r", line)


@lru_cache(maxsize=1)
def _load():
    if not TABLE.exists():
        return {}, {}
    raw = json.loads(TABLE.read_text())
    by_abs: dict[str, dict[str, str]] = defaultdict(dict)
    for key, hands in raw.items():
        line, hero = key.rsplit("|", 1)
        tgt = by_abs[abstract(line) + "|" + hero]
        for h, lab in hands.items():
            tgt.setdefault(h, lab)
    return raw, dict(by_abs)


def _dist(a: str, b: str) -> float:
    """Расстояние между двумя стартовыми руками: старшая карта, младшая карта, тип."""
    ah, al, bh, bl = (_ORDER.index(x) for x in (a[0], a[1], b[0], b[1]))
    kind = lambda k: 0 if len(k) == 2 else (1 if k[2] == "s" else 2)
    return abs(ah - bh) + abs(al - bl) + (0 if kind(a) == kind(b) else 1.5 if 0 in (kind(a), kind(b)) else 0.75)


def lookup(line: str, hero_pos: str, cards: list[str]) -> Hit | None:
    exact_t, abs_t = _load()
    key = charts.hand_key(cards)
    hands, same = exact_t.get(f"{line}|{hero_pos}"), True
    if hands is None or key not in hands:
        alt = abs_t.get(f"{abstract(line)}|{hero_pos}")
        if alt is not None and (hands is None or key in alt):
            hands, same = alt, hands is not None
    if not hands:
        return None
    if key in hands:
        return Hit(hands[key], True, [], same)
    near = sorted(hands, key=lambda h: _dist(key, h))[:3]
    if len(hands) < 12 or _dist(key, near[0]) > 3:
        return None                                    # в этой линии слишком мало рук, чтобы судить
    kinds = Counter("raise" if hands[h].endswith("bb") else hands[h] for h in near)
    best = kinds.most_common(1)[0][0]
    label = next(hands[h] for h in near if ("raise" if hands[h].endswith("bb") else hands[h]) == best)
    return Hit(label, False, near, same)


_RANKS = "AKQJT98765432"
ALL_KEYS = [a + b if i == j else (a + b + "s" if i < j else b + a + "o")
            for i, a in enumerate(_RANKS) for j, b in enumerate(_RANKS)]


@lru_cache(maxsize=None)
def range_keys(abs_line: str, pos: str, kind: str) -> tuple[str, ...] | None:
    """Руки, с которыми солвер в этой линии делает `kind` (raise или call).

    Линия без размеров ставок: '' — открытие, 'CO/r' — против опена с CO, 'CO/r/BTN/r' — против 3-бета.
    Рук, которых в таблице нет, относим к действию ближайших по силе рук. None — линия покрыта слишком слабо.
    """
    hands = _load()[1].get(f"{abs_line}|{pos}")
    if not hands or len(hands) < 40:
        return None
    norm = lambda lab: "raise" if lab.endswith("bb") or lab == "allin" else lab
    out = []
    for key in ALL_KEYS:
        if key in hands:
            lab = norm(hands[key])
        else:
            near = sorted(hands, key=lambda h: _dist(key, h))[:3]
            lab = Counter(norm(hands[h]) for h in near).most_common(1)[0][0]
        if lab == kind:
            out.append(key)
    return tuple(out)


def build(csv_path: Path) -> int:
    import csv
    votes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            h = row["hero_holding"]
            votes[f"{row['prev_line']}|{row['hero_pos']}"][charts.hand_key([h[:2], h[2:]])][row["correct_decision"]] += 1
    table = {k: {h: c.most_common(1)[0][0] for h, c in sorted(hs.items())} for k, hs in sorted(votes.items())}
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(table, separators=(",", ":")))
    _load.cache_clear()
    return sum(len(v) for v in table.values())


if __name__ == "__main__":
    import sys
    from .bench import DATA, load
    name = "preflop_60k_train_set_game_scenario_information.csv"
    load(name)
    print("записей в таблице:", build(DATA / name), "→", TABLE) if "--build" in sys.argv else print(__doc__)
