"""Стартовые диапазоны постфлопа, восстановленные из обучающей части PokerBench.

Руки героя в спотах флопа — выборка из диапазона, с которым солвер авторов датасета пришёл на флоп
по данной префлоп-линии. Частота руки в выборке даёт её вес в диапазоне.

    python -m poker_coach.range_data --build     # пересобрать data/ranges_pb.json из CSV
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from . import charts

FILE = Path(__file__).resolve().parent / "data" / "ranges_pb.json"
MIN_ROWS = 150          # меньше наблюдений — диапазону из данных не доверяем


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(FILE.read_text()) if FILE.exists() else {}


@lru_cache(maxsize=None)
def lookup(line: str, me: str, opp: str):
    """-> (комбинации, веса) диапазона игрока `me` для линии `line` против `opp` или None."""
    g = _load().get(f"{line}|{me}|{opp}")
    if not g:
        return None
    combos, weights = [], []
    for key, w in g.items():
        cs = charts.combos(key)
        combos += cs
        weights += [w] * len(cs)
    return combos, weights


def build(csv_path: Path) -> int:
    import csv
    from . import bench
    flop: dict[str, Counter] = defaultdict(Counter)       # руки на флопе до первого действия героя
    seen: dict[str, Counter] = defaultdict(Counter)       # руки на любой улице (для состава диапазона)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                st = bench.postflop_state(row)
            except Exception:
                continue
            key = f"{st.preflop_line.value}|{st.hero_pos}|{st.villain_pos}"
            hand = charts.hand_key(st.hero_cards)
            seen[key][hand] += 1
            if row["evaluation_at"] != "Flop":
                continue
            me = row["hero_position"]
            mine = [t for t in row["postflop_action"].split("/") if t.startswith(me)]
            # игрок в позиции ещё не действовал; игроку без позиции разрешаем его чек (он чекает почти всегда)
            if not mine or (me == "OOP" and mine == ["OOP_CHECK"]):
                flop[key][hand] += 1
    n_combos = lambda h: 6 if len(h) == 2 else (4 if h[2] == "s" else 12)
    out = {}
    for key, counts in flop.items():
        if sum(counts.values()) < MIN_ROWS:
            continue
        dens = {h: c / n_combos(h) for h, c in counts.items()}
        ranked = sorted(dens.values())
        top = ranked[int(len(ranked) * 0.85)]              # 85-й перцентиль: устойчивее максимума к шуму
        w = {h: round(min(1.0, d / top), 3) for h, d in dens.items()}
        for h, c in seen[key].items():                     # рука встречалась на поздних улицах — редкая, но есть
            if h not in w and c >= 2:
                w[h] = 0.15
        out[key] = dict(sorted(w.items()))
    FILE.write_text(json.dumps(out, separators=(",", ":")))
    _load.cache_clear()
    lookup.cache_clear()
    return len(out)


if __name__ == "__main__":
    import sys
    from .bench import DATA
    from .tune import TRAIN
    print("линий с диапазонами:", build(DATA / TRAIN), "→", FILE) if "--build" in sys.argv else print(__doc__)
