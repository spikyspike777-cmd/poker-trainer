"""Подбор порогов эвристик на ОБУЧАЮЩЕЙ части PokerBench (покоординатный перебор).

    python -m poker_coach.tune            # пишет poker_coach/data/params.json и печатает проверку на отложенной выборке

Тестовая часть здесь не используется: на ней параметры только измеряются (python -m poker_coach.bench).
"""
from __future__ import annotations

import json
import random
import sys
from multiprocessing import Pool

from . import bench, coach, params

TRAIN = "postflop_500k_train.csv"
URL_NAME = "postflop_500k_train_set_game_scenario_information.csv"
GRID = {
    "bet_keep": [0.35, 0.5, 0.65], "bet_bluffs": [0.05, 0.12, 0.2],
    "raise_keep": [0.2, 0.3, 0.45], "raise_bluffs": [0.0, 0.08, 0.15],
    "call_keep": [0.35, 0.45, 0.6, 0.75], "check_back_drop": [0.1, 0.25, 0.4],
    "raise_eq": [0.66, 0.72, 0.8, 0.88], "raise_eq_river": [0.72, 0.8, 0.88],
    "call_margin": [-0.04, 0.0, 0.04], "call_margin_river": [-0.03, 0.02, 0.06],
    "value_big": [0.75, 0.8, 0.88], "value": [0.58, 0.65, 0.72],
    "thin": [0.45, 0.5, 0.58, 0.99], "semibluff_eq": [0.25, 0.3, 0.4, 0.99],
}


def _job(args) -> bool:
    row, values = args
    coach.SOLVER = None
    params.set_params(values)
    try:
        kind, _ = bench.postflop_predict(row)
    except Exception:
        return False
    return kind == row["correct_decision"].split()[0].lower()


def score(pool, rows, values) -> float:
    res = pool.map(_job, [(r, values) for r in rows], chunksize=16)
    return sum(res) / len(res)


def load_train(n: int, seed: int):
    import csv
    path = bench.DATA / TRAIN
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(bench.URL + URL_NAME, path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    r = random.Random(seed)
    r.shuffle(rows)
    return rows[:n], rows[n:2 * n]


def tune(table_ranges: int, n: int = 2400, log=print):
    fit, hold = load_train(n, seed=7)
    cur = dict(params.DEFAULT, table_ranges=table_ranges)
    with Pool(4) as pool:
        base_fit, base_hold = score(pool, fit, cur), score(pool, hold, cur)
        best = base_fit
        log(f"[table_ranges={table_ranges}] старт: подбор {best:.3f}, отложенная {base_hold:.3f}")
        for name, grid in GRID.items():
            for v in grid:
                if v == cur[name]:
                    continue
                s = score(pool, fit, dict(cur, **{name: v}))
                if s > best + 0.002:                    # меньшие улучшения — шум на такой выборке
                    best, cur[name] = s, v
            log(f"  {name:18} -> {cur[name]:<5} подбор {best:.3f}")
        hold_score = score(pool, hold, cur)
    log(f"[table_ranges={table_ranges}] итог: подбор {best:.3f}; отложенная {base_hold:.3f} -> {hold_score:.3f}")
    return cur, hold_score, base_hold


def main() -> None:
    results = [tune(t) for t in (0, 1)]
    best, hold, base = max(results, key=lambda x: x[1])
    if hold > base + 0.003 or best["table_ranges"]:
        params.FILE.write_text(json.dumps(best, indent=1))
        print("записано:", params.FILE, json.dumps(best))
    else:
        print("подбор не дал улучшения на отложенной выборке — параметры не меняем")


if __name__ == "__main__":
    main()
