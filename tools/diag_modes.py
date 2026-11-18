"""Почему лучше сходящийся солвер хуже совпадает с PokerBench: раскладка по факторам.

Каждый спот решается в нескольких конфигурациях, которые отличаются ровно одним фактором:
режим обновлений, усреднение итераций, откуда берутся диапазоны (каким солвером решаются прошлые улицы).

    python tools/diag_modes.py --n 400 --out diag.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RAYON_NUM_THREADS", "1")

from poker_coach import bench, coach, ranges  # noqa: E402
from poker_coach.params import P  # noqa: E402
from poker_coach.solver import RustCFR  # noqa: E402

ITERS = dict(iterations=300, turn_iterations=120, flop_iterations=80)


def make(alternate: bool, quadratic: bool | None = None, seed: int = 0, **kw) -> RustCFR:
    s = RustCFR(**{**ITERS, **kw})
    s.alternate, s.quadratic, s.flop_seed = alternate, quadratic, seed
    return s


SET = "factors"


# (имя, солвер для решения, солвер для цепочки диапазонов или None — тот же, диапазоны из решения?)
def configs():
    S, A = make(False), make(True)
    if SET == "pair":
        return [("S", S, S, True), ("A", A, A, True)]
    if SET == "seeds":                              # то же самое, но с другой случайной выборкой карт на флопе
        S7, A7 = make(False, seed=7), make(True, seed=7)
        return [("S", S, S, True), ("A", A, A, True), ("S_seed7", S7, S7, True), ("A_seed7", A7, A7, True)]
    return [
        ("S", S, S, True),                          # сейчас по умолчанию: одновременные, линейное усреднение
        ("A", A, A, True),                          # поочерёдные, квадратичное
        ("A_lin", make(True, False), make(True, False), True),   # поочерёдные, линейное
        ("S_quad", make(False, True), make(False, True), True),  # одновременные, квадратичное
        ("A_chainS", A, S, True),                   # решение поочерёдным, диапазоны — как сейчас
        ("S_chainA", S, A, True),                   # решение как сейчас, диапазоны — поочерёдным
        ("S_rules", S, None, False),                # без цепочки: диапазоны по правилам
        ("A_rules", A, None, False),
    ]


_CFG = None


def _init(name: str) -> None:
    global SET
    SET = name


def job(row):
    global _CFG
    if _CFG is None:
        _CFG = configs()
    orig = ranges.solved_ranges.__wrapped__ if hasattr(ranges.solved_ranges, "__wrapped__") else ranges.solved_ranges
    gold = row["correct_decision"].split()[0].lower()
    out = {"street": row["evaluation_at"], "gold": gold}
    try:
        st = bench.postflop_state(row)
    except Exception:
        return None
    out["facing"] = st.to_call > 0
    out["ip"] = st.in_position
    for name, solver, chain, use_chain in _CFG:
        coach.SOLVER = solver
        P["solved_ranges"] = 1 if use_chain else 0

        def patched(s, _solver, _chain=chain):
            return orig(s, _chain)
        patched.__wrapped__ = orig
        ranges.solved_ranges = patched
        try:
            a = coach.analyze(st)
            kind, _label = bench._label(a, row)
            freq = {}
            for k, _amt, f, _ev in a.all_actions:
                fam = "bet" if k == "allin" and not out["facing"] else "raise" if k == "allin" else k
                freq[fam] = freq.get(fam, 0.0) + f
            out[name] = dict(pred=kind, gold_freq=freq.get(gold, 0.0), top=max(freq.values()) if freq else None,
                             solved=bool(a.mix), source=a.source[-45:])
        except Exception as e:
            out[name] = dict(pred="error", err=str(e)[:80])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--set", default="factors", choices=["factors", "seeds", "pair"])
    p.add_argument("--test", action="store_true", help="споты из тестовой части (та же выборка, что у bench --n 600)")
    p.add_argument("--fresh", action="store_true",
                   help="споты тестовой части, которые не входят в выборку bench --n 600 (на них ничего не настраивалось)")
    a = p.parse_args()
    global SET
    SET = a.set
    if a.test or a.fresh:
        import random
        allrows = bench.load(bench.POST)
        used = random.Random(0).sample(allrows, 600)
        if a.fresh:
            seen = {id(r) for r in used}
            rest = [r for r in allrows if id(r) not in seen]
            rows = random.Random(a.seed).sample(rest, a.n)
        else:
            rows = used[:a.n]
    else:
        from poker_coach.tune import load_train
        rows = load_train(a.n, seed=a.seed)[0]
    with Pool(4, initializer=_init, initargs=(SET,)) as pool, open(a.out, "w") as f:
        for i, r in enumerate(pool.imap_unordered(job, rows, chunksize=2), 1):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
