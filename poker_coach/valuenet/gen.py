"""Генератор обучающих данных для сети-оценщика «после тёрна».

Один пример: случайный борд из четырёх карт, банк, стек и диапазоны двух игроков. Цель — ценность
каждой руки, если торговля на тёрне закончена, а ривер ещё не вышел: наш солвер перебирает все 48 карт
ривера и на каждой решает торговлю ривера до равновесия.

    python -m poker_coach.valuenet.gen --out data/valuenet/turn --samples 100000 --workers 8

Запуск можно прерывать и продолжать: готовые файлы-порции пропускаются.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import eval7
import numpy as np

from .features import C_CTX, F_HAND

ALL_HANDS = [(a, b) for a in range(52) for b in range(a + 1, 52)]

RANKS, SUITS = "23456789TJQKA", "cdhs"
CARDS = [eval7.Card(RANKS[c // 4] + SUITS[c % 4]) for c in range(52)]
MAX_HANDS = 450                  # сколько рук оставляем в диапазоне: время решения растёт с их числом
RIVER_SIZES = (0.33, 0.75, 1.5)  # дерево ривера в обучающих данных: три размера ставки и до двух рейзов
RIVER_RAISES = 2


# ---------- случайные ситуации ----------

def _strength_order(board: list[int], rng: random.Random) -> list[tuple[int, int]]:
    """Все руки, не пересекающиеся с бордом, от слабых к сильным (с небольшим шумом)."""
    bset = set(board)
    b = [CARDS[c] for c in board]
    hands = [(x, y) for x, y in ALL_HANDS if x not in bset and y not in bset]
    scored = [(eval7.evaluate([CARDS[x], CARDS[y]] + b) * (1 + rng.gauss(0, 0.02)), (x, y)) for x, y in hands]
    scored.sort()
    return [h for _s, h in scored]


def _range_recursive(order, rng) -> dict:
    """Случайный диапазон в духе DeepStack: масса рекурсивно делится между слабой и сильной половиной."""
    w = [0.0] * len(order)

    def fill(lo: int, hi: int, mass: float) -> None:
        if hi - lo == 1:
            w[lo] = mass
            return
        mid = (lo + hi) // 2
        p = rng.random()
        fill(lo, mid, mass * p)
        fill(mid, hi, mass * (1 - p))

    fill(0, len(order), 1.0)
    return dict(zip(order, w))


def _range_subset(order, rng) -> dict:
    """Случайное подмножество рук с шумными весами."""
    k = rng.randint(30, MAX_HANDS)
    return {h: rng.lognormvariate(0, 0.5) for h in rng.sample(order, min(k, len(order)))}


_START = None


def _start_ranges():
    """Стартовые диапазоны тренера (из данных и чартов) — чтобы часть примеров была похожа на реальные."""
    global _START
    if _START is None:
        from .. import charts, range_data
        out = []
        for key in range_data._load():
            line, me, opp = key.split("|")
            combos, weights = range_data.lookup(line, me, opp)
            out.append([((c[0].rank * 4 + c[0].suit, c[1].rank * 4 + c[1].suit), w) for c, w in zip(combos, weights)])
        for rng_str in list(charts.RFI.values()):
            out.append([((a.rank * 4 + a.suit, b.rank * 4 + b.suit), 1.0) for a, b in charts.combos(rng_str)])
        _START = out
    return _START


def _range_realistic(order, rng) -> dict:
    """Стартовый диапазон тренера, суженный случайными «действиями» и зашумлённый."""
    pos = {tuple(sorted(h)): i for i, h in enumerate(order)}        # место руки по силе на этом борде
    base = rng.choice(_start_ranges())
    w = {tuple(sorted(h)): x for h, x in base if tuple(sorted(h)) in pos}
    n = len(order)
    for _ in range(rng.randint(0, 3)):                              # ставка, колл или чек на прошлых улицах
        kind = rng.choice(["bet", "call", "check"])
        lo, hi = {"bet": (0.5, 1.0), "call": (0.4, 0.9), "check": (0.0, 0.75)}[kind]
        lo, hi = lo + rng.uniform(-0.15, 0.15), hi + rng.uniform(-0.1, 0.1)
        bluffs = rng.uniform(0, 0.2) if kind == "bet" else 0.0
        for h in list(w):
            q = pos[h] / n
            keep = 1.0 if lo <= q <= hi else (0.6 if q < bluffs else rng.uniform(0, 0.15))
            w[h] *= keep
    tilt = rng.uniform(-1, 2) if rng.random() < 0.3 else 0.0        # иногда перекос к сильным или слабым рукам
    return {h: x * rng.lognormvariate(0, 0.4) * ((pos[h] + 1) / n) ** tilt for h, x in w.items() if x > 0}


def sample_spot(rng: random.Random):
    """-> (борд, банк, стек, [руки игрока 0, руки игрока 1], [веса, веса]); веса каждого игрока в сумме дают 1."""
    while True:
        board = rng.sample(range(52), 4)
        eff = 100.0 if rng.random() < 0.7 else rng.uniform(20, 200)   # эффективный стек на начало раздачи
        pot = min(math.exp(rng.uniform(math.log(2.0), math.log(2 * eff * 0.97))), 2 * eff - 1.0)
        stack = eff - pot / 2
        order = _strength_order(board, rng)
        hands, weights = [], []
        for _p in range(2):
            kind = rng.choices(["real", "rec", "sub"], weights=[50, 35, 15])[0]
            r = {"real": _range_realistic, "rec": _range_recursive, "sub": _range_subset}[kind](order, rng)
            top = sorted(r.items(), key=lambda kv: -kv[1])[:MAX_HANDS]
            top = [(h, x) for h, x in top if x > 1e-6 * top[0][1]]
            tot = sum(x for _h, x in top)
            hands.append([h for h, _x in top])
            weights.append([x / tot for _h, x in top])
        if min(len(hands[0]), len(hands[1])) >= 10:
            return board, pot, stack, hands, weights


# ---------- цель: решение солвером ----------

def _strengths(board, hands):
    """Силы рук на каждой из 48 карт ривера (0 — рука содержит эту карту)."""
    deck = [c for c in range(52) if c not in board]
    b = [CARDS[c] for c in board]
    return deck, [[[0 if r in h else eval7.evaluate([CARDS[h[0]], CARDS[h[1]]] + b + [CARDS[r]]) for h in hs]
                   for r in deck] for hs in hands]


def solve_target(board, pot, stack, hands, weights, iters: int = 100,
                 river_sizes=RIVER_SIZES, river_raises: int = RIVER_RAISES):
    """-> (контекст, признаки рук [2], поправки [2]).

    Поправка — насколько ценность руки с торговлей на ривере отличается от ценности без ставок,
    в долях банка; её и учит сеть. Признаки считает солвер — те же, что он считает при игре.
    """
    import solver_rs
    deck, strengths = _strengths(board, hands)
    w = [[float(x) for x in weights[0]], [float(x) for x in weights[1]]]
    ctx, f0, f1, e0, e1 = solver_rs.leaf_features(hands[0], hands[1], w[0], w[1], strengths[0], strengths[1],
                                                  deck, list(board), float(pot), float(stack))
    out = solver_rs.solve(
        hands[0], hands[1], w[0], w[1], strengths[0], strengths[1],
        [deck], [list(range(len(deck)))], [1 / 44], -1, float(pot), float(stack), [], 0,
        [list(river_sizes)], [river_raises], 3.0, 0.0, False, iters, 0, 0, [], False, False, False, True,
        list(board), True)
    targets = []
    for p, plain in enumerate((e0, e1)):
        m = compat_mass(hands[p], hands[1 - p], weights[1 - p])
        full = (np.array(out[8 + p]) - pot / 2 * m) / pot          # ценность с торговлей, в долях банка
        targets.append((full - np.array(plain)).astype(np.float32))
    feats = [np.array(f0, np.float32).reshape(-1, F_HAND), np.array(f1, np.float32).reshape(-1, F_HAND)]
    return np.array(ctx, np.float32), feats, targets


def compat_mass(mine, theirs, w_theirs) -> np.ndarray:
    """Для каждой моей руки: суммарный вес рук оппонента, не пересекающихся с ней по картам."""
    by_card = np.zeros(52)
    same = {}
    for (a, b), w in zip(theirs, w_theirs):
        by_card[a] += w
        by_card[b] += w
        same[(a, b)] = w
    total = float(sum(w_theirs))
    return np.array([total - by_card[a] - by_card[b] + same.get((a, b), 0.0) for a, b in mine])


# ---------- порции ----------

def make_shard(args) -> str:
    path, seed, count, iters = args
    if os.path.exists(path):
        return path
    rng = random.Random(seed)
    k = MAX_HANDS
    ctx = np.zeros((count, C_CTX), np.float32)
    n = np.zeros((count, 2), np.int16)                 # сколько рук у каждого игрока
    f = np.zeros((count, 2, k, F_HAND), np.float16)
    w = np.zeros((count, 2, k), np.float32)
    t = np.zeros((count, 2, k), np.float32)
    for i in range(count):
        b, p, s, hands, weights = sample_spot(rng)
        c, feats, targets = solve_target(b, p, s, hands, weights, iters)
        ctx[i] = c
        for pl in range(2):
            m = len(hands[pl])
            n[i, pl] = m
            f[i, pl, :m], w[i, pl, :m], t[i, pl, :m] = feats[pl], weights[pl], targets[pl]
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, ctx=ctx, n=n, f=f, w=w, t=t)
    os.replace(tmp, path)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="каталог для порций данных")
    p.add_argument("--samples", type=int, default=100000)
    p.add_argument("--shard", type=int, default=500, help="примеров в одном файле")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--iters", type=int, default=100, help="итераций солвера на пример (поочерёдные обновления)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", "1")       # параллелим по процессам, а не внутри солвера
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = (a.samples + a.shard - 1) // a.shard
    jobs = [(str(out / f"shard_{a.seed}_{i:05d}.npz"), a.seed * 1_000_003 + i, a.shard, a.iters) for i in range(n)]
    todo = [j for j in jobs if not os.path.exists(j[0])]
    print(f"порций: {n}, уже готово: {n - len(todo)}, осталось: {len(todo)} ({len(todo) * a.shard} примеров)", flush=True)
    start = time.time()
    with Pool(a.workers) as pool:
        for done, _path in enumerate(pool.imap_unordered(make_shard, todo), 1):
            rate = done * a.shard / (time.time() - start)
            left = (len(todo) - done) * a.shard / rate / 3600
            print(f"  {done}/{len(todo)} порций, {rate:.1f} примеров/с, осталось около {left:.1f} ч", flush=True)


if __name__ == "__main__":
    main()
