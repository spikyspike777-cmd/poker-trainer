"""Проверка обученной сети-оценщика.

    python -m poker_coach.valuenet.evaluate --net poker_coach/data/value_turn.npz --data data/valuenet/turn --spots 30

1. Ошибка предсказания на последних порциях данных (тех, что при обучении шли в проверку),
   по группам глубины стека.
2. Сквозная проверка: споты тёрна из PokerBench решаются дважды — точно (с перебором всех карт ривера)
   и с сетью вместо ривера; сравниваются стратегия, EV и время.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import statistics
import time

import numpy as np

from . import load
from .features import C_CTX, F_HAND
from .gen import RIVER_RAISES, RIVER_SIZES


def unpack(z) -> dict:
    """Разобрать блок весов из .npz на матрицы (тот же порядок, что в model.export и в solver_rs)."""
    hc, h, nb, blob = int(z["hc"]), int(z["h"]), int(z["blocks"]), np.asarray(z["blob"], np.float32)
    at = 0

    def take(*shape):
        nonlocal at
        size = int(np.prod(shape))
        out = blob[at:at + size].reshape(shape)
        at += size
        return out

    net = dict(wc1=take(hc, C_CTX), bc1=take(hc), wc2=take(hc, hc), bc2=take(hc), heads=[])
    for _ in range(2):
        head = dict(wf=take(h, F_HAND), wc=take(h, hc), b=take(h), blocks=[(take(h, h), take(h)) for _ in range(nb)])
        head["wo"], head["bo"] = take(h), float(take(1)[0])
        net["heads"].append(head)
    assert at == len(blob)
    return net


def forward(net: dict, ctx: np.ndarray, feats: list[np.ndarray], weights: list[np.ndarray]) -> list[np.ndarray]:
    """Прямой проход на numpy — эталон, с которым сверяются PyTorch и Rust. -> поправки для двух игроков."""
    c = np.maximum(net["wc2"] @ np.maximum(net["wc1"] @ ctx + net["bc1"], 0) + net["bc2"], 0)
    out = []
    for p in range(2):
        hd = net["heads"][p]
        z = np.maximum(feats[p] @ hd["wf"].T + hd["wc"] @ c + hd["b"], 0)
        for w, b in hd["blocks"]:
            z = z + np.maximum(z @ w.T + b, 0)
        out.append(z @ hd["wo"] + hd["bo"])
    err = float(out[0] @ weights[0] + out[1] @ weights[1])
    return [o - err / 2 for o in out]


def data_error(net_path: str, data_dir: str, limit: int = 3000) -> None:
    net = unpack(np.load(net_path))
    files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    groups: dict[str, list[tuple[float, float]]] = {"стек меньше банка": [], "стек 1–4 банка": [], "стек больше 4 банков": []}
    done = 0
    for path in files[-max(1, len(files) // 10):]:
        d = np.load(path)
        for i in range(len(d["ctx"])):
            n = d["n"][i]
            feats = [d["f"][i, p, :n[p]].astype(np.float32) for p in range(2)]
            w = [d["w"][i, p, :n[p]] for p in range(2)]
            t = np.concatenate([d["t"][i, p, :n[p]] for p in range(2)])
            pred = np.concatenate(forward(net, d["ctx"][i], feats, w))
            spr = float(np.expm1(d["ctx"][i, 1] * 4))                  # стек / банк, из второго признака контекста
            key = "стек меньше банка" if spr < 1 else "стек 1–4 банка" if spr < 4 else "стек больше 4 банков"
            groups[key].append((100 * float(np.abs(pred - t).mean()), 100 * float(np.abs(t).mean())))
            done += 1
            if done >= limit:
                break
        if done >= limit:
            break
    print(f"Ошибка предсказания на {done} примерах, % банка (среднее по рукам):")
    for name, v in groups.items():
        if v:
            errs, base = [x for x, _ in v], [y for _, y in v]
            print(f"  {name:22} n={len(v):5}  с сетью {statistics.mean(errs):.2f}%  без сети {statistics.mean(base):.2f}%  "
                  f"худшие 5% примеров: от {sorted(errs)[int(len(errs) * 0.95)]:.2f}%")


def end_to_end(net_path: str, n_spots: int, seed: int = 0) -> None:
    from .. import bench, coach, ranges
    from ..solver import RustCFR
    from ..state import to_eval7
    net = load(net_path)
    kw = dict(turn_sizes=(0.33, 0.75), turn_iterations=60)
    exact = RustCFR(turn_river_sizes=RIVER_SIZES, turn_river_raises=RIVER_RAISES, **kw)
    plain = RustCFR(turn_river_sizes=(0.75,), turn_river_raises=0, **kw)      # нынешнее упрощённое дерево
    fast = RustCFR(value_net=net, **kw)
    for s in (exact, plain, fast):
        s.alternate = True
    rows = [r for r in bench.load(bench.POST) if r["evaluation_at"] == "Turn"]
    random.Random(seed).shuffle(rows)
    stats = {"сеть": [], "упрощённое дерево": []}
    times = {"точно": 0.0, "сеть": 0.0, "упрощённое дерево": 0.0}
    n = 0
    for row in rows:
        if n >= n_spots:
            break
        st = bench.postflop_state(row)
        got = ranges.solved_ranges(st, coach.SOLVER) or None
        if got is None:
            continue
        h, hw, v, vw = got
        oop, w_oop, ip, w_ip = (v, vw, h, hw) if st.in_position else (h, hw, v, vw)
        pot0, stack0 = st.pot - st.to_call - 2 * st.hero_invested, st.stack + st.hero_invested
        if pot0 <= 0 or stack0 <= 0.5:
            continue
        args = (to_eval7(st.board), oop, ip, w_oop, w_ip, pot0, stack0)
        res = {}
        for name, solver in (("точно", exact), ("сеть", fast), ("упрощённое дерево", plain)):
            t = time.time()
            res[name] = solver.inspect(*args, player=0)
            times[name] += time.time() - t
        a1, f1, e1, _x = res["точно"]
        w = np.array(w_oop)
        f1, e1 = np.array(f1), np.array(e1)
        ev1 = (f1 * e1).sum(0)
        for name in stats:
            a2, f2, e2, _y = res[name]
            if len(a2) != len(a1):
                continue
            f2, e2 = np.array(f2), np.array(e2)
            ev2 = (f2 * e2).sum(0)
            stats[name].append(dict(check=float((w * np.abs(f1[0] - f2[0])).sum() / w.sum()),
                                    ev=100 * float((w * np.abs(ev1 - ev2)).sum() / w.sum() / pot0),
                                    range_ev=100 * abs(float((w * (ev1 - ev2)).sum() / w.sum())) / pot0,
                                    same=float((w * (f1.argmax(0) == f2.argmax(0))).sum() / w.sum())))
        n += 1
    if not n:
        print("не нашлось подходящих спотов тёрна")
        return
    med = statistics.median
    print(f"\nСквозная проверка на {n} спотах тёрна (первый ход игрока без позиции). Эталон — точное решение "
          f"с полным деревом ривера ({times['точно'] / n:.2f} с на спот).")
    for name, st in stats.items():
        if not st:
            continue
        print(f"  {name} ({times[name] / n:.2f} с на спот):")
        print(f"    частота чека по рукам: расхождение, медиана {100 * med([s['check'] for s in st]):.1f} п.п., "
              f"максимум {100 * max(s['check'] for s in st):.1f} п.п.")
        print(f"    то же главное действие: у {100 * statistics.mean(s['same'] for s in st):.0f}% диапазона")
        print(f"    EV руки: расхождение, медиана {med([s['ev'] for s in st]):.2f}% банка, максимум {max(s['ev'] for s in st):.2f}%")
        print(f"    EV диапазона: расхождение, медиана {med([s['range_ev'] for s in st]):.2f}% банка, "
              f"максимум {max(s['range_ev'] for s in st):.2f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--net", required=True)
    p.add_argument("--data", help="каталог с порциями данных — для ошибки предсказания")
    p.add_argument("--spots", type=int, default=0, help="сколько спотов тёрна решить для сквозной проверки")
    a = p.parse_args()
    if a.data:
        data_error(a.net, a.data)
    if a.spots:
        end_to_end(a.net, a.spots)


if __name__ == "__main__":
    main()
