"""Сверка нашего солвера с открытым postflop-solver на реальных спотах из PokerBench.

    cd tools/ref_compare && cargo build --release && cd ../..
    python tools/compare_ref.py --river 40 --turn 15

Оба солвера получают одинаковые диапазоны (с весами), банк, стек и одинаковое дерево без рейзов:
ривер — ставки 33% и 75% банка; тёрн — ставка 66%, на ривере 75%. Сравниваются стратегия и EV всех рук
в двух точках: первый ход игрока без позиции и ход игрока в позиции после чека.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poker_coach import bench, coach, ranges  # noqa: E402
from poker_coach.solver import RustCFR  # noqa: E402
from poker_coach.state import to_eval7  # noqa: E402

REF = ROOT / "tools" / "ref_compare" / "target" / "release" / "ref_compare"
idx = lambda c: c.rank * 4 + c.suit


def spot_from_row(row):
    """Спот начала улицы: (борд, руки и веса обоих игроков, банк, стек) или None."""
    st = bench.postflop_state(row)
    got = ranges.solved_ranges(st, coach.SOLVER)
    if got is None:
        return None
    h, hw, v, vw = got
    hero_set = set(to_eval7(st.hero_cards))
    h2 = [(c, w) for c, w in zip(h, hw) if w > 0.01]
    v2 = [(c, w) for c, w in zip(v, vw) if w > 0.01]
    oop, ip = (v2, h2) if st.in_position else (h2, v2)
    pot0 = st.pot - st.to_call - 2 * st.hero_invested
    stack0 = st.stack + st.hero_invested
    if pot0 <= 0 or stack0 < 1.6 * pot0 or min(len(oop), len(ip)) < 10:
        return None                                   # короткие стеки: у солверов разные правила добавления олл-ина
    return dict(board=to_eval7(st.board), oop=[c for c, _ in oop], w_oop=[w for _, w in oop],
                ip=[c for c, _ in ip], w_ip=[w for _, w in ip], pot=round(pot0, 2), stack=round(stack0, 2))


def node_stats(ours, ref, hands, weights, pot):
    """Расхождение в одной точке дерева. ours = (действия, частоты, EV); ref — узел из эталона."""
    acts, freq, ev = ours
    pos = {tuple(sorted((idx(a), idx(b)))): i for i, (a, b) in enumerate(hands)}
    rpos = {tuple(sorted(h)): j for j, h in enumerate(ref["hands"])}
    if len(acts) != len(ref["actions"]):
        return None
    tot = d_check = d_ev = ev_o = ev_r = 0.0
    for key, i in pos.items():
        j = rpos.get(key)
        if j is None:
            continue
        w = weights[i]
        mine = sum(freq[a][i] * ev[a][i] for a in range(len(acts)))
        theirs = sum(ref["strategy"][a][j] * ref["ev"][a][j] for a in range(len(acts)))
        d_check += w * abs(freq[0][i] - ref["strategy"][0][j])
        d_ev += w * abs(mine - theirs)
        ev_o, ev_r, tot = ev_o + w * mine, ev_r + w * theirs, tot + w
    return dict(check_diff=d_check / tot, ev_diff_pct=100 * d_ev / tot / pot,
                range_ev_diff_pct=100 * abs(ev_o - ev_r) / tot / pot,
                check_ours=sum(weights[i] * freq[0][i] for i in pos.values()) / tot,
                check_ref=sum(weights[i] * ref["strategy"][0][rpos[k]] for k, i in pos.items() if k in rpos) / tot)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--river", type=int, default=40)
    p.add_argument("--turn", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not REF.exists():
        sys.exit(f"нет эталона: соберите {REF.parent.parent.parent} (cargo build --release)")
    rows = bench.load(bench.POST)
    random.Random(a.seed).shuffle(rows)
    for street, want in (("River", a.river), ("Turn", a.turn)):
        spots = []
        for r in rows:
            if len(spots) >= want:
                break
            if r["evaluation_at"] == street:
                s = spot_from_row(r)
                if s:
                    spots.append(s)
        if not spots:
            continue
        river = street == "River"
        payload = [dict(board=[idx(c) for c in s["board"]], oop=[(idx(x), idx(y)) for x, y in s["oop"]],
                        ip=[(idx(x), idx(y)) for x, y in s["ip"]], w_oop=s["w_oop"], w_ip=s["w_ip"],
                        pot=s["pot"], stack=s["stack"], street_sizes="33%,75%" if river else "66%",
                        river_sizes="75%", max_iters=1000, target_pct=0.1) for s in spots]
        t = time.time()
        ref = json.loads(subprocess.run([str(REF)], input=json.dumps(payload), capture_output=True, text=True,
                                        check=True).stdout)
        t_ref = time.time() - t
        same = RustCFR(max_raises=0, bet_sizes=(0.33, 0.75), turn_sizes=(0.66,), turn_river_sizes=(0.75,))
        prod = RustCFR()
        stats = {"oop": [], "ip": []}
        expl_same, expl_prod, expl_ref, t_ours = [], [], [], 0.0
        for s, rf in zip(spots, ref):
            args = (s["board"], s["oop"], s["ip"], s["w_oop"], s["w_ip"], s["pot"], s["stack"])
            t = time.time()
            acts, freq, ev, expl = same.inspect(*args, player=0)
            t_ours += time.time() - t
            expl_same.append(100 * expl / s["pot"])
            expl_ref.append(100 * rf["exploitability_bb"] / s["pot"])
            expl_prod.append(100 * prod.inspect(*args, player=0)[3] / s["pot"])
            st0 = node_stats((acts, freq, ev), rf["oop"], s["oop"], s["w_oop"], s["pot"])
            if st0:
                stats["oop"].append(st0)
            if rf["ip"]:
                acts1, freq1, ev1, _ = same.inspect(*args, player=1, path=[(1, 0.0)])
                st1 = node_stats((acts1, freq1, ev1), rf["ip"], s["ip"], s["w_ip"], s["pot"])
                if st1:
                    st1["oop_check"] = st0["check_ref"] if st0 else 1.0     # как часто эталон вообще чекает сюда
                    stats["ip"].append(st1)
        med = statistics.median
        print(f"\n=== {'Ривер' if river else 'Тёрн'}: спотов {len(spots)} ===")
        print(f"время на спот: наш {t_ours / len(spots):.2f} с, эталон {t_ref / len(spots):.2f} с (эталон считает до 0.1% банка)")
        print(f"эксплуатируемость, % банка (медиана / максимум): наш, то же дерево {med(expl_same):.2f} / {max(expl_same):.2f}; "
              f"эталон {med(expl_ref):.2f} / {max(expl_ref):.2f}; наш, боевое дерево с рейзами {med(expl_prod):.2f} / {max(expl_prod):.2f}")
        for name, key in (("первый ход без позиции", "oop"), ("в позиции после чека", "ip")):
            st = stats[key]
            if not st:
                continue
            print(f"{name} (n={len(st)}):")
            print(f"  частота чека по диапазону: наш {100 * med([x['check_ours'] for x in st]):.0f}%, "
                  f"эталон {100 * med([x['check_ref'] for x in st]):.0f}% (медианы)")
            print(f"  расхождение частоты чека по рукам: медиана {100 * med([x['check_diff'] for x in st]):.1f} п.п., "
                  f"максимум {100 * max(x['check_diff'] for x in st):.1f} п.п.")
            print(f"  расхождение EV руки: медиана {med([x['ev_diff_pct'] for x in st]):.2f}% банка, "
                  f"максимум {max(x['ev_diff_pct'] for x in st):.2f}%")
            print(f"  расхождение EV диапазона: медиана {med([x['range_ev_diff_pct'] for x in st]):.2f}% банка, "
                  f"максимум {max(x['range_ev_diff_pct'] for x in st):.2f}%")
            if key == "ip":
                often = [x for x in st if x["oop_check"] >= 0.3]
                rare = [x for x in st if x["oop_check"] < 0.3]
                for label, part in (("где чек играется от 30% случаев", often), ("где чек играется реже", rare)):
                    if part:
                        print(f"    {label} (n={len(part)}): расхождение EV диапазона — медиана "
                              f"{med([x['range_ev_diff_pct'] for x in part]):.2f}%, максимум "
                              f"{max(x['range_ev_diff_pct'] for x in part):.2f}%; частоты чека — "
                              f"{100 * med([x['check_diff'] for x in part]):.1f} п.п.")


if __name__ == "__main__":
    main()
