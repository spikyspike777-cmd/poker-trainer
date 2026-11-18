"""Отчёт по выгрузке tools/diag_modes.py."""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from math import comb


def sign_p(x: int, y: int) -> float:
    """Двусторонний точный критерий знаков: насколько вероятен такой перевес при равных режимах."""
    n = x + y
    k = min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

rows = [json.loads(l) for l in open(sys.argv[1])]
names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
ok = lambda r, n: r[n]["pred"] == r["gold"]
print(f"спотов: {len(rows)}\n")
print(f"{'конфигурация':12} {'всего':>7} {'флоп':>7} {'тёрн':>7} {'ривер':>7}   средняя частота главного действия")
for n in names:
    acc = 100 * sum(ok(r, n) for r in rows) / len(rows)
    by = {}
    for st in ("Flop", "Turn", "River"):
        rs = [r for r in rows if r["street"] == st]
        by[st] = 100 * sum(ok(r, n) for r in rs) / max(len(rs), 1)
    tops = [r[n]["top"] for r in rows if r[n].get("top") is not None]
    print(f"{n:12} {acc:6.1f}% {by['Flop']:6.1f}% {by['Turn']:6.1f}% {by['River']:6.1f}%   {100 * statistics.mean(tops):.0f}%")


def compare(a: str, b: str) -> None:
    diff = [r for r in rows if r[a]["pred"] != r[b]["pred"]]
    only_a = [r for r in diff if ok(r, a)]
    only_b = [r for r in diff if ok(r, b)]
    print(f"\n{a} против {b}: ответы различаются в {len(diff)} спотах; прав только {a}: {len(only_a)}, "
          f"только {b}: {len(only_b)}, оба неправы: {len(diff) - len(only_a) - len(only_b)}; "
          f"p = {sign_p(len(only_a), len(only_b)):.3f}")
    pat = Counter((r["gold"], r[a]["pred"], r[b]["pred"]) for r in diff)
    print("  самые частые (эталон, ответ " + a + ", ответ " + b + "):", pat.most_common(6))
    for name, part in ((f"где прав {a}", only_a), (f"где прав {b}", only_b)):
        if part:
            ga = statistics.mean(r[a]["gold_freq"] for r in part)
            gb = statistics.mean(r[b]["gold_freq"] for r in part)
            ta = statistics.mean(r[a]["top"] or 0 for r in part)
            tb = statistics.mean(r[b]["top"] or 0 for r in part)
            print(f"  {name}: частота эталонного действия {a} {100 * ga:.0f}%, {b} {100 * gb:.0f}%; "
                  f"частота главного действия {a} {100 * ta:.0f}%, {b} {100 * tb:.0f}%")
    st = Counter(r["street"] for r in diff)
    print("  по улицам:", dict(st), "| отвечая на ставку:", sum(r["facing"] for r in diff))


for a, b in (("S", "S_seed7"), ("A", "A_seed7"), ("S_seed7", "A_seed7"), ("S", "A"), ("S", "A_lin"), ("S", "S_quad"), ("S", "A_chainS"), ("S", "S_chainA"), ("S_rules", "A_rules")):
    if a in names and b in names:
        compare(a, b)
