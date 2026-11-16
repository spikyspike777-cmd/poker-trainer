"""Проверка порогов мультивей-эвристики симуляцией.

Герой играет по советам тренера за столом на шестерых против одних и тех же ботов на одних и тех же
раздачах; меняются только пороги для банков на трёх и более игроков. Сравнивается результат героя
в раздачах, где у него было хотя бы одно такое решение (в остальных раздачах игра совпадает).

    python tools/sim_multiway.py --hands 3000
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_coach import coach, params  # noqa: E402
from poker_coach.engine import Game, H  # noqa: E402

STYLES = ["station", "station", "lag", "station", "standard"]     # много коллов — больше банков на троих и более
BASE = {"mw_value": 0.55, "mw_raise": 0.65, "mw_call_margin": 0.0}     # пороги до проверки; сейчас mw_value = 0.50
CANDIDATES = {
    "ставить с эквити от 35%": {"mw_value": 0.35},
    "ставить с эквити от 45%": {"mw_value": 0.45},
    "ставить с эквити от 50%": {"mw_value": 0.50},
    "ставить с эквити от 65%": {"mw_value": 0.65},
    "рейзить с эквити от 55%": {"mw_raise": 0.55},
    "рейзить с эквити от 75%": {"mw_raise": 0.75},
    "коллировать шире (запас −5%)": {"mw_call_margin": -0.05},
    "коллировать уже (запас +5%)": {"mw_call_margin": 0.05},
}


def play(args) -> tuple[int, float, bool]:
    """Одна раздача с заданными порогами героя -> (номер, итог героя в bb, были ли у героя мультивей-решения)."""
    hand, override = args
    coach.SOLVER = None                               # один на один — эвристика: так симуляция в десятки раз быстрее
    g = Game.__new__(Game)
    g.rng = random.Random(1000 + hand)
    g.use_profile, g.adaptive, g.button = False, False, hand % 6
    g.setup(6, STYLES)
    multiway = False
    while not g.finished:
        if len(g.board) >= 3 and len(g.alive()) >= 3:
            multiway = True
        params.P.update(BASE)
        params.P.update(override)
        adv = g.advice(H)
        params.P.update(BASE)                         # боты всегда играют с базовыми порогами
        group = {"fold": "fold", "check": "check", "call": "call"}.get(adv.action, "raise")
        opts = [o for o in g.options() if o.kind == group] or g.options()
        if adv.action == "allin":
            opt = opts[-1]
        elif group == "raise" and adv.amount:
            opt = min(opts, key=lambda o: abs(o.to - adv.amount))
        else:
            opt = opts[0]
        g.hero_act(opt)
    return hand, g.net, multiway


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=3000)
    a = p.parse_args()
    with Pool(4) as pool:
        base = pool.map(play, [(h, {}) for h in range(a.hands)], chunksize=20)
        mw = [h for h, _net, m in base if m]
        base_net = {h: net for h, net, _m in base}
        print(f"раздач: {a.hands}; с мультивей-решением героя: {len(mw)} ({100 * len(mw) / a.hands:.0f}%)")
        print(f"базовые пороги: итог героя в этих раздачах {100 * statistics.mean(base_net[h] for h in mw):+.0f} bb на 100 раздач")
        for name, override in CANDIDATES.items():
            res = pool.map(play, [(h, override) for h in mw], chunksize=10)
            diff = [net - base_net[h] for h, net, _m in res]
            changed = sum(1 for d in diff if abs(d) > 1e-9)
            mean = statistics.mean(diff)
            se = statistics.pstdev(diff) / len(diff) ** 0.5
            print(f"{name:32} разница с базой {100 * mean:+6.0f} ± {100 * 2 * se:.0f} bb на 100 таких раздач "
                  f"(изменился итог в {changed} раздачах)")


if __name__ == "__main__":
    main()
