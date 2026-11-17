"""Прогон тренера на тестовых наборах PokerBench (arXiv:2501.08328).

    python -m poker_coach.bench                 # префлоп 1k + выборка 2000 постфлоп-спотов
    python -m poker_coach.bench --n 0           # весь постфлоп (10k, долго)
    python -m poker_coach.bench --no-solver     # только эвристика

Метрики как в статье: AA — совпал тип действия (fold/call/check/bet/raise),
EM — совпал и размер (наш размер приводится к ближайшему из доступных в споте).
"""
from __future__ import annotations

import argparse
import ast
import csv
import random
import urllib.request
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from . import charts, coach
from .state import POSITIONS, POSTFLOP_ORDER, GameState, PreflopLine as L

DATA = Path(__file__).resolve().parent.parent / "data" / "pokerbench"
URL = "https://huggingface.co/datasets/RZ412/PokerBench/resolve/main/"
USE_HISTORY = True
PRE = "preflop_1k_test_set_game_scenario_information.csv"
POST = "postflop_10k_test_set_game_scenario_information.csv"


def load(name: str) -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    if not path.exists():
        urllib.request.urlretrieve(URL + name, path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _cards(s: str) -> list[str]:
    return [s[i:i + 2] for i in range(0, len(s), 2)]


def _parse_line(line: str) -> list[tuple[str, str]]:
    t = [x for x in line.split("/") if x]
    return list(zip(t[::2], t[1::2]))


# ---------- постфлоп ----------

def postflop_state(row: dict) -> GameState:
    acts = _parse_line(row["preflop_action"])
    raises = [(p, float(a[:-2])) for p, a in acts if a.endswith("bb")]
    players = list(dict.fromkeys(p for p, a in acts if a != "fold"))
    opener = raises[0][0]
    other = next(p for p in players if p != opener)
    ip = max(opener, other, key=POSTFLOP_ORDER.index)
    oop = other if ip == opener else opener
    hero = ip if row["hero_position"] == "IP" else oop
    villain = oop if hero == ip else ip
    opened = hero == opener
    n = len(raises)
    if n == 1:
        line = L.HERO_OPEN_VILLAIN_CALL if opened else L.VILLAIN_OPEN_HERO_CALL
    elif n == 2:
        line = L.VILLAIN_3BET_HERO_CALL if opened else L.HERO_3BET_VILLAIN_CALL
    else:
        line = L.VILLAIN_3BET_HERO_CALL if opened else L.HERO_3BET_VILLAIN_CALL
    board = _cards(row["board_flop"])
    street = row["evaluation_at"]
    if street in ("Turn", "River"):
        board += _cards(row["board_turn"])
    if street == "River":
        board += _cards(row["board_river"])
    me = row["hero_position"]
    spent, inv, history, st = raises[-1][1], {"IP": 0.0, "OOP": 0.0}, [], "flop"
    amounts: list[float] = []
    for tok in row["postflop_action"].split("/"):
        if tok == "dealcards":
            spent += inv[me]
            inv = {"IP": 0.0, "OOP": 0.0}
            st = "turn" if st == "flop" else "river"
            continue
        parts = tok.split("_")
        if len(parts) < 2 or parts[0] not in inv:
            continue                                   # карта после dealcards
        who, kind = parts[0], parts[1].lower()
        amount = 0.0
        if kind in ("bet", "raise"):
            inv[who] = amount = float(parts[2])
        elif kind == "call":
            inv[who] = max(inv.values())
        history.append((st, "hero" if who == me else "villain", kind))
        amounts.append(amount)
    opp = "OOP" if me == "IP" else "IP"
    to_call = max(0.0, inv[opp] - inv[me])
    stack = max(100 - spent - inv[me], to_call, 0.5)
    pot = max(float(row["pot_size"]), to_call + 0.5)
    kw = dict(hero_cards=_cards(row["holding"]), board=board, hero_pos=hero, villain_pos=villain,
              preflop_line=line, pot=pot, to_call=to_call, stack=stack, hero_invested=inv[me])
    if USE_HISTORY:
        kw["history"] = history
        kw["history_amounts"] = amounts
    return GameState(**kw)


def _nearest(amount: float | None, options: list[float]) -> float | None:
    if not options:
        return None
    if amount is None:
        return max(options)                            # олл-ин → самый крупный размер
    return min(options, key=lambda x: abs(x - amount))


def _label(a, row: dict) -> tuple[str, str]:
    """Совет тренера -> (тип действия, точная метка в формате датасета)."""
    moves = ast.literal_eval(row["available_moves"])
    kinds = {m.split()[0].lower() for m in moves}
    kind = a.action
    if kind in ("bet", "raise", "allin"):
        kind = "bet" if "bet" in kinds else "raise" if "raise" in kinds else "call" if "call" in kinds else "check"
    if kind not in kinds:
        kind = "check" if "check" in kinds else "fold" if kind == "check" else "call"
    if kind in ("bet", "raise"):
        sizes = [float(m.split()[1]) for m in moves if m.lower().startswith(kind)]
        amt = None if a.action == "allin" else a.amount
        return kind, f"{kind} {_nearest(amt, sizes):g}"
    return kind, kind


def postflop_predict(row: dict) -> tuple[str, str]:
    return _label(coach.analyze(postflop_state(row)), row)


OPTS: dict = {}       # настройки прогона: передаются в рабочие процессы


def _post_job(args):
    row, o = args
    from .params import P
    USE_HISTORY_set(o)
    P["table_ranges"] = 0 if o.get("chart_ranges") else P["table_ranges"]
    P["solved_ranges"] = 1 if o.get("solved") else 0
    P.update(o.get("params", {}))
    if not o.get("solver", True):
        coach.SOLVER = None
    elif coach.SOLVER is not None:
        for k, v in o.get("solver_args", {}).items():
            setattr(coach.SOLVER, k, tuple(v) if isinstance(v, list) else v)
        if o.get("params", {}).get("value_net") and getattr(coach.SOLVER, "value_net", 1) is None:
            from . import valuenet
            coach.SOLVER.value_net = valuenet.load()
        if o.get("no_flop") and hasattr(coach.SOLVER, "solve_flop"):
            coach.SOLVER.solve_flop = False
    top, extra = None, None
    gold = row["correct_decision"].lower()
    g = gold.split()
    try:
        st = postflop_state(row)
        a = coach.analyze(st)
        kind, label = _label(a, row)
        top = a.mix[0][2] if a.mix else None
        if a.all_actions:
            extra = _features(a, st, row, g)
    except Exception as e:                             # спот не разобрался — считаем ошибкой
        kind, label = "error", f"error: {type(e).__name__}: {e}"
    gold_label = f"{g[0]} {float(g[1]):g}" if len(g) > 1 else gold
    return row["evaluation_at"], g[0], kind, label == gold_label, label, top, extra


def _features(a, st, row: dict, g: list[str]) -> dict:
    """Признаки спота + насколько действие эталона хуже лучшего по нашему солверу."""
    from . import calibrate
    moves = {m.split()[0].lower() for m in ast.literal_eval(row["available_moves"])}
    feats, best, top_ev, freq = calibrate.features(a, st, moves)
    gold_ev = best.get(g[0])
    return dict(feats, gold=g[0], gold_gap_bb=None if gold_ev is None else top_ev - gold_ev,
                gold_gap_pot=None if gold_ev is None else (top_ev - gold_ev) / st.pot, gold_freq=freq.get(g[0], 0.0))


def USE_HISTORY_set(o) -> None:
    global USE_HISTORY
    USE_HISTORY = o.get("history", True)


def run_postflop(n: int, opts: dict, seed: int = 0, procs: int = 4):
    rows = load(POST) if not opts.get("train") else _train_rows()
    if opts.get("street"):
        rows = [r for r in rows if r["evaluation_at"] == opts["street"]]
    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    with Pool(procs) as pool:
        return pool.map(_post_job, [(r, opts) for r in rows], chunksize=4)


def _train_rows() -> list[dict]:
    """Случайные 20 000 строк обучающей части постфлопа (для выбора настроек без подгонки под тест)."""
    from .tune import TRAIN, load_train
    fit, hold = load_train(10000, seed=11)
    return fit + hold


# ---------- префлоп ----------

def preflop_predict(row: dict) -> tuple[str, float | None, bool]:
    """-> (fold/call/check/raise/allin, размер рейза, стандартная ли это линия для ручных чартов)."""
    hero, cards = row["hero_pos"], _cards(row["hero_holding"])
    acts = _parse_line(row["prev_line"])
    raises = [(p, 100.0 if a == "allin" else float(a[:-2])) for p, a in acts if a.endswith("bb") or a == "allin"]
    allin = any(a == "allin" for _p, a in acts)
    last_i = max((i for i, (_p, a) in enumerate(acts) if a.endswith("bb") or a == "allin"), default=-1)
    callers = [p for p, a in acts[last_i + 1:] if a == "call"]
    limpers = [p for p, a in acts if a == "call"] if not raises else []
    blind = {"SB": 0.5, "BB": 1.0}.get(hero, 0.0)
    hero_in = max([s for p, s in raises if p == hero], default=blind)
    n = len(raises)
    line, villain = L.OTHER, None
    if n == 0 and not limpers and hero != "BB":
        line = L.UNOPENED
    elif not allin and not callers and n in (1, 2, 3):
        last = raises[-1][0]
        if n == 1 and POSITIONS.index(last) < POSITIONS.index(hero):
            line, villain = L.VS_OPEN, last
        elif n == 2 and raises[0][0] == hero and POSITIONS.index(hero) < POSITIONS.index(last):
            line, villain = L.VS_3BET, last
        elif n == 3 and raises[1][0] == hero and raises[0][0] == last and POSITIONS.index(last) < POSITIONS.index(hero):
            line, villain = L.VS_4BET, last
    to_call = 0.0 if line == L.UNOPENED else max((raises[-1][1] if raises else 1.0) - hero_in, 0.0)
    pot = float(row["pot_size"])
    st = GameState(hero_cards=cards, hero_pos=hero, villain_pos=villain, preflop_line=line,
                   pot=max(pot, to_call + 0.1), to_call=to_call, stack=max(100 - hero_in, 0.5),
                   hero_invested=0, preflop_actions=row["prev_line"])
    a = coach.analyze(st)
    kind = "raise" if a.action in ("raise", "allin") else a.action
    return kind, a.amount, line != L.OTHER


def run_preflop():
    out = []
    for row in load(PRE):
        gold = row["correct_decision"]
        gkind = "raise" if gold.endswith("bb") or gold == "allin" else gold
        moves = ast.literal_eval(row["available_moves"])
        try:
            kind, amt, covered = preflop_predict(row)
        except Exception:
            kind, amt, covered = "error", None, False
        if kind == "raise":
            sizes = [float(m[:-2]) for m in moves if m.endswith("bb")]
            if not sizes and "allin" in moves:
                label = "allin"
            elif not sizes:
                kind = label = "call" if "call" in moves else "check"
            else:
                label = f"{_nearest(amt, sizes)}bb"
        else:
            if kind == "fold" and "check" in moves and "fold" not in moves:
                kind = "check"
            label = kind
        out.append((covered, gkind, kind, label == gold))
    return out


# ---------- отчёт ----------

def _pct(x: int, n: int) -> str:
    return f"{100 * x / n:5.1f}%" if n else "   — "


def report(pre, post) -> str:
    lines = ["PokerBench, тестовые наборы", ""]
    if pre:
        n = len(pre)
        cov = [r for r in pre if r[0]]
        lines += [f"Префлоп ({n} спотов)",
                  f"  AA {_pct(sum(g == k for _c, g, k, _e in pre), n)}   EM {_pct(sum(e for *_x, e in pre), n)}",
                  f"  стандартные линии (открытие, против опена, 3-бета, 4-бета): {len(cov)} спотов, "
                  f"AA {_pct(sum(g == k for _c, g, k, _e in cov), len(cov))}; "
                  f"мультивей, лимпы, олл-ины: {n - len(cov)} спотов, "
                  f"AA {_pct(sum(g == k for c, g, k, _e in pre if not c), n - len(cov))}", ""]
    if post:
        n = len(post)
        lines.append(f"Постфлоп ({n} спотов)")
        lines.append(f"  AA {_pct(sum(r[1] == r[2] for r in post), n)}   EM {_pct(sum(r[3] for r in post), n)}")
        by = defaultdict(list)
        for r in post:
            by[r[0]].append(r)
        for st in ("Flop", "Turn", "River"):
            rs = by.get(st, [])
            lines.append(f"  {st:5} n={len(rs):5}  AA {_pct(sum(r[1] == r[2] for r in rs), len(rs))}")
        gold = Counter(r[1] for r in post)
        lines.append("  по правильному действию: " + ", ".join(
            f"{g} {_pct(sum(1 for r in post if r[1] == g and r[2] == g), c).strip()}" for g, c in gold.most_common()))
        solved = [r for r in post if r[5] is not None]
        if solved:
            sure = [r for r in solved if r[5] >= 0.8]
            mixed = [r for r in solved if r[5] < 0.8]
            lines.append(f"  решено солвером: {len(solved)} спотов; где он уверен (частота действия от 80%): "
                         f"n={len(sure)}, AA {_pct(sum(r[1] == r[2] for r in sure), len(sure)).strip()}; "
                         f"где смешивает: n={len(mixed)}, AA {_pct(sum(r[1] == r[2] for r in mixed), len(mixed)).strip()}")
            rest = [r for r in post if r[5] is None]
            lines.append(f"  эвристикой: n={len(rest)}, AA {_pct(sum(r[1] == r[2] for r in rest), len(rest)).strip()}")
            ex = [r for r in solved if r[6]]
            soft = lambda lim: sum(1 for r in ex if r[1] == r[2] or (
                r[6]["gold_gap_pot"] is not None and r[6]["gold_gap_pot"] <= lim))
            lines.append(f"  мягкая точность (совпало, либо действие эталона по нашему солверу теряет не больше "
                         f"1% банка): {_pct(soft(0.01), len(ex)).strip()}; не больше 3% банка: {_pct(soft(0.03), len(ex)).strip()}")
            wrong = [r for r in ex if r[1] != r[2] and r[6]["gold_gap_pot"] is not None]
            if wrong:
                gaps = sorted(r[6]["gold_gap_pot"] for r in wrong)
                lines.append(f"  в несовпавших спотах действие эталона теряет по нашему солверу: медиана "
                             f"{100 * gaps[len(gaps) // 2]:.1f}% банка; средняя частота, которую мы ему даём: "
                             f"{100 * sum(r[6]['gold_freq'] for r in wrong) / len(wrong):.0f}%")
        errs = sum(1 for r in post if r[2] == "error")
        if errs:
            ex = next(r[4] for r in post if r[2] == "error")
            lines.append(f"  не разобрано спотов: {errs} (например: {ex[:150]})")
    return "\n".join(lines)


def main() -> None:
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="размер выборки постфлопа, 0 — все")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", action="store_true", help="постфлоп из обучающей части (для выбора настроек)")
    p.add_argument("--street", choices=["Flop", "Turn", "River"])
    p.add_argument("--no-solver", action="store_true")
    p.add_argument("--no-flop", action="store_true", help="флоп — эвристикой")
    p.add_argument("--no-solved-ranges", action="store_true", help="диапазоны по правилам сужения, без решения прошлых улиц")
    p.add_argument("--chart-ranges", action="store_true", help="стартовые диапазоны только по ручным чартам")
    p.add_argument("--no-history", action="store_true", help="не учитывать историю ставок")
    p.add_argument("--no-table", action="store_true", help="префлоп только по ручным чартам")
    p.add_argument("--solver-args", default="{}", help='настройки солвера в JSON, например {"turn_sizes": [0.66]}')
    p.add_argument("--calibrate", action="store_true", help="выбирать действие моделью поверх солвера")
    p.add_argument("--dump", help="записать признаки спотов (выход солвера) в файл JSONL")
    p.add_argument("--set", action="append", default=[], help="параметр из params.py: имя=значение")
    p.add_argument("--skip-postflop", action="store_true")
    p.add_argument("--skip-preflop", action="store_true")
    a = p.parse_args()
    from .params import P
    opts = dict(solver=not a.no_solver, history=not a.no_history, chart_ranges=a.chart_ranges,
                solved=bool(P["solved_ranges"]) and not a.no_solved_ranges, no_flop=a.no_flop, train=a.train,
                street=a.street, solver_args=json.loads(a.solver_args),
                params={k: float(v) for k, v in (x.split("=") for x in a.set)})
    if a.calibrate:
        opts["params"]["calibrate"] = 1
    coach.USE_GTO_TABLE = not a.no_table
    pre = [] if a.skip_preflop else run_preflop()
    post = [] if a.skip_postflop else run_postflop(a.n, opts, seed=a.seed)
    print(report(pre, post))
    if a.dump:
        with open(a.dump, "w") as f:
            for r in post:
                if r[6]:
                    f.write(json.dumps(dict(r[6], pred=r[2])) + "\n")


if __name__ == "__main__":
    main()
