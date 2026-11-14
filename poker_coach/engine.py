"""Раздача один на один против бота (остальные четыре игрока сбросили).

Бот играет по тем же чартам и правилам, что и тренер. Каждое решение героя
сравнивается с советом тренера. Все суммы — в bb; «to» — сколько всего вложено на улице.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import eval7

from . import calc, charts
from .coach import Analysis, analyze
from .state import POSITIONS, POSTFLOP_ORDER, RANKS, SUITS, GameState, PreflopLine as L, to_eval7

H, V = "H", "V"
STREETS = ["preflop", "flop", "turn", "river"]
STREET_RU = {"preflop": "Префлоп", "flop": "Флоп", "turn": "Тёрн", "river": "Ривер"}
_SHOVE_RANGE = "KK+, AKs"


@dataclass
class Option:
    label: str
    kind: str            # fold / check / call / raise
    to: float = 0.0


@dataclass
class Decision:
    street: str
    chosen: str
    advice: Analysis
    matched: bool


def _group(kind: str) -> str:
    return {"fold": "fold", "check": "passive", "call": "passive"}.get(kind, "aggressive")


class Game:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.new_hand()

    # ---------- раздача ----------

    def new_hand(self) -> None:
        r = self.rng
        i, j = sorted(r.sample(range(6), 2))
        first, second = POSITIONS[i], POSITIONS[j]
        self.pos = {H: first, V: second} if r.random() < 0.5 else {H: second, V: first}
        self.opener = H if self.pos[H] == first else V       # кто ходит первым на префлопе
        deck = [a + b for a in RANKS for b in SUITS]
        r.shuffle(deck)
        self.cards = {H: deck[:2], V: deck[2:4]}
        if self.opener == V:                                  # бот-первый всегда входит в игру
            rng_combos = charts.combos(charts.RFI[self.pos[V]])
            ok = [c for c in rng_combos if str(c[0]) not in self.cards[H] and str(c[1]) not in self.cards[H]]
            self.cards[V] = [str(x) for x in r.choice(ok)]
            deck = [c for c in deck if c not in self.cards[H] + self.cards[V]]
            self._runout = deck[:5]
        else:
            self._runout = deck[4:9]
        self.board: list[str] = []
        blind = {"SB": 0.5, "BB": 1.0}
        self.inv = {p: blind.get(self.pos[p], 0.0) for p in (H, V)}
        self.stack = {p: 100.0 - self.inv[p] for p in (H, V)}
        self.dead = 1.5 - sum(self.inv.values())              # блайнды сбросивших игроков
        self.prev = 0.0                                       # вложено на прошлых улицах
        self.street = "preflop"
        self.raises = 0
        self.to_act = self.opener
        self.checked = False
        self.finished = False
        self.final_pot = 0.0
        self.result = ""
        self.net = 0.0
        self.history: list[tuple[str, str, str]] = []     # (улица, игрок, действие) на постфлопе
        self.log: list[str] = [f"Вы на {self.pos[H]}, оппонент на {self.pos[V]}. Остальные сбросили."]
        self.decisions: list[Decision] = []
        self.last: Decision | None = None
        self._run_bot()

    @property
    def pot(self) -> float:
        return round(self.dead + self.prev + self.inv[H] + self.inv[V], 2)

    def _other(self, p: str) -> str:
        return V if p == H else H

    def _ip(self, p: str) -> bool:
        return POSTFLOP_ORDER.index(self.pos[p]) > POSTFLOP_ORDER.index(self.pos[self._other(p)])

    # ---------- совет тренера для игрока p ----------

    def _line(self, p: str) -> L:
        n, opened = self.raises, p == self.opener
        if self.street == "preflop":
            if n == 0:
                return L.UNOPENED
            if n >= 4:                       # против олл-ина на 5-бет; диапазон подменяется в advice()
                return L.VS_3BET if opened else L.VS_4BET
            return {1: L.VS_OPEN, 2: L.VS_3BET, 3: L.VS_4BET}[n]
        if n == 1:
            return L.HERO_OPEN_VILLAIN_CALL if opened else L.VILLAIN_OPEN_HERO_CALL
        if n == 2:
            return L.VILLAIN_3BET_HERO_CALL if opened else L.HERO_3BET_VILLAIN_CALL
        # банк с 4-бетом: приближаем диапазонами 3-бет-банка
        return L.VILLAIN_3BET_HERO_CALL if opened else L.HERO_3BET_VILLAIN_CALL

    def state(self, p: str = H) -> GameState:
        o, line = self._other(p), self._line(p)
        unopened = line == L.UNOPENED
        return GameState(
            hero_cards=self.cards[p], board=self.board, hero_pos=self.pos[p],
            villain_pos=None if unopened else self.pos[o], preflop_line=line, pot=self.pot,
            to_call=0 if unopened else max(0.0, round(self.inv[o] - self.inv[p], 2)), stack=self.stack[p],
            hero_invested=0 if self.street == "preflop" else self.inv[p],
            history=[(st, "hero" if who == p else "villain", k) for st, who, k in self.history])

    def advice(self, p: str = H) -> Analysis:
        st, o = self.state(p), self._other(p)
        a = analyze(st)
        if st.to_call > 0 and self.stack[o] == 0:            # оппонент в олл-ине: только колл или фолд
            if self.street == "preflop":
                hero = to_eval7(self.cards[p])
                a.equity = calc.equity_vs_range(hero, charts.combos(_SHOVE_RANGE), [])
                a.villain.combos = calc.live_combos(charts.combos(_SHOVE_RANGE), hero)
                a.villain.label = "олл-ин на префлопе (KK+, AKs)"
            call = a.equity >= a.required
            a.action, a.amount, a.sizing = ("call", st.to_call, "") if call else ("fold", None, "")
            a.source = "эквити против шансов банка"
            a.reasons = [f"Оппонент в олл-ине. Ваше эквити против его диапазона — {a.equity:.0%}, "
                         f"для колла нужно {a.required:.0%}: " + ("колл выгоден." if call else "колл в минус.")]
        return a

    # ---------- действия ----------

    def options(self) -> list[Option]:
        if self.finished or self.to_act != H:
            return []
        to_call = round(self.inv[V] - self.inv[H], 2)
        opts: list[Option] = []
        pre = self.street == "preflop"
        if to_call > 0 or (pre and self.raises == 0):
            opts.append(Option("Фолд", "fold"))
        if to_call > 0 and not (pre and self.raises == 0):
            pay = min(to_call, self.stack[H])
            opts.append(Option(f"Колл {pay:g}", "call"))
        elif not pre:
            opts.append(Option("Чек", "check"))
        if self.stack[V] == 0 or self.stack[H] <= to_call:
            return opts
        allin_to = self.inv[H] + self.stack[H]
        sizes: list[tuple[str, float]] = []
        if pre:
            cur, n = self.inv[V], self.raises
            if n == 0:
                sizes.append(("Рейз", charts.OPEN_SIZE[self.pos[H]]))
            elif n == 1:
                sizes.append(("3-бет", round(cur * (3 if self._ip(H) else 4), 1)))
            elif n == 2:
                sizes.append(("4-бет", round(cur * 2.3, 1)))
        elif to_call == 0:
            sizes += [(f"Бет {int(f * 100)}%", round(self.pot * f, 1)) for f in (0.33, 0.5, 0.75, 1.25)]
        else:
            sizes.append(("Рейз 3×", round(self.inv[V] * 3, 1)))
        for name, to in sizes:
            if to < allin_to * 0.8:
                opts.append(Option(f"{name} {to:g}", "raise", to))
        opts.append(Option(f"Олл-ин {allin_to:g}", "raise", allin_to))
        return opts

    def hero_act(self, opt: Option) -> None:
        adv = self.advice(H)
        matched = _group(opt.kind) == _group(adv.action)
        if adv.mix:      # солвер смешивает: решение верное, если он сам так играет хотя бы в 15% случаев
            matched = sum(f for k, _a, f, _e in adv.mix if _group(k) == _group(opt.kind)) >= 0.15
        d = Decision(self.street, opt.label, adv, matched)
        self.decisions.append(d)
        self.last = d
        self._apply(H, opt.kind, opt.to)
        self._run_bot()

    def _run_bot(self) -> None:
        while not self.finished and self.to_act == V:
            a = self.advice(V)
            to_call = self.inv[H] - self.inv[V]
            kind, to = a.action, 0.0
            if a.mix:
                kind, amt, _f, _e = self.rng.choices(a.mix, weights=[m[2] for m in a.mix])[0]
                a.amount = amt
            if kind in ("bet", "raise", "allin"):
                to = self.inv[V] + self.stack[V] if kind == "allin" else max(a.amount, 2 * self.inv[H])
                kind = "raise"
                if self.stack[H] == 0 or self.stack[V] <= to_call:
                    kind = "call"
            if kind == "fold" and to_call <= 0:
                kind = "check"
            if kind == "check" and to_call > 0:
                kind = "fold"
            if kind == "call" and to_call <= 0:
                kind = "check"
            self._apply(V, kind, to)

    def _apply(self, p: str, kind: str, to: float) -> None:
        o, who = self._other(p), "Вы" if p == H else "Оппонент"
        pre = STREET_RU[self.street]
        if kind == "fold":
            self.log.append(f"{pre}: {who} — фолд.")
            self._finish(winner=o)
            return
        if self.street != "preflop" and kind != "fold":
            first_bet = self.inv[o] == 0 and self.inv[p] == 0
            self.history.append((self.street, p, "bet" if kind == "raise" and first_bet else kind))
        if kind == "check":
            self.log.append(f"{pre}: {who} — чек.")
            if self.checked:
                self._next_street()
            else:
                self.checked, self.to_act = True, o
            return
        if kind == "call":
            pay = min(self.inv[o] - self.inv[p], self.stack[p])
            self._pay(p, pay)
            if self.inv[o] > self.inv[p]:                    # колл неполным стеком: лишнее возвращается
                back = self.inv[o] - self.inv[p]
                self.inv[o] -= back
                self.stack[o] += back
            self.log.append(f"{pre}: {who} — колл {pay:g}.")
            self._next_street()
            return
        to = min(to, self.inv[p] + self.stack[p])
        self._pay(p, to - self.inv[p])
        allin = " (олл-ин)" if self.stack[p] == 0 else ""
        word = "бет" if self.street != "preflop" and self.inv[o] == 0 else "рейз до"
        self.log.append(f"{pre}: {who} — {word} {to:g}{allin}.")
        if self.street == "preflop":
            self.raises += 1
        self.checked, self.to_act = False, o

    def _pay(self, p: str, amount: float) -> None:
        amount = round(amount, 2)
        self.inv[p] = round(self.inv[p] + amount, 2)
        self.stack[p] = round(self.stack[p] - amount, 2)

    def _next_street(self) -> None:
        self.prev += self.inv[H] + self.inv[V]
        self.inv = {H: 0.0, V: 0.0}
        self.checked = False
        allin = self.stack[H] == 0 or self.stack[V] == 0
        while True:
            if self.street == "river":
                self._showdown()
                return
            self.street = STREETS[STREETS.index(self.street) + 1]
            self.board = self._runout[:{"flop": 3, "turn": 4, "river": 5}[self.street]]
            self.log.append(f"{STREET_RU[self.street]}: {' '.join(self.board)}. Банк {self.pot:g}.")
            if not allin:
                break
        self.to_act = V if self._ip(H) else H

    def _showdown(self) -> None:
        b = to_eval7(self.board)
        hv, vv = (eval7.evaluate(to_eval7(self.cards[p]) + b) for p in (H, V))
        self.log.append(f"Вскрытие: у оппонента {' '.join(self.cards[V])} "
                        f"({calc.hand_name(to_eval7(self.cards[V]), b)}), "
                        f"у вас {calc.hand_name(to_eval7(self.cards[H]), b)}.")
        self._finish(winner=H if hv > vv else V if vv > hv else None)

    def _finish(self, winner: str | None) -> None:
        pot = self.final_pot = self.pot
        self.prev, self.dead, self.inv = 0.0, 0.0, {H: 0.0, V: 0.0}
        if winner is None:
            self.stack[H] += pot / 2
            self.stack[V] += pot / 2
        else:
            self.stack[winner] += pot
        self.net = round(self.stack[H] - 100, 2)
        self.finished = True
        who = {H: "Вы выиграли банк", V: "Оппонент выиграл банк", None: "Банк поделён"}[winner]
        self.result = f"{who} {pot:g} bb. Ваш итог: {self.net:+g} bb."
        self.log.append(self.result)
