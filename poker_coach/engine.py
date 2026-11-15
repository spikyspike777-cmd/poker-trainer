"""Стол на 2–6 игроков: герой против ботов с разными характерами.

Каждое решение героя сравнивается с советом тренера. Боты принимают решения той же
логикой тренера (со своей стороны стола), искажённой характером. Все суммы — в bb;
«to» — сколько всего вложено игроком на текущей улице. В начале каждой раздачи у всех 100 bb,
поэтому побочных банков не бывает.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import eval7

from . import bots, calc, charts, preflop_gto
from .coach import Analysis, analyze, matches
from .profile import BASE_VPIP, PlayerStats
from .state import POSITIONS, RANKS, SUITS, GameState, PreflopLine as L, classify_preflop, to_eval7

H = 0                                   # индекс героя за столом
STREETS = ["preflop", "flop", "turn", "river"]
STREET_RU = {"preflop": "Префлоп", "flop": "Флоп", "turn": "Тёрн", "river": "Ривер"}
# За столами на 7–9 игроков добавляются ранние позиции. Чартов и таблицы солвера для них нет,
# поэтому тренер считает их как UTG (самый узкий диапазон стола 6-max).
EARLY = ["EP1", "EP2", "EP3"]
SEATS_BY_N = {2: ["BTN", "BB"], 3: ["BTN", "SB", "BB"], 4: ["CO", "BTN", "SB", "BB"],
              5: ["HJ", "CO", "BTN", "SB", "BB"], 6: POSITIONS,
              7: EARLY[2:] + POSITIONS, 8: EARLY[1:] + POSITIONS, 9: EARLY + POSITIONS}
POSTFLOP_FULL = ["SB", "BB"] + EARLY + ["UTG", "HJ", "CO", "BTN"]
_NAMES = {7: ["UTG", "LJ"], 8: ["UTG", "UTG+1", "LJ"], 9: ["UTG", "UTG+1", "UTG+2", "LJ"]}


def model_pos(pos: str) -> str:
    """Позиция, по которой тренер ищет чарты: ранние позиции больших столов играются как UTG."""
    return "UTG" if pos in EARLY else pos


def display_pos(pos: str, n: int) -> str:
    """Привычное название позиции за столом на n игроков."""
    seats = SEATS_BY_N[n]
    names = _NAMES.get(n, [])
    i = seats.index(pos)
    return names[i] if i < len(names) else pos
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


@dataclass
class Player:
    name: str
    style: bots.Style
    stats: PlayerStats = field(default_factory=PlayerStats)     # что герой заметил за этим ботом
    pos: str = ""                        # внутреннее имя места (уникально за столом)
    label: str = ""                      # название позиции для показа
    cards: list[str] = field(default_factory=list)
    stack: float = 100.0
    inv: float = 0.0                     # вложено на текущей улице
    folded: bool = False


class Game:
    def __init__(self, n_players: int = 6, styles: str | list[str] = "standard",
                 rng: random.Random | None = None, use_profile: bool = True, adaptive: bool = False):
        self.rng = rng or random.Random()
        self.use_profile = use_profile          # учитывает ли тренер профили ботов в советах герою
        self.adaptive = adaptive                # подстраиваются ли боты под героя
        self.hero_stats = PlayerStats()         # что боты заметили за героем
        self.button = 0
        self.setup(n_players, styles)

    def setup(self, n_players: int, styles: str | list[str]) -> None:
        """Пересадить стол: число игроков и характеры ботов ('random' — случайные)."""
        if styles == "random":
            styles = [self.rng.choice(list(bots.STYLES)) for _ in range(n_players - 1)]
        elif isinstance(styles, str):
            styles = [styles] * (n_players - 1)
        self.n = n_players
        self.players = [Player("Вы", bots.STYLES["standard"])] + [
            Player(f"Бот {i + 1}", bots.STYLES[s]) for i, s in enumerate(styles)]
        self.hero_stats = PlayerStats()
        self.new_hand()

    # ---------- раздача ----------

    def new_hand(self) -> None:
        for _ in range(30):                      # пересдаём, если раздача кончилась без решения героя
            self._deal()
            self._advance()
            if not self.finished or self.decisions:
                return

    def _deal(self) -> None:
        r, n = self.rng, self.n
        self.button = (self.button + 1) % n
        seats = SEATS_BY_N[n]
        ring = seats[seats.index("BTN") + 1:] + seats[:seats.index("BTN") + 1]    # позиции по кругу после баттона
        deck = [a + b for a in RANKS for b in SUITS]
        r.shuffle(deck)
        blind = {"SB": 0.5, "BB": 1.0}
        if n == 2:
            blind = {"BTN": 0.5, "BB": 1.0}
        for k, pos in enumerate(ring):
            p = self.players[(self.button + 1 + k) % n]
            p.pos, p.label, p.cards = pos, display_pos(pos, n), [deck.pop(), deck.pop()]
            p.inv = blind.get(pos, 0.0)
            p.stack, p.folded = 100.0 - p.inv, False
        self._runout = deck[:5]
        self.board: list[str] = []
        self.prev = 0.0                           # вложено на прошлых улицах
        self.street = "preflop"
        self.cur = 1.0                            # текущая ставка на улице
        self.raises = 0                           # число рейзов на префлопе
        self.pre_raisers: list[int] = []
        self.pre_tokens: list[str] = []           # префлоп-линия в нотации PokerBench
        self.voluntary: set[int] = set()          # кто уже добровольно вкладывал деньги
        self.seen_pre: set[int] = set()           # чьё первое префлоп-решение уже записано в профиль
        self.history: list[tuple[str, int, str]] = []     # (улица, игрок, действие) на постфлопе
        self.history_amounts: list[float] = []            # до какой суммы ставка или рейз (0 для чека и колла)
        self.pending = self._order()              # кому ещё ходить на этой улице
        self.finished, self.result, self.net, self.final_pot = False, "", 0.0, 0.0
        self.shown: set[int] = set()
        seats_txt = ", ".join(f"{p.name} — {p.label}" for p in self.players)
        self.log: list[str] = [f"Игроков: {n}. {seats_txt}."]
        self.decisions: list[Decision] = []
        self.last: Decision | None = None

    def _order(self) -> list[int]:
        """Игроки, которые ещё в раздаче и не в олл-ине, в порядке хода на текущей улице."""
        order = SEATS_BY_N[self.n] if self.street == "preflop" else [p for p in POSTFLOP_FULL if p in SEATS_BY_N[self.n]]
        by_pos = {p.pos: i for i, p in enumerate(self.players)}
        return [by_pos[pos] for pos in order if not self.players[by_pos[pos]].folded
                and self.players[by_pos[pos]].stack > 0]

    @property
    def pot(self) -> float:
        return round(self.prev + sum(p.inv for p in self.players), 2)

    @property
    def to_act(self) -> int | None:
        return self.pending[0] if self.pending and not self.finished else None

    def alive(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if not p.folded]

    def _ip(self, i: int, j: int) -> bool:
        return POSTFLOP_FULL.index(self.players[i].pos) > POSTFLOP_FULL.index(self.players[j].pos)

    # ---------- состояние и совет для игрока i ----------

    def _relation(self, me: int, opp: int) -> L:
        """Префлоп-линия между двумя игроками на постфлопе (с точки зрения `me`)."""
        last = self.pre_raisers[-1] if self.pre_raisers else None
        n = len(self.pre_raisers)
        if last == me:
            return L.HERO_OPEN_VILLAIN_CALL if n == 1 else L.HERO_3BET_VILLAIN_CALL
        if last == opp:
            return L.VILLAIN_OPEN_HERO_CALL if n == 1 else L.VILLAIN_3BET_HERO_CALL
        return L.VILLAIN_OPEN_HERO_CALL if opp in self.pre_raisers else L.HERO_OPEN_VILLAIN_CALL

    def _main_villain(self, me: int) -> int:
        """Главный оппонент: кто последним ставил; если никто — первый из оставшихся."""
        others = [i for i in self.alive() if i != me]
        for st, who, kind in reversed(self.history):
            if who in others and kind in ("bet", "raise"):
                return who
        for who in reversed(self.pre_raisers):
            if who in others:
                return who
        return others[0]

    def state(self, i: int = H) -> GameState:
        p = self.players[i]
        pos = model_pos(p.pos)
        to_call = max(0.0, round(min(self.cur - p.inv, p.stack), 2))
        common = dict(hero_cards=p.cards, board=self.board, hero_pos=pos, pot=max(self.pot, to_call + 0.1),
                      to_call=to_call, stack=p.stack)
        if self.street == "preflop":
            actions = "/".join(self.pre_tokens)
            line, vpos = classify_preflop(actions, pos)
            v = self.pre_raisers[-1] if vpos is not None else None
            return GameState(**common, villain_pos=vpos, preflop_line=line, preflop_actions=actions,
                             in_position_override=self._ip(i, v) if v is not None else None,
                             villain_adj=self._adjust_for(i, v))
        v = self._main_villain(i)
        extras = [j for j in self.alive() if j not in (i, v)]
        names = {i: "hero", v: "villain", **{j: f"v{k}" for k, j in enumerate(extras, start=2)}}
        hist = [(st, names[w], k) for st, w, k in self.history if w in names]
        amts = [a for (st, w, k), a in zip(self.history, self.history_amounts) if w in names]
        return GameState(
            **common, villain_pos=model_pos(self.players[v].pos), preflop_line=self._relation(i, v),
            hero_invested=p.inv, history=hist, history_amounts=amts, in_position_override=self._ip(i, v),
            extra_villains=[(model_pos(self.players[j].pos), self._relation(i, j)) for j in extras],
            villain_adj=self._adjust_for(i, v))

    def _adjust_for(self, me: int, v: int | None):
        """Поправки на профиль оппонента: герою — про бота, адаптивному боту — про героя."""
        if v is None:
            return None
        if me == H:
            return self.players[v].stats.adjust() if self.use_profile else None
        return self.hero_stats.adjust() if self.adaptive and v == H else None

    def advice(self, i: int = H) -> Analysis:
        st = self.state(i)
        a = analyze(st)
        others = [self.players[j] for j in self.alive() if j != i]
        if st.to_call > 0 and all(o.stack == 0 for o in others):       # все оппоненты в олл-ине: колл или фолд
            if self.street == "preflop" and a.equity is None:
                a.equity = calc.equity_vs_range(to_eval7(st.hero_cards), charts.combos(_SHOVE_RANGE), [])
            if a.equity is not None:
                call = a.equity >= a.required
                a.action, a.amount, a.sizing, a.mix = ("call" if call else "fold"), (st.to_call if call else None), "", []
                a.source = "эквити против шансов банка"
                a.reasons = [f"Оппонент в олл-ине. Ваше эквити против его диапазона — {a.equity:.0%}, "
                             f"для колла нужно {a.required:.0%}: " + ("колл выгоден." if call else "колл в минус.")]
        return a

    # ---------- действия ----------

    def options(self) -> list[Option]:
        if self.to_act != H:
            return []
        p = self.players[H]
        to_call = round(self.cur - p.inv, 2)
        opts: list[Option] = []
        if to_call > 0:
            opts += [Option("Фолд", "fold"), Option(f"Колл {min(to_call, p.stack):g}", "call")]
        else:
            opts.append(Option("Чек", "check"))
        can_reraise = any(self.players[j].stack > 0 for j in self.alive() if j != H)
        if p.stack <= to_call or not can_reraise:
            return opts
        allin_to = p.inv + p.stack
        sizes: list[tuple[str, float]] = []
        if self.street == "preflop":
            limpers = sum(1 for t in self.pre_tokens if t.endswith("/call")) if self.raises == 0 else 0
            if self.raises == 0:
                sizes.append(("Рейз", charts.OPEN_SIZE.get(model_pos(p.pos), 2.5) + limpers))
            elif self.raises == 1:
                sizes.append(("3-бет", round(self.cur * 3.5, 1)))
            elif self.raises == 2:
                sizes.append(("4-бет", round(self.cur * 2.3, 1)))
        elif self.cur == 0:
            sizes += [(f"Бет {int(f * 100)}%", round(self.pot * f, 1)) for f in (0.33, 0.5, 0.75, 1.25)]
        else:
            sizes.append(("Рейз 3×", round(self.cur * 3, 1)))
        for name, to in sizes:
            if self.cur < to < allin_to * 0.8:
                opts.append(Option(f"{name} {to:g}", "raise", to))
        opts.append(Option(f"Олл-ин {allin_to:g}", "raise", allin_to))
        return opts

    def hero_act(self, opt: Option) -> None:
        adv = self.advice(H)
        d = Decision(self.street, opt.label, adv, matches(adv, opt.kind))
        self.decisions.append(d)
        self.last = d
        self._apply(H, opt.kind, opt.to)
        self._advance()

    def _advance(self) -> None:
        """Ходят боты, пока очередь не дойдёт до героя или раздача не закончится."""
        while not self.finished:
            if not self.pending:
                self._end_round()
                continue
            i = self.pending[0]
            if i == H:
                return
            self._bot_act(i)

    def _bot_act(self, i: int) -> None:
        p = self.players[i]
        a = self.advice(i)
        to_call = self.cur - p.inv
        allin_to = p.inv + p.stack
        kind, to, amount = a.action, 0.0, a.amount
        if a.mix:
            kind, amount, _f, _e = self.rng.choices(a.mix, weights=[m[2] for m in a.mix])[0]
        if kind in ("bet", "raise", "allin"):
            to = allin_to if kind == "allin" else max(amount or 0.0, 2 * self.cur)
            kind = "raise"
        base_continue = kind != "fold"
        kind, to = bots.restyle(
            p.style, self.rng, preflop=self.street == "preflop", kind=kind, to=to, cards=p.cards,
            equity=a.equity, pot=self.pot, cur_bet=self.cur, to_call=to_call, raises=self.raises)
        if kind == "raise":
            to = min(max(to, 2 * self.cur, self.cur + 1), allin_to)
            others_have_chips = any(self.players[j].stack > 0 for j in self.alive() if j != i)
            if not others_have_chips or p.stack <= to_call or to <= self.cur:
                kind = "call"
        if kind == "fold" and to_call <= 0:
            kind = "check"
        if kind == "check" and to_call > 0:
            kind = "fold"
        if kind == "call" and to_call <= 0:
            kind = "check"
        self._apply(i, kind, to)

    def _record(self, i: int, kind: str) -> None:
        """Записать действие в профиль: за ботом наблюдает герой, за героем — боты."""
        stats = self.hero_stats if i == H else self.players[i].stats
        p = self.players[i]
        if self.street != "preflop":
            first_bet = self.cur == 0
            stats.record_postflop("bet" if kind == "raise" else kind, self.cur > p.inv)
            return
        if i in self.seen_pre or kind == "check":
            return
        self.seen_pre.add(i)
        stats.record_preflop(kind != "fold", kind == "raise", self._expected_continue(model_pos(p.pos)))

    def _expected_continue(self, pos: str) -> float:
        """Доля рук, с которыми солвер продолжает игру в текущей префлоп-линии."""
        line = preflop_gto.abstract("/".join(self.pre_tokens))
        keys = [k for kind in ("raise", "call") for k in (preflop_gto.range_keys(line, pos, kind) or ())]
        if not keys:
            return BASE_VPIP
        return len(charts.keys_to_combos(tuple(keys))) / 1326

    def _apply(self, i: int, kind: str, to: float) -> None:
        p = self.players[i]
        self._record(i, kind)
        pre = self.street == "preflop"
        tag = f"{STREET_RU[self.street]}: {p.name} ({p.label})"
        mp = model_pos(p.pos)
        self.pending.remove(i)
        if kind == "fold":
            p.folded = True
            self.log.append(f"{tag} — фолд.")
            if pre and i in self.voluntary:
                self.pre_tokens.append(f"{mp}/fold")
            if len(self.alive()) == 1:
                self._finish([self.alive()[0]])
            return
        if kind == "check":
            self.log.append(f"{tag} — чек.")
            if not pre:
                self.history.append((self.street, i, "check"))
                self.history_amounts.append(0.0)
            return
        if kind == "call":
            pay = min(self.cur - p.inv, p.stack)
            self._pay(p, pay)
            self.log.append(f"{tag} — колл {pay:g}" + (" (олл-ин)." if p.stack == 0 else "."))
            if pre:
                self.pre_tokens.append(f"{mp}/call")
                self.voluntary.add(i)
            else:
                self.history.append((self.street, i, "call"))
                self.history_amounts.append(0.0)
            return
        to = min(to, p.inv + p.stack)
        first_bet = not pre and self.cur == 0
        self._pay(p, to - p.inv)
        self.cur = to
        allin = p.stack == 0
        self.log.append(f"{tag} — {'бет' if first_bet else 'рейз до'} {to:g}" + (" (олл-ин)." if allin else "."))
        if pre:
            self.raises += 1
            self.pre_raisers.append(i)
            self.pre_tokens.append(f"{mp}/" + ("allin" if allin else f"{to:.1f}bb"))
            self.voluntary.add(i)
        else:
            self.history.append((self.street, i, "bet" if first_bet else "raise"))
            self.history_amounts.append(to)
        order = self._order()                    # после рейза снова ходят все остальные, начиная со следующего
        k = next((n for n, j in enumerate(order) if self._after(j, i)), 0)
        self.pending = [j for j in order[k:] + order[:k] if j != i]

    def _after(self, j: int, i: int) -> bool:
        """Ходит ли игрок j после игрока i на текущей улице."""
        order = SEATS_BY_N[self.n] if self.street == "preflop" else [p for p in POSTFLOP_FULL if p in SEATS_BY_N[self.n]]
        return order.index(self.players[j].pos) > order.index(self.players[i].pos)

    @staticmethod
    def _pay(p: Player, amount: float) -> None:
        amount = round(amount, 2)
        p.inv = round(p.inv + amount, 2)
        p.stack = round(p.stack - amount, 2)

    def _refund(self) -> None:
        """Вернуть неуравненную часть ставки."""
        invs = sorted(((p.inv, i) for i, p in enumerate(self.players)), reverse=True)
        (top, i), second = invs[0], invs[1][0]
        if top > second:
            p = self.players[i]
            p.inv, p.stack = second, round(p.stack + top - second, 2)

    def _end_round(self) -> None:
        self._refund()
        self.prev += sum(p.inv for p in self.players)
        for p in self.players:
            p.inv = 0.0
        self.cur = 0.0
        while True:
            if self.street == "river":
                self._showdown()
                return
            self.street = STREETS[STREETS.index(self.street) + 1]
            self.board = self._runout[:{"flop": 3, "turn": 4, "river": 5}[self.street]]
            self.log.append(f"{STREET_RU[self.street]}: {' '.join(self.board)}. Банк {self.pot:g}.")
            self.pending = self._order()
            if len(self.pending) >= 2:
                return                            # если ходить некому (олл-ины) — открываем карты до ривера

    def _showdown(self) -> None:
        b = to_eval7(self.board)
        alive = self.alive()
        vals = {i: eval7.evaluate(to_eval7(self.players[i].cards) + b) for i in alive}
        self.shown = set(alive)
        for i in alive:
            p = self.players[i]
            if i != H:
                self.log.append(f"Вскрытие: {p.name} показывает {' '.join(p.cards)} "
                                f"({calc.hand_name(to_eval7(p.cards), b)}).")
            last = [k for st, w, k in self.history if st == "river" and w == i]
            if last and last[-1] in ("bet", "raise"):
                stats = self.hero_stats if i == H else p.stats
                stats.record_showdown(calc.is_weak_showdown(to_eval7(p.cards), b))
        self.log.append(f"Вскрытие: у вас {calc.hand_name(to_eval7(self.players[H].cards), b)}.")
        best = max(vals.values())
        self._finish([i for i in alive if vals[i] == best])

    def _finish(self, winners: list[int]) -> None:
        self._refund()
        pot = self.final_pot = self.pot
        self.prev = 0.0
        for p in self.players:
            p.inv = 0.0
        for i in winners:
            self.players[i].stack = round(self.players[i].stack + pot / len(winners), 2)
        self.pending = []
        self.net = round(self.players[H].stack - 100, 2)
        self.finished = True
        for i, p in enumerate(self.players):
            if i != H:
                p.stats.hands += 1
        self.hero_stats.hands += 1
        names = ", ".join(self.players[i].name for i in winners)
        if winners == [H]:
            who = "Вы выиграли банк"
        elif len(winners) > 1:
            who = f"Банк поделён ({names}):"
        else:
            who = f"{names} выигрывает банк"
        self.result = f"{who} {pot:g} bb. Ваш итог: {self.net:+g} bb."
        self.log.append(self.result)
