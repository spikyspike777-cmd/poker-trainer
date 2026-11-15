"""Разбор hand history в формате PokerStars: каждое решение героя сверяется с тренером."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import calc
from .coach import Analysis, analyze, matches
from .profile import BASE_VPIP, PlayerStats
from .state import POSITIONS, GameState, PreflopLine as L, to_eval7

_POS_BY_N = {2: ["SB", "BB"], 3: ["SB", "BB", "BTN"], 4: ["SB", "BB", "CO", "BTN"],
             5: ["SB", "BB", "HJ", "CO", "BTN"], 6: ["SB", "BB", "UTG", "HJ", "CO", "BTN"]}
_MONEY = r"[$€£]?([\d,]+(?:\.\d+)?)"
_ACTION = re.compile(rf"^(.+?): (folds|checks|calls|bets|raises)(?: {_MONEY})?(?: to {_MONEY})?")
_STREETS = {"*** FLOP ***": "flop", "*** TURN ***": "turn", "*** RIVER ***": "river"}
STREET_RU = {"preflop": "Префлоп", "flop": "Флоп", "turn": "Тёрн", "river": "Ривер"}


@dataclass
class DecisionReview:
    street: str
    actual: str                           # что сделал герой, по-русски
    kind: str                             # fold / check / call / raise
    state: GameState | None = None
    advice: Analysis | None = None
    matched: bool | None = None           # None — спот не поддерживается
    note: str = ""


@dataclass
class HandReview:
    hand_id: str
    hero_cards: list[str]
    hero_pos: str
    board: list[str] = field(default_factory=list)
    decisions: list[DecisionReview] = field(default_factory=list)
    net_bb: float = 0.0


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def split_hands(text: str) -> list[str]:
    parts = re.split(r"(?=PokerStars (?:Hand|Game|Zoom Hand) #)", text)
    return [p.strip() for p in parts if "*** HOLE CARDS ***" in p]


def parse_hand(text: str, profiles: dict[str, PlayerStats] | None = None) -> HandReview | None:
    """profiles — статистика игроков по прошлым раздачам сессии; пополняется этой раздачей."""
    profiles = profiles if profiles is not None else {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    m = re.search(rf"\({_MONEY}/{_MONEY}", lines[0])
    hero_m = re.search(r"Dealt to (.+?) \[(\w\w) (\w\w)\]", text)
    btn_m = re.search(r"Seat #(\d+) is the button", text)
    if not (m and hero_m and btn_m):
        return None
    bb = _num(m.group(2))
    hero = hero_m.group(1)
    seats: dict[int, str] = {}
    stacks: dict[str, float] = {}
    for ln in lines:
        if ln.startswith("*** HOLE CARDS"):
            break
        sm = re.match(rf"Seat (\d+): (.+?) \({_MONEY} in chips\)", ln)
        if sm and "sitting out" not in ln:
            seats[int(sm.group(1))] = sm.group(2)
            stacks[sm.group(2)] = _num(sm.group(3)) / bb
    order = sorted(seats)
    if hero not in stacks or len(order) not in _POS_BY_N:
        return None
    btn = int(btn_m.group(1))
    start = next((i for i, s in enumerate(order) if s > btn), 0)
    ring = [seats[s] for s in order[start:] + order[:start]]         # начиная с первого после баттона
    if len(ring) == 2:
        ring = ring[::-1]                                             # один на один баттон — это SB
    pos = dict(zip(ring, _POS_BY_N[len(ring)]))

    hid = re.search(r"#(\d+)", lines[0]).group(1)
    review = HandReview(hid, [hero_m.group(2), hero_m.group(3)], pos[hero])
    inv = {p: 0.0 for p in stacks}                # вложено на текущей улице
    total = {p: 0.0 for p in stacks}              # вложено за раздачу
    active = set(stacks)
    street, board, pot = "preflop", [], 0.0
    pre_raises: list[str] = []                    # кто рейзил на префлопе, по порядку
    pre_callers: list[str] = []                   # коллы и лимпы на префлопе
    history: list[tuple[str, str, str]] = []
    amounts: list[float] = []                     # параллельно history: сумма ставки или рейза
    collected = 0.0
    events: list[tuple[str, str, str, bool]] = []       # (игрок, улица, действие, отвечал ли на ставку)
    shown: dict[str, list[str]] = {}
    pre_tokens: list[str] = []                    # префлоп-линия в нотации PokerBench
    acted: set[str] = set()                       # кто уже вложил деньги добровольно

    def put(player: str, to: float) -> None:
        nonlocal pot
        delta = to - inv[player]
        inv[player], pot = to, pot + delta
        total[player] += delta
        stacks[player] -= delta

    for ln in lines:
        if ln.startswith("*** SUMMARY"):
            break
        pm = re.match(rf"(.+?): posts (?:small blind|big blind|the ante|small & big blinds) {_MONEY}", ln)
        if pm and pm.group(1) in inv:
            put(pm.group(1), inv[pm.group(1)] + _num(pm.group(2)) / bb)
            continue
        key = next((k for k in _STREETS if ln.startswith(k)), None)
        if key:
            street = _STREETS[key]
            board = re.findall(r"\b([2-9TJQKA][cdhs])\b", ln.split("***")[2])
            inv = {p: 0.0 for p in inv}
            continue
        um = re.match(rf"Uncalled bet \({_MONEY}\) returned to (.+)", ln)
        if um and um.group(2) in total:
            total[um.group(2)] -= _num(um.group(1)) / bb
            continue
        cm = re.match(rf"(.+?) collected {_MONEY} from", ln)
        if cm:
            if cm.group(1) == hero:
                collected += _num(cm.group(2)) / bb
            continue
        sh = re.match(r"(.+?): shows \[(\w\w) (\w\w)\]", ln)
        if sh and sh.group(1) in inv:
            shown[sh.group(1)] = [sh.group(2), sh.group(3)]
            continue
        am = _ACTION.match(ln)
        if not am or am.group(1) not in inv:
            continue
        who, verb = am.group(1), am.group(2)
        amount = _num(am.group(3)) / bb if am.group(3) else 0.0
        to = _num(am.group(4)) / bb if am.group(4) else 0.0
        kind = {"folds": "fold", "checks": "check", "calls": "call", "bets": "raise", "raises": "raise"}[verb]
        if who == hero:
            review.decisions.append(_review(
                street, kind, verb, amount, to, hero, pos, board, pot, inv, stacks, active,
                pre_raises, pre_callers, list(zip(history, amounts)), review.hero_cards, profiles,
                "/".join(pre_tokens)))
        else:
            events.append((who, street, {"bets": "bet", "raises": "raise", "calls": "call",
                                         "checks": "check", "folds": "fold"}[verb], max(inv.values()) > inv[who]))
        if street == "preflop":
            if verb == "raises":
                pre_tokens.append(f"{pos[who]}/" + ("allin" if "all-in" in ln else f"{to:.1f}bb"))
                acted.add(who)
            elif verb == "calls":
                pre_tokens.append(f"{pos[who]}/call")
                acted.add(who)
            elif verb == "folds" and who in acted:
                pre_tokens.append(f"{pos[who]}/fold")
        if verb == "folds":
            active.discard(who)
        elif verb == "calls":
            put(who, inv[who] + amount)
            if street == "preflop":
                pre_callers.append(who)
        elif verb == "bets":
            put(who, amount)
        elif verb == "raises":
            put(who, to)
            if street == "preflop":
                pre_raises.append(who)
                pre_callers.clear()
        if street != "preflop" and verb != "folds":
            history.append((street, who, {"bets": "bet", "raises": "raise", "calls": "call", "checks": "check"}[verb]))
            amounts.append(amount if verb == "bets" else to if verb == "raises" else 0.0)
    review.board = board
    for name in stacks:
        if name == hero:
            continue
        st = profiles.setdefault(name, PlayerStats())
        st.hands += 1
        mine = [e for e in events if e[0] == name]
        pre = [e for e in mine if e[1] == "preflop" and e[2] != "check"]
        if pre:
            st.record_preflop(any(e[2] != "fold" for e in pre), any(e[2] == "raise" for e in pre), BASE_VPIP)
        for _n, st_name, kind, facing in mine:
            if st_name != "preflop":
                st.record_postflop(kind, facing)
        river = [e[2] for e in mine if e[1] == "river"]
        if name in shown and len(board) == 5 and river and river[-1] in ("bet", "raise"):
            st.record_showdown(calc.is_weak_showdown(to_eval7(shown[name]), to_eval7(board)))
    review.net_bb = round(collected - total[hero], 2)
    return review


def _review(street, kind, verb, amount, to, hero, pos, board, pot, inv, stacks, active,
            pre_raises, pre_callers, history, cards, profiles, pre_line) -> DecisionReview:
    ru = {"folds": "фолд", "checks": "чек", "calls": f"колл {amount:g} bb",
          "bets": f"бет {amount:g} bb", "raises": f"рейз до {to:g} bb"}[verb]
    d = DecisionReview(street, ru, kind)
    others = [p for p in active if p != hero]
    n = len(pre_raises)
    line = villain = None
    if street == "preflop":
        if n == 0 and not pre_callers:
            line = L.UNOPENED
        elif n == 1 and pre_raises[0] != hero and not pre_callers:
            line, villain = L.VS_OPEN, pre_raises[0]
        elif n == 2 and pre_raises[0] == hero and not pre_callers:
            line, villain = L.VS_3BET, pre_raises[1]
        elif n == 3 and pre_raises[1] == hero and pre_raises[2] == pre_raises[0] and not pre_callers:
            line, villain = L.VS_4BET, pre_raises[2]
        else:
            line = L.OTHER                      # лимпы, несколько игроков, олл-ины — по таблице солвера
    elif len(others) != 1 or not 1 <= n <= 3:
        d.note = "Постфлоп поддерживается только один на один в рейзнутом банке."
    else:
        villain = others[0]
        opened = pre_raises[0] == hero
        if pre_raises[0] not in (hero, villain) or pre_raises[-1] not in (hero, villain):
            d.note = "Рейзил игрок, которого уже нет в раздаче — такие линии пока не поддерживаются."
        elif n == 1:
            line = L.HERO_OPEN_VILLAIN_CALL if opened else L.VILLAIN_OPEN_HERO_CALL
        elif n == 2:
            line = L.VILLAIN_3BET_HERO_CALL if opened else L.HERO_3BET_VILLAIN_CALL
        else:
            line = L.VILLAIN_3BET_HERO_CALL if opened else L.HERO_3BET_VILLAIN_CALL
    if line is None:
        return d
    to_call = 0.0 if line == L.UNOPENED else max(0.0, max(inv.values()) - inv[hero])
    pre = street == "preflop"
    try:
        d.state = GameState(
            hero_cards=cards, board=board, hero_pos=pos[hero], villain_pos=pos[villain] if villain else None,
            preflop_line=line, pot=round(pot, 2), to_call=round(min(to_call, stacks[hero]), 2),
            stack=round(stacks[hero], 2), hero_invested=0 if pre else round(inv[hero], 2),
            preflop_actions=pre_line if pre else None,
            history=[(st, "hero" if w == hero else "villain", k) for (st, w, k), _a in history],
            history_amounts=[a for _h, a in history],
            villain_adj=profiles[villain].adjust() if villain in profiles else None)
        d.advice = analyze(d.state)
        d.matched = matches(d.advice, kind)
    except (ValidationError, ValueError) as e:
        d.note = "Не удалось разобрать ситуацию: " + str(e).splitlines()[-1].strip()
    return d


def review_session(text: str) -> list[HandReview]:
    out = []
    profiles: dict[str, PlayerStats] = {}        # профиль копится по ходу сессии: только прошлые раздачи
    for chunk in split_hands(text):
        h = parse_hand(chunk, profiles)
        if h is not None:
            out.append(h)
    return out
