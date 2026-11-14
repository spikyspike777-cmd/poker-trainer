"""Генератор случайных учебных ситуаций для тренажёра."""
from __future__ import annotations

import random

from .state import POSITIONS, GameState, PreflopLine as L, RANKS, SUITS, HERO_OPENED

_WEIGHTS = {
    L.UNOPENED: 2, L.VS_OPEN: 2, L.VS_3BET: 1,
    L.HERO_OPEN_VILLAIN_CALL: 3, L.VILLAIN_OPEN_HERO_CALL: 3,
    L.HERO_3BET_VILLAIN_CALL: 1, L.VILLAIN_3BET_HERO_CALL: 1,
}


def random_state(rng: random.Random | None = None) -> GameState:
    r = rng or random.Random()
    line = r.choices(list(_WEIGHTS), weights=list(_WEIGHTS.values()))[0]
    deck = [a + b for a in RANKS for b in SUITS]
    r.shuffle(deck)
    hero_cards = deck[:2]
    if line == L.UNOPENED:
        pos = r.choice(POSITIONS[:-1])
        return GameState(hero_cards=hero_cards, hero_pos=pos, preflop_line=line, pot=1.5,
                         stack=100 - {"SB": 0.5}.get(pos, 0))
    i, j = sorted(r.sample(range(6), 2))            # i открылся, j ответил
    opener, other = POSITIONS[i], POSITIONS[j]
    hero_pos, vil_pos = (opener, other) if line in HERO_OPENED else (other, opener)
    open_size = 3.0 if opener == "SB" else 2.5
    three = open_size * (4 if other in ("SB", "BB") else 3)
    kw = dict(hero_cards=hero_cards, hero_pos=hero_pos, villain_pos=vil_pos, preflop_line=line)
    if line == L.VS_OPEN:
        blind = {"SB": 0.5, "BB": 1.0}.get(hero_pos, 0)
        return GameState(**kw, pot=1.5 + open_size, to_call=open_size - blind, stack=100 - blind)
    if line == L.VS_3BET:
        return GameState(**kw, pot=1.5 + open_size + three, to_call=three - open_size, stack=100 - open_size)
    threebet_pot = line in (L.HERO_3BET_VILLAIN_CALL, L.VILLAIN_3BET_HERO_CALL)
    put_in = three if threebet_pot else open_size
    pot, stack = 2 * put_in + 1.0, 100 - put_in
    n_board = r.choice([3, 3, 4, 5])
    for _ in range(n_board - 3):                     # прошлые улицы: бет полбанка — колл
        bet = round(pot * 0.5, 1)
        pot, stack = pot + 2 * bet, stack - bet
    to_call = 0.0
    if r.random() < 0.5:
        to_call = min(round(pot * r.choice([0.33, 0.5, 0.75]), 1), stack)
        pot += to_call
    return GameState(**kw, board=deck[2:2 + n_board], pot=round(pot, 1), to_call=to_call, stack=round(stack, 1))
