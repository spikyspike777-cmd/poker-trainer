"""Схема раздачи (GameState) и её проверка. Все суммы — в больших блайндах."""
from __future__ import annotations

from enum import Enum

import eval7
from pydantic import BaseModel, Field, field_validator, model_validator

POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
# Порядок хода на постфлопе: кто правее, тот в позиции.
POSTFLOP_ORDER = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
RANKS = "23456789TJQKA"
SUITS = "cdhs"


class PreflopLine(str, Enum):
    """Что произошло на префлопе между героем и оппонентом."""

    UNOPENED = "unopened"                    # до героя все сбросили
    VS_OPEN = "vs_open"                      # оппонент открылся, ход героя
    VS_3BET = "vs_3bet"                      # герой открылся, оппонент 3-бет, ход героя
    VS_4BET = "vs_4bet"                      # герой 3-бет, оппонент 4-бет, ход героя
    HERO_OPEN_VILLAIN_CALL = "hero_open_villain_call"
    VILLAIN_OPEN_HERO_CALL = "villain_open_hero_call"
    HERO_3BET_VILLAIN_CALL = "hero_3bet_villain_call"
    VILLAIN_3BET_HERO_CALL = "villain_3bet_hero_call"


PREFLOP_DECISIONS = {PreflopLine.UNOPENED, PreflopLine.VS_OPEN, PreflopLine.VS_3BET, PreflopLine.VS_4BET}

VILLAIN_OPENED = {PreflopLine.VS_OPEN, PreflopLine.VS_4BET, PreflopLine.VILLAIN_OPEN_HERO_CALL,
                  PreflopLine.HERO_3BET_VILLAIN_CALL}
HERO_OPENED = {PreflopLine.VS_3BET, PreflopLine.HERO_OPEN_VILLAIN_CALL, PreflopLine.VILLAIN_3BET_HERO_CALL}

LINE_LABELS = {
    PreflopLine.UNOPENED: "До меня все сбросили",
    PreflopLine.VS_OPEN: "Оппонент открылся рейзом",
    PreflopLine.VS_3BET: "Я открылся, оппонент сделал 3-бет",
    PreflopLine.VS_4BET: "Я сделал 3-бет, оппонент 4-бет",
    PreflopLine.HERO_OPEN_VILLAIN_CALL: "Я открылся, оппонент заколлировал",
    PreflopLine.VILLAIN_OPEN_HERO_CALL: "Оппонент открылся, я заколлировал",
    PreflopLine.HERO_3BET_VILLAIN_CALL: "Я сделал 3-бет, оппонент заколлировал",
    PreflopLine.VILLAIN_3BET_HERO_CALL: "Оппонент сделал 3-бет, я заколлировал",
}


def parse_cards(text: str) -> list[str]:
    """'As Kh', 'AsKh', 'as,kh' -> ['As', 'Kh']. Бросает ValueError на мусоре."""
    s = text.replace(",", " ").replace("10", "T").strip()
    tokens = s.split() if " " in s else [s[i:i + 2] for i in range(0, len(s), 2)]
    out = []
    for t in tokens:
        if len(t) != 2 or t[0].upper() not in RANKS or t[1].lower() not in SUITS:
            raise ValueError(f"Не понимаю карту «{t}». Формат: As, Kh, Td, 7c")
        out.append(t[0].upper() + t[1].lower())
    return out


def to_eval7(cards: list[str]) -> list[eval7.Card]:
    return [eval7.Card(c) for c in cards]


class GameState(BaseModel):
    hero_cards: list[str]
    board: list[str] = Field(default_factory=list)
    hero_pos: str
    villain_pos: str | None = None
    preflop_line: PreflopLine
    pot: float = Field(gt=0, description="Банк сейчас, включая ставку оппонента")
    to_call: float = Field(default=0, ge=0, description="Сколько нужно доставить")
    stack: float = Field(default=100, gt=0, description="Эффективный стек, оставшийся у героя")
    hero_invested: float = Field(default=0, ge=0, description="Сколько герой уже вложил на этой улице")
    history: list[tuple[str, str, str]] = Field(
        default_factory=list,
        description="Постфлоп-действия по порядку: (улица flop/turn/river, hero/villain, check/bet/raise/call)")

    @field_validator("hero_cards", "board", mode="before")
    @classmethod
    def _parse(cls, v):
        return parse_cards(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check(self):
        if len(self.hero_cards) != 2:
            raise ValueError("У героя должно быть ровно 2 карты")
        if len(self.board) not in (0, 3, 4, 5):
            raise ValueError("На борде должно быть 0, 3, 4 или 5 карт")
        allc = self.hero_cards + self.board
        parse_cards(" ".join(allc))
        if len(set(allc)) != len(allc):
            raise ValueError("Одна и та же карта указана дважды")
        if self.hero_pos not in POSITIONS:
            raise ValueError(f"Позиция героя должна быть одной из {POSITIONS}")
        if self.preflop_line != PreflopLine.UNOPENED:
            if self.villain_pos not in POSITIONS:
                raise ValueError("Укажите позицию оппонента")
            if self.villain_pos == self.hero_pos:
                raise ValueError("Герой и оппонент не могут сидеть на одной позиции")
        if self.preflop_line == PreflopLine.UNOPENED and self.hero_pos == "BB":
            raise ValueError("Если до BB все сбросили, раздача окончена — решать нечего")
            hi, vi = POSITIONS.index(self.hero_pos), POSITIONS.index(self.villain_pos)
            if self.preflop_line in VILLAIN_OPENED and vi > hi:
                raise ValueError(f"Оппонент на {self.villain_pos} ходит после {self.hero_pos} и не мог открыться раньше героя")
            if self.preflop_line in HERO_OPENED and hi > vi:
                raise ValueError(f"Герой на {self.hero_pos} ходит после {self.villain_pos} и не мог открыться раньше оппонента")
        is_pre = self.preflop_line in PREFLOP_DECISIONS
        if is_pre and self.board:
            raise ValueError("Выбрана префлоп-ситуация, но указан борд")
        if not is_pre and not self.board:
            raise ValueError("Выбрана постфлоп-ситуация — укажите борд (минимум 3 карты)")
        if self.to_call >= self.pot:
            raise ValueError("Ставка оппонента уже входит в банк, поэтому колл не может быть больше банка")
        return self

    @property
    def street(self) -> str:
        return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(self.board)]

    @property
    def in_position(self) -> bool:
        if self.villain_pos is None:
            return True
        return POSTFLOP_ORDER.index(self.hero_pos) > POSTFLOP_ORDER.index(self.villain_pos)
