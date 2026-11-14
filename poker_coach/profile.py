"""Профиль оппонента: статистика его действий и поправки к советам тренера.

Всё считается относительно «нормы» — того, как в тех же ситуациях играет стандартный
бот по чартам. Нормы постфлопа измерены симуляцией (см. tests/test_profile.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

# Нормы стандартной игры (измерены на стандартном боте).
BASE_AGGR = 0.33          # доля ставок и рейзов среди постфлоп-действий
BASE_FOLD_TO_BET = 0.30   # как часто сбрасывает на ставку на постфлопе
BASE_BLUFF = 0.27         # доля блефов среди ставок на ривере, дошедших до вскрытия
BASE_VPIP = 0.24          # как часто входит в игру за столом 6-max (для hand history)


class Adjust(BaseModel):
    """Поправки для тренера. Значения 1.0 и None означают «как у стандартного игрока»."""
    range_scale: float = 1.0            # во сколько раз его префлоп-диапазон шире нормы
    bluff_scale: float = 1.0            # во сколько раз больше блефов в его ставках
    fold_to_bet: float | None = None    # оценка частоты фолда на ставку
    label: str = ""                     # «тайтовый, пассивный» и т. п.; пусто — отклонений не видно
    hands: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def exploit(self) -> bool:
        return bool(self.label)


@dataclass
class PlayerStats:
    hands: int = 0
    pre_spots: int = 0            # префлоп-решений, где был выбор
    pre_expected: float = 0.0     # сколько раз стандартный игрок продолжил бы в этих же ситуациях
    pre_continued: int = 0
    pre_raised: int = 0
    bets: int = 0                 # ставки и рейзы на постфлопе
    passive: int = 0              # чеки и коллы
    folds: int = 0
    faced_bet: int = 0
    folded_to_bet: int = 0
    shown_river_bets: int = 0
    shown_river_bluffs: int = 0

    def record_preflop(self, continued: bool, raised: bool, expected: float) -> None:
        self.pre_spots += 1
        self.pre_expected += expected
        self.pre_continued += continued
        self.pre_raised += raised

    def record_postflop(self, kind: str, facing_bet: bool) -> None:
        if kind in ("bet", "raise"):
            self.bets += 1
        elif kind == "fold":
            self.folds += 1
        else:
            self.passive += 1
        if facing_bet:
            self.faced_bet += 1
            self.folded_to_bet += kind == "fold"

    def record_showdown(self, bluff: bool) -> None:
        """Оппонент ставил на ривере и показал карты."""
        self.shown_river_bets += 1
        self.shown_river_bluffs += bluff

    @property
    def post_actions(self) -> int:
        return self.bets + self.passive + self.folds

    @property
    def aggression(self) -> float | None:
        return self.bets / self.post_actions if self.post_actions else None

    def adjust(self) -> Adjust:
        """Поправки со сглаживанием: при малой выборке значения близки к норме."""
        a = Adjust(hands=self.hands)
        if self.pre_spots >= 10:
            a.range_scale = round(min(1.7, max(0.55, (self.pre_continued + 5) / (self.pre_expected + 5))), 2)
        n = self.post_actions
        if n >= 8:
            aggr = (self.bets + BASE_AGGR * 15) / (n + 15)
            scale = aggr / BASE_AGGR
            if self.shown_river_bets >= 5:
                shown = (self.shown_river_bluffs + BASE_BLUFF * 5) / (self.shown_river_bets + 5) / BASE_BLUFF
                scale = 0.7 * scale + 0.3 * shown
            a.bluff_scale = round(min(2.5, max(0.3, scale)), 2)
        if self.faced_bet >= 8:
            a.fold_to_bet = round((self.folded_to_bet + BASE_FOLD_TO_BET * 4) / (self.faced_bet + 4), 2)

        tags, notes = [], []
        if a.range_scale <= 0.75:
            tags.append("тайтовый")
            notes.append(f"входит в игру реже нормы (×{a.range_scale:g}) — его диапазон сильнее обычного")
        elif a.range_scale >= 1.3:
            tags.append("лузовый")
            notes.append(f"входит в игру чаще нормы (×{a.range_scale:g}) — в его диапазоне больше слабых рук")
        if a.bluff_scale >= 1.35:
            tags.append("агрессивный")
            notes.append(f"ставит и рейзит в {self.aggression:.0%} постфлоп-действий при норме {BASE_AGGR:.0%} — "
                         "в его ставках больше блефов")
        elif a.bluff_scale <= 0.7:
            tags.append("пассивный")
            notes.append(f"ставит и рейзит всего в {self.aggression:.0%} постфлоп-действий при норме {BASE_AGGR:.0%} — "
                         "его ставка обычно означает силу")
        if a.fold_to_bet is not None:
            if a.fold_to_bet <= 0.14:
                tags.append("редко сбрасывает")
                notes.append(f"на ставку сбрасывает примерно в {a.fold_to_bet:.0%} случаев — блефовать против него невыгодно")
            elif a.fold_to_bet >= 0.42:
                tags.append("часто сбрасывает")
                notes.append(f"на ставку сбрасывает примерно в {a.fold_to_bet:.0%} случаев — блефы против него окупаются")
        a.label, a.notes = ", ".join(tags), notes
        return a

    def summary(self) -> str:
        parts = [f"раздач: {self.hands}"]
        if self.pre_spots:
            parts.append(f"продолжает на префлопе {self.pre_continued}/{self.pre_spots} "
                         f"(норма ≈{self.pre_expected:.0f})")
        if self.post_actions:
            parts.append(f"ставки и рейзы {self.aggression:.0%} действий")
        if self.faced_bet:
            parts.append(f"фолд на ставку {self.folded_to_bet}/{self.faced_bet}")
        if self.shown_river_bets:
            parts.append(f"блефов на вскрытии {self.shown_river_bluffs}/{self.shown_river_bets}")
        return " · ".join(parts)
