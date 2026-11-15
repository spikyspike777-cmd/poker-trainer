"""Настраиваемые пороги эвристик. Значения по умолчанию подобраны вручную;
`python -m poker_coach.tune` подбирает их на обучающей части PokerBench и пишет в data/params.json."""
from __future__ import annotations

import json
from pathlib import Path

FILE = Path(__file__).resolve().parent / "data" / "params.json"

DEFAULT = {
    # стратег на постфлопе
    "raise_eq": 0.72,          # эквити для рейза против ставки (флоп, тёрн)
    "raise_eq_river": 0.80,    # то же на ривере
    "call_margin_river": 0.02, # запас эквити сверх шансов банка для колла на ривере
    "call_margin": 0.0,        # то же на флопе и тёрне
    "value_big": 0.80,         # эквити для крупной ставки
    "value": 0.65,             # эквити для ставки в полбанка
    "thin": 0.50,              # эквити для маленькой ставки в позиции (флоп, тёрн)
    "semibluff_eq": 0.30,      # минимальное эквити для полублефа с дро
    # сужение диапазонов: какую часть (сверху по силе) игрок оставляет при действии
    "bet_keep": 0.50, "bet_bluffs": 0.12,
    "raise_keep": 0.30, "raise_bluffs": 0.08,
    "call_keep": 0.45,
    "check_back_drop": 0.25,   # чек в позиции: убираем верх диапазона
    "solved_ranges": 1,        # 1 — диапазоны текущей улицы берутся из решения солвером прошлых улиц
    "table_ranges": 0,         # 1 — диапазоны из таблицы солвера, 0 — из ручных чартов
}

P = dict(DEFAULT)
if FILE.exists():
    P.update(json.loads(FILE.read_text()))


def set_params(values: dict) -> None:
    P.update(values)
