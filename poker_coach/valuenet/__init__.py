"""Сеть-оценщик: предсказывает ценность каждой руки обоих игроков в точке, где торговля на тёрне
закончена, а ривер ещё не вышел. Заменяет в солвере перебор карт ривера и решение торговли на каждой.

Порядок работы (подробно — в README, раздел «Сеть-оценщик»):

    python -m poker_coach.valuenet.gen --out data/valuenet/turn --samples 100000 --workers 8
    python -m poker_coach.valuenet.train --data data/valuenet/turn --out poker_coach/data/value_turn
    python -m poker_coach.valuenet.evaluate --net poker_coach/data/value_turn.npz --data data/valuenet/turn --spots 30
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT = Path(__file__).resolve().parent.parent / "data" / "value_turn.npz"


def load(path: str | Path = DEFAULT):
    """Загрузить обученную сеть в солвер на Rust. None, если файла или расширения нет."""
    path = Path(path)
    try:
        import solver_rs
    except ImportError:
        return None
    if not path.exists():
        return None
    z = np.load(path)
    return solver_rs.ValueNet(int(z["hc"]), int(z["h"]), int(z["blocks"]),
                              np.ascontiguousarray(z["blob"], dtype="<f4").tobytes())
