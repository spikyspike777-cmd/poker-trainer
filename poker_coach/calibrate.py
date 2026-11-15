"""Модель поверх выхода солвера: учится угадывать действие эталона PokerBench.

Солвер во многих спотах смешивает действия, а эталон записывает одно. Модель видит частоты и EV
действий по нашему солверу, улицу, позицию, размер ставки — и выбирает действие. Обучается только
на обучающей части PokerBench.

    python -m poker_coach.bench --train --n 3000 --seed 2 --skip-preflop --dump data/features/train.jsonl
    python -m poker_coach.calibrate --fit data/features/train.jsonl --eval data/features/test.jsonl

В приложении модель выключена (params: calibrate=0): она подгоняет ответ под конкретный эталон и
игроку ничего не объясняет. Нужна она для замера и как пример дообучения на данных.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FAMILIES = ["fold", "check", "call", "bet", "raise"]
FEATS = (["street", "ip", "call_frac", "invested", "spr", "equity", "required"]
         + [f"p_{f}" for f in FAMILIES] + [f"gap_{f}" for f in FAMILIES] + [f"can_{f}" for f in FAMILIES])
MODEL = Path(__file__).resolve().parent / "data" / "calibrator.joblib"


def load(path: str):
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["gold"] in FAMILIES]
    x = [[float(r[f]) for f in FEATS] for r in rows]
    y = [FAMILIES.index(r["gold"]) for r in rows]
    return rows, x, y


def _predict(model, x, rows) -> list[int]:
    """Самое вероятное действие среди доступных в споте."""
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    out = []
    for p, r in zip(proba, rows):
        best, best_p = None, -1.0
        for j, c in enumerate(classes):
            if r[f"can_{FAMILIES[c]}"] and p[j] > best_p:
                best, best_p = c, p[j]
        out.append(best if best is not None else classes[int(p.argmax())])
    return out


def _new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0,
                                          random_state=0)


def fit(train_path: str, eval_path: str | None = None, log=print) -> None:
    import joblib
    from sklearn.model_selection import KFold
    rows, x, y = load(train_path)
    base = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
    hits = 0
    for tr, te in KFold(5, shuffle=True, random_state=0).split(x):           # честная оценка на обучающей части
        m = _new_model().fit([x[i] for i in tr], [y[i] for i in tr])
        pred = _predict(m, [x[i] for i in te], [rows[i] for i in te])
        hits += sum(p == y[i] for p, i in zip(pred, te))
    log(f"обучающая часть: спотов {len(rows)}; солвер без модели {base:.1%}; с моделью (5 блоков) {hits / len(rows):.1%}")
    model = _new_model().fit(x, y)
    joblib.dump(model, MODEL)
    choose.cache_clear() if hasattr(choose, "cache_clear") else None
    _model.cache_clear()
    if eval_path:
        rows_t, x_t, y_t = load(eval_path)
        pred = _predict(model, x_t, rows_t)
        base_t = sum(r["pred"] == r["gold"] for r in rows_t) / len(rows_t)
        acc = sum(p == t for p, t in zip(pred, y_t)) / len(rows_t)
        log(f"проверочная часть: спотов {len(rows_t)}; солвер без модели {base_t:.1%}; с моделью {acc:.1%}")
        for name, sel in (("где солвер уверен", lambda r: max(r[f"p_{f}"] for f in FAMILIES) >= 0.8),
                          ("где солвер смешивает", lambda r: max(r[f"p_{f}"] for f in FAMILIES) < 0.8)):
            idx = [i for i, r in enumerate(rows_t) if sel(r)]
            if idx:
                b = sum(rows_t[i]["pred"] == rows_t[i]["gold"] for i in idx) / len(idx)
                a = sum(pred[i] == y_t[i] for i in idx) / len(idx)
                log(f"  {name}: n={len(idx)}, без модели {b:.1%}, с моделью {a:.1%}")


@lru_cache(maxsize=1)
def _model():
    if not MODEL.exists():
        return None
    import joblib
    return joblib.load(MODEL)


def choose(features: dict) -> str | None:
    """Действие, которое модель считает ответом эталона, или None, если модели нет."""
    model = _model()
    if model is None:
        return None
    pred = _predict(model, [[float(features[f]) for f in FEATS]], [features])
    return FAMILIES[pred[0]]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fit", required=True)
    p.add_argument("--eval")
    a = p.parse_args()
    fit(a.fit, a.eval)
