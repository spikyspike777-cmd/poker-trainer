"""Обучение сети-оценщика.

    python -m poker_coach.valuenet.train --data data/valuenet/turn --out poker_coach/data/value_turn

Пишет <out>.pt (чекпойнт для продолжения) и <out>.npz (веса для солвера; сохраняется лучшая по проверочной
части версия). Ошибка печатается в процентах банка; «без сети» — ошибка, если поправку считать нулём.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time

import numpy as np
import torch

from .model import ValueNet, export


def load_shards(folder: str) -> tuple[dict, list[int]]:
    files = sorted(glob.glob(os.path.join(folder, "shard_*.npz")))
    if not files:
        raise SystemExit(f"в {folder} нет порций данных — сначала запустите poker_coach.valuenet.gen")
    parts = [np.load(f) for f in files]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("ctx", "n", "f", "w", "t")}
    sizes = [len(p["ctx"]) for p in parts]
    return data, sizes


def evaluate(model, data, rows, batch: int, device) -> tuple[float, float, float]:
    """-> (ошибка по рукам, ошибка с весами диапазона, ошибка без сети), всё в процентах банка."""
    model.eval()
    err = cnt = werr = zero = 0.0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            ctx, f, w, t, mask = (data[k][r].to(device) for k in ("ctx", "f", "w", "t", "mask"))
            d = (model(ctx, f.float(), w) - t).abs() * mask
            err, cnt = err + d.sum().item(), cnt + mask.sum().item()
            werr += (d * w).sum().item() / 2            # веса каждого игрока в сумме дают 1
            zero += (t.abs() * mask).sum().item()
    return 100 * err / max(cnt, 1), 100 * werr / max(len(rows), 1), 100 * zero / max(cnt, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="poker_coach/data/value_turn")
    p.add_argument("--hc", type=int, default=128, help="ширина слоёв контекста")
    p.add_argument("--h", type=int, default=128, help="ширина сети руки (от неё зависит скорость при игре)")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val", type=float, default=0.1, help="доля порций данных для проверки")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true", help="продолжить с <out>.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="взять только первые N обучающих примеров (для кривой обучения)")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    raw, sizes = load_shards(a.data)
    n = len(raw["ctx"])
    data = {k: torch.as_tensor(raw[k]) for k in ("ctx", "f", "w", "t")}
    k = data["w"].shape[2]
    data["mask"] = (torch.arange(k)[None, None, :] < torch.as_tensor(raw["n"]).long()[:, :, None]).float()
    n_val = max(1, int(n * a.val))                      # проверка — последние примеры (целые порции в конце)
    val_rows, train_rows = torch.arange(n - n_val, n), torch.arange(0, n - n_val)
    if a.limit:
        train_rows = train_rows[:a.limit]
    print(f"примеров: {n} из {len(sizes)} порций; обучение {len(train_rows)}, проверка {n_val}; устройство {device}", flush=True)
    model = ValueNet(a.hc, a.h, a.blocks).to(device)
    # усреднение весов по ходу обучения: в начале быстрое, затем с памятью около 1000 шагов
    ema = torch.optim.swa_utils.AveragedModel(
        model, avg_fn=lambda avg, cur, num: avg + (cur - avg) * max(0.001, 1.0 / (num + 1)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = max(1, len(train_rows) // a.batch)
    total = a.epochs * steps
    warm = min(500, total // 20 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    start, best = 0, float("inf")
    if a.resume and os.path.exists(a.out + ".pt"):
        ck = torch.load(a.out + ".pt", map_location=device)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start, best = ck["epoch"], ck["best"]
        print(f"продолжаю с эпохи {start}, лучшая ошибка {best:.3f}% банка")
    huber = torch.nn.HuberLoss(reduction="none", delta=0.25)
    for epoch in range(start, a.epochs):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        rows = train_rows[torch.randperm(len(train_rows))]
        for i in range(0, max(len(rows) - a.batch + 1, 1), a.batch):
            r = rows[i:i + a.batch]
            ctx, f, w, t, mask = (data[key][r].to(device, non_blocking=True) for key in ("ctx", "f", "w", "t", "mask"))
            loss = (huber(model(ctx, f.float(), w), t) * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update_parameters(model)
            run, seen = run + loss.item(), seen + 1
        mae, wmae, zero = evaluate(ema.module, data, val_rows, a.batch, device)
        flag = ""
        if mae < best:
            best, flag = mae, "  ← лучшая, веса сохранены"
            export(ema.module, a.out + ".npz")
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch + 1, best=best), a.out + ".pt")
        print(f"эпоха {epoch + 1:3d}/{a.epochs}: потери {run / max(seen, 1):.5f}; ошибка на проверке {mae:.3f}% банка "
              f"(с весами диапазона {wmae:.3f}%; без сети {zero:.3f}%); {time.time() - t0:.0f} с{flag}", flush=True)


if __name__ == "__main__":
    main()
