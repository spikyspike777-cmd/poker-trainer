"""Архитектура сети-оценщика (PyTorch) и выгрузка весов для солвера на Rust.

Контекст (банк, борд, форма диапазонов) проходит через два слоя; затем каждая рука оценивается
общей для всех рук сетью своего игрока по её признакам и этому контексту. Веса общие для всех рук,
поэтому один решённый спот даёт сотни обучающих примеров.

Прямой проход повторён в solver_rs/src/lib.rs (net_forward) и на numpy в evaluate.py:
любое изменение архитектуры нужно делать во всех трёх местах.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import C_CTX, F_HAND


class Head(nn.Module):
    def __init__(self, hc: int, h: int, blocks: int):
        super().__init__()
        self.wf = nn.Linear(F_HAND, h, bias=False)
        self.wc = nn.Linear(hc, h)                     # сдвиг живёт здесь
        self.blocks = nn.ModuleList(nn.Linear(h, h) for _ in range(blocks))
        self.out = nn.Linear(h, 1)
        for b in self.blocks:                          # остаточные ветки стартуют слабыми
            nn.init.normal_(b.weight, std=0.5 / h ** 0.5)
            nn.init.zeros_(b.bias)
        nn.init.normal_(self.out.weight, std=0.1 / h ** 0.5)
        nn.init.zeros_(self.out.bias)

    def forward(self, f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """f: (b, k, F_HAND), c: (b, hc) -> (b, k)"""
        z = torch.relu(self.wf(f) + self.wc(c)[:, None, :])
        for b in self.blocks:
            z = z + torch.relu(b(z))
        return self.out(z).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, hc: int = 128, h: int = 128, blocks: int = 2):
        super().__init__()
        self.hc, self.h, self.n_blocks = hc, h, blocks
        self.c1, self.c2 = nn.Linear(C_CTX, hc), nn.Linear(hc, hc)
        self.heads = nn.ModuleList(Head(hc, h, blocks) for _ in range(2))

    def forward(self, ctx: torch.Tensor, f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """ctx: (b, C_CTX); f: (b, 2, k, F_HAND); w: (b, 2, k) — веса рук (0 там, где руки нет) -> (b, 2, k)"""
        c = torch.relu(self.c2(torch.relu(self.c1(ctx))))
        t = torch.stack([self.heads[p](f[:, p], c) for p in range(2)], dim=1)
        # игра с нулевой суммой: поправки двух игроков, взвешенные их диапазонами, в сумме дают ноль
        err = (t * w).sum(dim=(1, 2), keepdim=True)
        return t - err / 2


def export(model: ValueNet, path: str) -> None:
    """Сохранить веса в .npz для загрузки в солвер (poker_coach.valuenet.load)."""
    f = lambda t: t.detach().cpu().numpy().astype("<f4").ravel()
    parts = [f(model.c1.weight), f(model.c1.bias), f(model.c2.weight), f(model.c2.bias)]
    for head in model.heads:
        parts += [f(head.wf.weight), f(head.wc.weight), f(head.wc.bias)]
        for b in head.blocks:
            parts += [f(b.weight), f(b.bias)]
        parts += [f(head.out.weight), f(head.out.bias)]
    np.savez(path, hc=model.hc, h=model.h, blocks=model.n_blocks, blob=np.concatenate(parts))
