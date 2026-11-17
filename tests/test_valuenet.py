import random

import numpy as np
import pytest

from poker_coach import valuenet
from poker_coach.valuenet import evaluate, gen
from poker_coach.valuenet.features import C_CTX, F_HAND

solver_rs = pytest.importorskip("solver_rs")


def _random_net(tmp_path, hc=24, h=32, blocks=2, seed=0):
    r = np.random.default_rng(seed)
    size = hc * C_CTX + hc + hc * hc + hc + 2 * (h * F_HAND + h * hc + h + blocks * (h * h + h) + h + 1)
    path = tmp_path / "net.npz"
    np.savez(path, hc=hc, h=h, blocks=blocks, blob=(r.standard_normal(size) * 0.2).astype(np.float32))
    return path


def test_rust_forward_matches_numpy(tmp_path):
    """Сеть внутри солвера на Rust даёт те же поправки, что и эталонный расчёт на numpy."""
    path = _random_net(tmp_path)
    net = valuenet.load(path)
    board, pot, stack, hands, weights = gen.sample_spot(random.Random(3))
    deck, strengths = gen._strengths(board, hands)
    w = [list(map(float, weights[0])), list(map(float, weights[1]))]
    ctx, f0, f1, e0, e1 = solver_rs.leaf_features(hands[0], hands[1], w[0], w[1], strengths[0], strengths[1],
                                                  deck, list(board), float(pot), float(stack))
    feats = [np.array(f0, np.float32).reshape(-1, F_HAND), np.array(f1, np.float32).reshape(-1, F_HAND)]
    want = evaluate.forward(evaluate.unpack(np.load(path)), np.array(ctx, np.float32), feats,
                            [np.array(w[0]), np.array(w[1])])
    # дерево тёрна из одних чеков: значение в корне — это оценка листа сетью
    out = solver_rs.solve(hands[0], hands[1], w[0], w[1], strengths[0], strengths[1], [deck],
                          [list(range(len(deck)))], [1 / 44], -1000, float(pot), float(stack), [], 0, [[0.75]], [0],
                          3.0, 0.0, False, 1, 0, 0, [], False, False, False, True, list(board), True, net)
    for p, plain in enumerate((e0, e1)):
        m = gen.compat_mass(hands[p], hands[1 - p], weights[1 - p])
        got = (np.array(out[8 + p]) - pot / 2 * m) / pot - np.array(plain)      # поправка, которую добавила сеть
        assert np.allclose(got, want[p], atol=1e-4), float(np.abs(got - want[p]).max())


def test_torch_matches_numpy(tmp_path):
    torch = pytest.importorskip("torch")
    from poker_coach.valuenet.model import ValueNet, export
    model = ValueNet(hc=16, h=24, blocks=2)
    export(model, str(tmp_path / "m.npz"))
    r = np.random.default_rng(1)
    ctx = r.standard_normal(C_CTX).astype(np.float32)
    feats = [r.standard_normal((30, F_HAND)).astype(np.float32), r.standard_normal((20, F_HAND)).astype(np.float32)]
    w = [np.full(30, 1 / 30, np.float32), np.full(20, 1 / 20, np.float32)]
    want = evaluate.forward(evaluate.unpack(np.load(tmp_path / "m.npz")), ctx, feats, w)
    f = torch.zeros(1, 2, 30, F_HAND)
    wt = torch.zeros(1, 2, 30)
    for p in range(2):
        f[0, p, :len(feats[p])] = torch.tensor(feats[p])
        wt[0, p, :len(w[p])] = torch.tensor(w[p])
    got = model(torch.tensor(ctx)[None], f, wt)[0].detach().numpy()
    assert np.allclose(got[0, :30], want[0], atol=1e-4) and np.allclose(got[1, :20], want[1], atol=1e-4)


def test_targets_and_features():
    board, pot, stack, hands, weights = gen.sample_spot(random.Random(11))
    ctx, feats, (t0, t1) = gen.solve_target(board, pot, stack, hands, weights, iters=20)
    assert abs(float(np.dot(weights[0], t0) + np.dot(weights[1], t1))) < 1e-4      # поправки в сумме дают ноль
    assert max(np.abs(t0).max(), np.abs(t1).max()) <= stack / pot + 1 + 1e-6
    assert ctx.shape == (C_CTX,) and feats[0].shape == (len(hands[0]), F_HAND)
    assert abs(ctx[19:29].sum() - 1) < 1e-4 and abs(ctx[29:39].sum() - 1) < 1e-4   # распределения диапазонов по силе
    assert np.all(np.abs(feats[0][:, 0]) <= 1 + 1e-5) and np.all((feats[0][:, 4] >= 0) & (feats[0][:, 4] <= 1))
    assert abs(sum(weights[0]) - 1) < 1e-6 and len(hands[0]) <= gen.MAX_HANDS
