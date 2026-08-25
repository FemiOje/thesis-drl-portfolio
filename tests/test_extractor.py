"""Phase 2 gate: the EIIE extractor does not flatten or ignore the tensor."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.config import load_config
from src.extractors import EIIEExtractor

CFG = load_config()
W, F = CFG.env.window, CFG.env.n_features


def make(m):
    obs_space = spaces.Dict({
        "tensor": spaces.Box(0.0, np.inf, (F, m, W), np.float32),
        "weights": spaces.Box(0.0, 1.0, (m + 1,), np.float32),
    })
    torch.manual_seed(0)
    return EIIEExtractor(obs_space, n_assets=m, window=W, n_features=F)


def batch(m, b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = 1 + 0.05 * torch.randn(b, F, m, W, generator=g)
    w = torch.rand(b, m + 1, generator=g)
    return {"tensor": x, "weights": w / w.sum(-1, keepdim=True)}


def n_params(mod):
    return sum(p.numel() for p in mod.parameters())


def test_parameter_count_is_independent_of_asset_count():
    """DECISIVE. If any layer spans the asset axis, it is flattening, and the count
    varies with M."""
    counts = {m: n_params(make(m)) for m in (4, 8, 16)}
    assert len(set(counts.values())) == 1, counts


def test_permutation_equivariance():
    """DECISIVE. Parameters are genuinely shared across rows and the network cannot
    memorise asset identity by position. Cash stays at index 0."""
    m = 8
    net = make(m).eval()
    ob = batch(m)
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(3))
    ob_p = {"tensor": ob["tensor"][:, :, perm, :],
            "weights": torch.cat([ob["weights"][:, :1], ob["weights"][:, 1:][:, perm]], 1)}
    with torch.no_grad():
        out, out_p = net(ob), net(ob_p)
    assert torch.allclose(out_p[:, 0], out[:, 0], atol=1e-6)
    assert torch.allclose(out_p[:, 1:], out[:, 1:][:, perm], atol=1e-6)


def test_output_changes_for_every_time_index():
    """Catches the failure where the window is accepted but collapsed to its last
    column."""
    m = 8
    net = make(m).eval()
    ob = batch(m)
    with torch.no_grad():
        base = net(ob)
    for t in range(W):
        pert = {k: v.clone() for k, v in ob.items()}
        pert["tensor"][:, :, :, t] += 0.5
        with torch.no_grad():
            out = net(pert)
        assert not torch.allclose(out, base, atol=1e-6), f"output ignores t={t}"


def test_runtime_shape_assertion_fires_on_flattened_input():
    net = make(8)
    with pytest.raises(AssertionError):
        net({"tensor": torch.zeros(2, F * 8 * W), "weights": torch.zeros(2, 9)})


def test_output_shape_and_features_dim():
    for m in (4, 8, 16):
        net = make(m)
        assert net.features_dim == m + 1
        assert net(batch(m)).shape == (4, m + 1)


def test_previous_weights_affect_the_output():
    m = 8
    net = make(m).eval()
    ob = batch(m)
    alt = {k: v.clone() for k, v in ob.items()}
    alt["weights"] = torch.roll(alt["weights"], 1, dims=1)
    with torch.no_grad():
        assert not torch.allclose(net(ob), net(alt), atol=1e-6)


def test_every_asset_row_influences_only_its_own_logit():
    """Kernels of height 1 mean no cross-row mixing before the softmax."""
    m = 8
    net = make(m).eval()
    ob = batch(m)
    with torch.no_grad():
        base = net(ob)
    pert = {k: v.clone() for k, v in ob.items()}
    pert["tensor"][:, :, 3, :] += 0.5
    with torch.no_grad():
        out = net(pert)
    changed = (out - base).abs().max(0).values > 1e-6
    assert changed[4] and changed.sum() == 1     # logit index 4 = asset row 3 (cash is 0)


def test_gradients_reach_every_parameter():
    net = make(8)
    net(batch(8)).sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
