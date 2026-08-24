"""Feature extractors for the portfolio policies.

Why this file exists
--------------------
SB3's ``MultiInputPolicy`` builds a ``CombinedExtractor``, which gives a Dict
entry a CNN only if that entry is an *image space*. Our price tensor ``X`` is a
``Box(0, inf, (3, 50, 8))`` — not an image space — so it takes the other branch:

    'X'      -> nn.Flatten()      # 3 x 50 x 8 = 1200
    'w_prev' -> nn.Flatten()      #                 9
    features_dim: 1209

Every piece of structure Jiang's Eq. 18 encodes is destroyed there. The 50
entries being consecutive days, the 8 columns being different assets, the 3
channels being high/low/close — after the flatten those are 1209 unrelated
inputs, and the network has to rediscover all of it from scratch.

``AssetSharedConvExtractor`` below replaces that flatten with the convolution
stack of Jiang §5.2 / Fig. 2.

NOT full EIIE -- name it accurately in the write-up
---------------------------------------------------
This is the *encoder* half of Jiang's design, not the whole topology. Four
things differ, and claiming "EIIE" without them is indefensible:

1. Jiang's final layer is itself a (1,1) convolution emitting one score per
   asset, so weight sharing reaches the action. Here SB3's default MLP head
   maps 169 features -> 9 logits, breaking sharing at the head.
2. Jiang injects ``w_{t-1}`` as an extra *channel* before that final
   convolution, so it is evaluated per-asset. Here it is concatenated flat.
3. Jiang appends an explicit learned cash bias before the softmax.
4. The Portfolio Vector Memory and online stochastic batch learning are not
   implemented at all.

Describe this as "a weight-shared convolutional encoder following the EIIE
principle", not as EIIE.

The Identical Independent Evaluators property
---------------------------------------------
Both convolutions use a kernel that is ``k`` timesteps **x 1 asset**. Sliding
such a kernel over the tensor applies the *same* filter to every asset column,
independently. That is exactly Jiang's "Identical Independent Evaluators" idea:
the network learns how to evaluate *an* asset once, instead of learning eight
unrelated per-asset functions. Two consequences worth stating in the write-up:

* parameter count is independent of ``m`` — adding a 9th stock changes nothing;
* every asset's price history contributes to training the same kernels, so the
  effective sample size per parameter is ``m`` times larger.

Shapes, for the 8-asset universe (verified in ``__main__`` below):

    X            (B,  3, 50, 8)
    conv1 (3,1)  (B,  2, 48, 8)     2 filters over a 3-day span, per asset
    conv2 (48,1) (B, 20,  1, 8)     collapses the remaining time axis
    flatten      (B, 160)
    + w_prev     (B, 169)           -> features_dim

Compared with the flatten path: 169 features instead of 1209, and 1,960 encoder
parameters instead of 77,376 in the first MLP layer alone.

Relationship to full EIIE
-------------------------
Jiang's final layer is itself a (1,1) convolution producing one score per asset,
plus a cash bias, so weight sharing holds all the way to the action. Here the
extractor feeds SB3's default MLP head, which maps 169 features -> 9 logits and
therefore breaks sharing at the head. This is the pragmatic SB3-compatible
version: it preserves the inductive bias in the encoder, which is where the
1200-way flatten did its damage, without requiring a custom policy network.
Going the rest of the way is a separate change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AssetSharedConvExtractor(BaseFeaturesExtractor):
    """Weight-shared convolutional encoder for the (features, time, asset) tensor.

    Parameters
    ----------
    observation_space : spaces.Dict
        Must contain ``X`` of shape ``(f, n, m)`` and ``w_prev`` of shape ``(m+1,)``.
    n_filters : int
        Output channels of the second convolution (Jiang uses 20).
    kernel_time : int
        Time span of the first convolution (Jiang uses 3).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        n_filters: int = 20,
        kernel_time: int = 3,
    ) -> None:
        f, n, m = observation_space["X"].shape
        n_w = observation_space["w_prev"].shape[0]
        if kernel_time >= n:
            raise ValueError(f"kernel_time={kernel_time} must be < window n={n}")

        super().__init__(observation_space, features_dim=n_filters * m + n_w)

        # Kernels are (time, asset) = (k, 1): width 1 over the asset axis is what
        # makes the filters shared across assets. Do not widen it — a kernel that
        # spans assets would tie the model to a fixed universe ordering.
        self.conv = nn.Sequential(
            nn.Conv2d(f, 2, kernel_size=(kernel_time, 1)),
            nn.ReLU(),
            nn.Conv2d(2, n_filters, kernel_size=(n - kernel_time + 1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # w_prev is already flat (B, m+1); the portfolio state is appended after
        # the convolution so it is not smeared across the asset kernels.
        return torch.cat([self.conv(observations["X"]), observations["w_prev"]], dim=1)


#: Backwards-compatible alias: probe checkpoints were pickled under this name.
EIIEExtractor = AssetSharedConvExtractor

#: Selectable by name from the training CLI. "flatten" means "leave SB3 alone",
#: i.e. the CombinedExtractor default that produced the original results.
#: "eiie" is kept as an alias of "conv" so earlier commands still work.
EXTRACTORS: dict[str, type[BaseFeaturesExtractor] | None] = {
    "flatten": None,
    "conv": AssetSharedConvExtractor,
    "eiie": AssetSharedConvExtractor,
}


if __name__ == "__main__":
    # Shape/parameter self-check against the real observation space.
    import os
    import sys

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _sub in ("env", "data"):
        sys.path.insert(0, os.path.join(_ROOT, _sub))
    import portfolio_env as pe

    env = pe.make_env("train", episode_length=50, random_start=False)
    ex = AssetSharedConvExtractor(env.observation_space)
    obs, _ = env.reset(seed=0)
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
    out = ex(batch)

    flat_dim = int(torch.tensor(env.observation_space["X"].shape).prod()) + \
        env.observation_space["w_prev"].shape[0]
    conv_params = sum(p.numel() for p in ex.conv.parameters())

    print("=== AssetSharedConvExtractor self-check ===")
    print(f"X                : {tuple(batch['X'].shape)}")
    print(f"features out     : {tuple(out.shape)}  (features_dim={ex.features_dim})")
    print(f"flatten baseline : {flat_dim} features, "
          f"{flat_dim * 64:,} params in the first MLP layer")
    print(f"eiie             : {ex.features_dim} features, "
          f"{conv_params:,} params in the encoder")
    assert out.shape == (1, ex.features_dim)
    assert torch.isfinite(out).all()
    print("EXTRACTOR OK.")
