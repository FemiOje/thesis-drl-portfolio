"""EIIE feature extractor, shared identically by PG, PPO and DDPG.

Every kernel has height 1, so the asset axis is never convolved over, flattened or
reduced. It survives as an independent axis until the softmax. That is the controlled
variable of the whole study.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EIIEExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_assets=8, window=20, n_features=3,
                 n_conv1=2, n_conv2=20):
        super().__init__(observation_space, features_dim=n_assets + 1)
        self.n_assets, self.window, self.n_features = n_assets, window, n_features
        self.conv1 = nn.Conv2d(n_features, n_conv1, (1, 3))          # -> (B, 2, m, n-2)
        self.conv2 = nn.Conv2d(n_conv1, n_conv2, (1, window - 2))    # -> (B, 20, m, 1)
        self.conv3 = nn.Conv2d(n_conv2 + 1, 1, (1, 1))               # +1 = w_{t-1} channel
        self.cash_bias = nn.Parameter(torch.zeros(1))
        # Eq. 18 inputs sit at ~1.0, so a random bias dominates the ~0.05 signal and
        # ReLU kills whole channels: with 2 filters, 14/50 seeds start fully dead
        # (constant output, zero gradient, no learning). Zero biases + the centring in
        # forward() give 0/50. ReLU itself is unchanged.
        for c in (self.conv1, self.conv2, self.conv3):
            nn.init.zeros_(c.bias)

    def forward(self, obs):
        x, w_prev = obs["tensor"], obs["weights"]
        assert x.shape[1:] == (self.n_features, self.n_assets, self.window), \
            f"expected (B, {self.n_features}, {self.n_assets}, {self.window}), got {tuple(x.shape)}"
        x = torch.relu(self.conv1(x - 1.0))    # centre; see __init__
        x = torch.relu(self.conv2(x))
        w = w_prev[:, 1:].reshape(-1, 1, self.n_assets, 1)            # risky weights only
        x = self.conv3(torch.cat([x, w], dim=1))                      # (B, 1, m, 1)
        logits = x.reshape(-1, self.n_assets)
        cash = self.cash_bias.expand(logits.shape[0], 1)
        return torch.cat([cash, logits], dim=1)                       # (B, m+1), NOT weights
