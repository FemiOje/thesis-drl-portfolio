"""Stack smoke test.

Trains Stable-Baselines3 PPO on Pendulum-v1 for 10k timesteps to confirm the
whole stack (torch + gymnasium + stable-baselines3) is installed and functional
before any custom thesis code is written. This is deliberately NOT the portfolio
environment — it only exercises the RL plumbing.

Run:  python experiments/smoke_test.py
"""

import sys
import time

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO


def main() -> int:
    print("=== Smoke test: PPO on Pendulum-v1 ===")
    print(f"python            : {sys.version.split()[0]}")
    print(f"torch             : {torch.__version__} (cuda={torch.cuda.is_available()})")
    print(f"stable-baselines3 : {sb3.__version__}")
    print(f"gymnasium         : {gym.__version__}")
    print(f"numpy             : {np.__version__}")
    print()

    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env, verbose=1, seed=0, device="cpu")

    t0 = time.time()
    model.learn(total_timesteps=10_000)
    elapsed = time.time() - t0
    print(f"\nTraining completed 10,000 timesteps in {elapsed:.1f}s")

    # Quick sanity rollout: run one deterministic episode, report total reward.
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    env.close()
    print(f"Eval episode total reward: {total_reward:.1f} "
          "(Pendulum rewards are negative; closer to 0 is better)")

    print("\nSMOKE TEST PASSED: RL stack is working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
