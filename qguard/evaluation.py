from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Any

import numpy as np

from sivc_marl.config import EvaluationConfig
from sivc_marl.envs.iov_env import IoVEnv
from sivc_marl.model import ACTION_NAMES, SIVCMAPPOPolicy
from sivc_marl.utils.metrics import EpisodeMetrics


def evaluate_policy(
    policy: SIVCMAPPOPolicy,
    cfg: EvaluationConfig,
    seeds: Iterable[int],
    deterministic: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = IoVEnv(cfg.env, cfg.value, cfg.consensus, seed=int(seed))
        observation = env.reset()
        metrics = EpisodeMetrics()
        action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
        policy_time = 0.0
        done = False
        rng = np.random.default_rng(int(seed))

        while not done:
            started = perf_counter()
            actions = policy.act(
                observation,
                cfg.env,
                deterministic=deterministic,
                rng=rng,
            )
            policy_time += perf_counter() - started
            action_counts += np.bincount(actions, minlength=len(ACTION_NAMES))
            observation, _, done, info = env.step(actions)
            metrics.update(info["metrics"])

        row = metrics.summary()
        action_total = max(int(action_counts.sum()), 1)
        row.update(
            {
                f"action_{name}_ratio": float(action_counts[index] / action_total)
                for index, name in enumerate(ACTION_NAMES)
            }
        )
        row.update(
            {
                "evaluation_seed": int(seed),
                "policy_decision_ms": 1000.0 * policy_time / max(metrics.steps, 1),
                "evaluation_mode": "argmax" if deterministic else "sampled",
            }
        )
        rows.append(row)
    return rows
