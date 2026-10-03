from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sivc_marl.features import extract_agent_features

ACTION_NAMES = ("stay", "north", "south", "east", "west", "verify", "serve")


class ActorNetwork(nn.Module):
    """Parameter-shared decentralized actor used by SIVC-MAPPO."""

    def __init__(self, obs_dim: int = 14, action_dim: int = 7, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class SIVCMAPPOPolicy:
    """Inference wrapper for the released SIVC-MAPPO actor weights."""

    def __init__(self, actor: ActorNetwork, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> SIVCMAPPOPolicy:
        actor = ActorNetwork()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError("Expected a state dictionary in the released weights file")
        actor.load_state_dict(state_dict, strict=True)
        return cls(actor, device=device)

    @torch.inference_mode()
    def logits(self, agent_features: np.ndarray | torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(
            agent_features,
            dtype=torch.float32,
            device=self.device,
        )
        if features.ndim != 2 or features.shape[1] != 14:
            raise ValueError("agent_features must have shape [num_uavs, 14]")
        return self.actor(features)

    def act(
        self,
        observation: dict[str, Any],
        env_cfg: Any,
        deterministic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        scores = self.logits(extract_agent_features(observation, env_cfg))
        if deterministic:
            return scores.argmax(dim=-1).cpu().numpy().astype(int).tolist()

        probabilities = torch.softmax(scores, dim=-1).cpu().numpy()
        generator = rng or np.random.default_rng()
        return [
            int(generator.choice(len(ACTION_NAMES), p=agent_probabilities))
            for agent_probabilities in probabilities
        ]
