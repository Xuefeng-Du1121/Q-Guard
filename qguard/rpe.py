"""
Rate-Preserving Execution (RPE) Operator

A deterministic execution operator that preserves per-agent action rates
through error diffusion, preventing mode-rate mismatch in multi-agent systems.
"""

import torch
from typing import Optional


class RPEOperator:
    """
    Rate-Preserving Execution operator using Floyd-Steinberg error diffusion.
    
    Maintains per-agent cumulative error to preserve long-term execution rates
    while making deterministic decisions at each timestep.
    """
    
    def __init__(self, num_agents: int, num_actions: int):
        """
        Args:
            num_agents: Number of agents in the system
            num_actions: Number of possible actions per agent
        """
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.error = torch.zeros(num_agents, num_actions)
    
    def execute(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Execute actions using RPE with error diffusion.
        
        Args:
            logits: Action logits of shape [num_agents, num_actions]
        
        Returns:
            actions: Selected actions of shape [num_agents]
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Add accumulated error
        adjusted = probs + self.error
        
        # Select argmax on adjusted distribution
        actions = torch.argmax(adjusted, dim=-1)
        
        # Compute one-hot target
        target = torch.zeros_like(probs)
        target.scatter_(1, actions.unsqueeze(1), 1.0)
        
        # Update error (diffuse the rounding error)
        self.error = adjusted - target
        
        return actions
    
    def reset(self):
        """Reset error accumulator (call at episode boundaries)"""
        self.error.zero_()


def compute_Q(events_reported: int, events_reaching_committee: int) -> float:
    """
    Compute quorum reach rate Q.
    
    Q = |E_reach| / |E_att|
    
    Args:
        events_reported: Number of events reported by any agent
        events_reaching_committee: Number of events that reached committee size k
    
    Returns:
        Q: Quorum reach rate in [0, 1]
    """
    if events_reported == 0:
        return 0.0
    return events_reaching_committee / events_reported


def compute_kappa(events_executed: int, event_hazard_sum: float) -> float:
    """
    Compute commitment ratio κ.
    
    κ = events_executed / Σπ_t(C_e)
    
    Args:
        events_executed: Number of events actually executed
        event_hazard_sum: Sum of policy probabilities for commit actions
    
    Returns:
        κ: Commitment ratio (ideally ~1.0)
    """
    if event_hazard_sum == 0:
        return 0.0
    return events_executed / event_hazard_sum
