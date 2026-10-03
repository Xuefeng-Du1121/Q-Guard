"""
Basic usage example for Q-Guard

This demonstrates how to use RPE operator and Q-Guard audit metrics.
"""

import torch
from qguard.rpe import RPEOperator, compute_Q, compute_kappa


def example_rpe():
    """Example: Using RPE for deterministic execution"""
    print("=== RPE Operator Example ===\n")
    
    # Initialize RPE operator
    num_agents = 3
    num_actions = 4
    rpe = RPEOperator(num_agents, num_actions)
    
    # Simulate policy logits (unnormalized log probabilities)
    logits = torch.randn(num_agents, num_actions)
    print(f"Policy logits:\n{logits}\n")
    
    # Execute actions with RPE
    actions = rpe.execute(logits)
    print(f"Selected actions (RPE): {actions}\n")
    
    # Compare with argmax
    argmax_actions = torch.argmax(logits, dim=-1)
    print(f"Argmax actions: {argmax_actions}\n")


def example_qguard_audit():
    """Example: Q-Guard audit metrics"""
    print("=== Q-Guard Audit Example ===\n")
    
    # Simulated protocol logs
    events_reported = 100
    events_reaching_committee_sampling = 85
    events_reaching_committee_argmax = 12
    
    # Compute Q for sampling vs argmax
    Q_sampling = compute_Q(events_reported, events_reaching_committee_sampling)
    Q_argmax = compute_Q(events_reported, events_reaching_committee_argmax)
    
    print(f"Quorum Reach Rate (Q):")
    print(f"  Sampling: {Q_sampling:.3f}")
    print(f"  Argmax:   {Q_argmax:.3f}\n")
    
    # Check Q-Guard alarm (threshold ρ = 0.5)
    ratio = Q_argmax / Q_sampling
    print(f"Q ratio: {ratio:.3f}")
    if ratio < 0.5:
        print("⚠️  Q-Guard ALARM: Quorum collapse detected!\n")
    else:
        print("✓ Q-Guard: No alarm\n")
    
    # Compute commitment ratio κ
    events_executed_sampling = 82
    events_executed_argmax = 8
    event_hazard_sum = 78.5  # Sum of policy probabilities
    
    kappa_sampling = compute_kappa(events_executed_sampling, event_hazard_sum)
    kappa_argmax = compute_kappa(events_executed_argmax, event_hazard_sum)
    
    print(f"Commitment Ratio (κ):")
    print(f"  Sampling: {kappa_sampling:.2f}")
    print(f"  Argmax:   {kappa_argmax:.2f}")
    print(f"\nArgmax commits at {kappa_argmax/kappa_sampling*100:.0f}% of the policy's intended rate")


if __name__ == "__main__":
    example_rpe()
    print("\n" + "="*50 + "\n")
    example_qguard_audit()
