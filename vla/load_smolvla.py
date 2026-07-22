"""Load the public SmolVLA checkpoint (lerobot/smolvla_base).

Runs inside the Docker env (lerobot installed there, never on the host).
The lerobot SmolVLA API has moved across releases; pin a known-good version in
the Dockerfile. This module isolates the import so the rest of the harness is
version-agnostic.
"""
from __future__ import annotations

import torch

CHECKPOINT = "lerobot/smolvla_base"


def load_policy(device: str = "cuda", checkpoint: str = CHECKPOINT):
    """Return a SmolVLA policy in eval mode on `device`."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device).eval()
    return policy


def dummy_observation(policy, device: str = "cuda") -> dict:
    """A synthetic observation batch for numerical-parity / latency tests.

    Shapes are derived from the policy config so we don't hard-code dataset
    specifics. Replace with a real LeRobot dataset sample for sim eval.
    """
    cfg = policy.config
    obs = {}
    # image inputs
    for key, ft in getattr(cfg, "input_features", {}).items():
        shape = tuple(ft.shape)
        obs[key] = torch.randn(1, *shape, device=device)
    if not obs:  # fallback if introspection differs across lerobot versions
        obs["observation.image"] = torch.randn(1, 3, 256, 256, device=device)
        obs["observation.state"] = torch.randn(1, 7, device=device)
    obs["task"] = ["pick up the cube"]
    return obs


if __name__ == "__main__":
    p = load_policy()
    n = sum(x.numel() for x in p.parameters())
    print(f"loaded {CHECKPOINT}: {n/1e6:.1f}M params on {next(p.parameters()).device}")
