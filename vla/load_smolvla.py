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
    try:  # lerobot >= 0.6
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:  # older layout
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device).eval()
    return policy


from functools import lru_cache


@lru_cache(maxsize=1)
def _tokenizer(vlm_model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(vlm_model_name)


def dummy_observation(policy, device: str = "cuda",
                      task: str = "pick up the cube") -> dict:
    """A synthetic observation batch for numerical-parity / latency tests.

    Shapes come from the policy config; the language field is pre-tokenized
    with the VLM's own tokenizer, matching what the lerobot preprocessing
    pipeline feeds `select_action` (observation.language.tokens + mask).
    Replace with a real LeRobot dataset sample for sim eval.
    """
    cfg = policy.config
    obs = {}
    # image / state inputs from the config's feature spec
    for key, ft in getattr(cfg, "input_features", {}).items():
        shape = tuple(ft.shape)
        obs[key] = torch.randn(1, *shape, device=device)
    if not obs:  # fallback if introspection differs across lerobot versions
        obs["observation.image"] = torch.randn(1, 3, 256, 256, device=device)
        obs["observation.state"] = torch.randn(1, 7, device=device)

    # language: tokenize like the lerobot pre-processor does
    tok = _tokenizer(getattr(cfg, "vlm_model_name",
                             "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"))
    enc = tok(task, return_tensors="pt", padding="max_length",
              truncation=True,
              max_length=getattr(cfg, "tokenizer_max_length", 48))
    obs["observation.language.tokens"] = enc["input_ids"].to(device)
    obs["observation.language.attention_mask"] = (
        enc["attention_mask"].to(device, dtype=torch.bool))
    obs["task"] = [task]
    return obs


if __name__ == "__main__":
    p = load_policy()
    n = sum(x.numel() for x in p.parameters())
    print(f"loaded {CHECKPOINT}: {n/1e6:.1f}M params on {next(p.parameters()).device}")
