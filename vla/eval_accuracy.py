"""Accuracy parity: original vs. kernel-tuned SmolVLA.

Two layers (see plan):
  1. Numerical parity  — identical observations -> compare action tensors
     (max-abs / MSE / cosine). Fast, deterministic, no env. The regression gate.
  2. Task success rate — sim rollouts via LeRobot eval (the real metric). Hooked
     here; enable with --rollouts once a sim env is configured.

Usage (in Docker):
    python3 vla/eval_accuracy.py --variant original --out results/acc_original.json
    python3 vla/eval_accuracy.py --variant tuned --baseline results/acc_original.json
"""
from __future__ import annotations

import argparse
import json

import torch

from vla.load_smolvla import load_policy, dummy_observation
from vla.patch_kernels import use_custom_kernels


@torch.no_grad()
def get_action(policy, obs):
    # lerobot policies expose select_action; fall back to forward if renamed
    fn = getattr(policy, "select_action", None) or policy.forward
    return fn(obs)


@torch.no_grad()
def numerical_parity(policy, n: int = 16, seed: int = 0):
    """Compare actions with vs without the custom kernels on identical inputs."""
    torch.manual_seed(seed)
    obs_list = [dummy_observation(policy) for _ in range(n)]

    base, tuned = [], []
    for obs in obs_list:
        base.append(get_action(policy, obs).float())
    with use_custom_kernels(True):
        for obs in obs_list:
            tuned.append(get_action(policy, obs).float())

    b = torch.stack(base)
    t = torch.stack(tuned)
    return {
        "max_abs_err": (b - t).abs().max().item(),
        "mse": ((b - t) ** 2).mean().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            b.flatten(1), t.flatten(1), dim=1).mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["original", "tuned"], default="tuned")
    ap.add_argument("--rollouts", type=int, default=0,
                    help="sim episodes for task success rate (0 = skip)")
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="", help="baseline json to diff against")
    args = ap.parse_args()

    policy = load_policy()
    result = {"variant": args.variant}
    result["numerical_parity"] = numerical_parity(policy)

    if args.rollouts:
        # TODO: wire LeRobot sim eval (gym env + rollout loop) -> success rate.
        # success = run_sim_rollouts(policy, episodes=args.rollouts,
        #                            patched=(args.variant == "tuned"))
        result["task_success_rate"] = None
        print("note: --rollouts requested but sim eval not yet wired (roadmap)")

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)
        print("\n[parity vs baseline]")
        print(f"  max_abs_err: {result['numerical_parity']['max_abs_err']:.2e}")
        print(f"  cosine:      {result['numerical_parity']['cosine']:.6f}")


if __name__ == "__main__":
    main()
