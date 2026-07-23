"""Accuracy parity vs the eager original, per variant (the MLPerf-gate proxy).

The action expert is flow-matching: it draws starting noise from the global
RNG, so inference is stochastic (self-diff ~2.9 unseeded, exactly 0 when
seeded). Seeding identically before every call removes that, leaving only the
variant-substitution difference.

Gate (per MLPerf methodology): a variant is reportable only if it retains
>= 99% of the reference metric — until sim rollouts exist, the proxy is
numerical parity (cosine similarity of action chunks).

Usage (in Docker):
    python3 vla/eval_accuracy.py --variant original --out results/acc_original.json
    python3 vla/eval_accuracy.py --variant compiled --baseline results/acc_original.json
"""
from __future__ import annotations

import argparse
import json

import torch

from vla.load_smolvla import load_policy, dummy_observation
from vla.variants import VARIANTS, base_infer, make_infer


@torch.no_grad()
def get_action(policy, obs):
    """One full model inference (chunk forward — select_action pops a queue)."""
    return base_infer(policy)(obs)


@torch.no_grad()
def numerical_parity(policy, variant: str, n: int = 16, seed: int = 0):
    """Seeded eager reference vs the variant on identical inputs."""
    torch.manual_seed(seed)
    obs_list = [dummy_observation(policy) for _ in range(n)]

    ref_fn = base_infer(policy)
    ref = []
    for i, obs in enumerate(obs_list):
        torch.manual_seed(seed + i)
        ref.append(ref_fn(obs).float().clone())

    ctx, fn = make_infer(policy, variant)
    out = []
    with ctx:
        fn(obs_list[0])  # warm/compile outside the seeded comparison
        for i, obs in enumerate(obs_list):
            torch.manual_seed(seed + i)
            # clone: under CUDA graphs (compiled-ro) outputs live in static
            # buffers that the next replay overwrites
            out.append(fn(obs).float().clone())

    b, t = torch.stack(ref), torch.stack(out)
    cos = torch.nn.functional.cosine_similarity(
        b.flatten(1), t.flatten(1), dim=1).mean().item()
    return {
        "max_abs_err": (b - t).abs().max().item(),
        "mse": ((b - t) ** 2).mean().item(),
        "cosine": cos,
        "retention_gate_99": bool(cos >= 0.99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="original")
    ap.add_argument("--rollouts", type=int, default=0,
                    help="sim episodes for task success rate (0 = skip; roadmap)")
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="", help="baseline json to diff against")
    args = ap.parse_args()

    policy = load_policy()
    result = {"variant": args.variant,
              "numerical_parity": numerical_parity(policy, args.variant)}

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    if args.baseline:
        with open(args.baseline) as f:
            json.load(f)  # existence check; parity above is already vs eager
        p = result["numerical_parity"]
        print(f"\n[gate] cosine {p['cosine']:.6f} -> "
              f"{'PASS' if p['retention_gate_99'] else 'FAIL'} (>=0.99 proxy)")


if __name__ == "__main__":
    main()
