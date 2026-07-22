"""MLPerf-style measurement of SmolVLA with MLCommons LoadGen.

Methodology (MLPerf Inference, edge):
  - SingleStream : one query at a time; headline = p90 latency. Closest match
    to a robot control loop.
  - Offline      : all queries issued at once; headline = throughput.
  - Server is datacenter-oriented (Poisson arrivals, p99 QPS) and is skipped.

The accuracy side of the MLPerf gate (tuned retains >= 99% of the original's
metric) lives in vla/eval_accuracy.py; this file measures performance only.

Usage (in Docker):
    python3 vla/mlperf_harness.py --variant original --scenario SingleStream
    python3 vla/mlperf_harness.py --variant tuned --scenario Offline
Logs land under results/mlperf/<variant>/<scenario>/.
"""
from __future__ import annotations

import argparse
import os

import torch

import mlperf_loadgen as lg

from vla.load_smolvla import load_policy, dummy_observation
from vla.patch_kernels import use_custom_kernels
from vla.eval_accuracy import get_action

SCENARIOS = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
}


class SmolVLASut:
    """SUT: each LoadGen query is one policy inference on a pooled observation."""

    def __init__(self, policy, obs_pool):
        self.policy = policy
        self.obs_pool = obs_pool

    def issue_queries(self, samples):
        for s in samples:
            obs = self.obs_pool[s.index % len(self.obs_pool)]
            with torch.no_grad():
                get_action(self.policy, obs)
            torch.cuda.synchronize()
            resp = lg.QuerySampleResponse(s.id, 0, 0)
            lg.QuerySamplesComplete([resp])

    def flush(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["original", "tuned"], default="original")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="SingleStream")
    ap.add_argument("--pool-size", type=int, default=16,
                    help="distinct observations cycled through queries")
    ap.add_argument("--min-queries", type=int, default=270,
                    help="minimum query count (MLPerf edge default is higher; "
                         "this keeps local runs short)")
    ap.add_argument("--min-duration-ms", type=int, default=60_000)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join("results", "mlperf", args.variant,
                                         args.scenario)
    os.makedirs(outdir, exist_ok=True)

    policy = load_policy()
    torch.manual_seed(0)
    obs_pool = [dummy_observation(policy) for _ in range(args.pool_size)]

    settings = lg.TestSettings()
    settings.scenario = SCENARIOS[args.scenario]
    settings.mode = lg.TestMode.PerformanceOnly
    settings.min_query_count = args.min_queries
    settings.min_duration_ms = args.min_duration_ms

    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = outdir
    log_settings.log_output.copy_summary_to_stdout = True

    sut_impl = SmolVLASut(policy, obs_pool)
    sut = lg.ConstructSUT(sut_impl.issue_queries, sut_impl.flush)
    qsl = lg.ConstructQSL(args.pool_size, args.pool_size,
                          lambda s: None, lambda s: None)  # obs stay resident

    with use_custom_kernels(args.variant == "tuned"):
        # warmup outside the timed window
        for _ in range(10):
            get_action(policy, obs_pool[0])
        torch.cuda.synchronize()
        lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    print(f"logs -> {outdir}/mlperf_log_summary.txt")


if __name__ == "__main__":
    main()
