# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K evaluation using vLLM server and isolated GSM8K script.
Replacement for lm-eval-harness with better performance and control.

Usage:
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.txt
"""

import argparse
import sys
import statistics
from pathlib import Path

import shlex

import pytest
import yaml

sys.path.append('/home/boh/vllm_source')
sys.path.append('/home/boh/vllm_source/tests')
from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from gsm8k_eval import evaluate_gsm8k

TOL = 0.08  # Absolute tolerance for accuracy comparison


def run_gsm8k_eval(eval_config: dict, server_url: str, seed: int = 42) -> dict:
    """Run GSM8K evaluation using our isolated script."""
    # Extract host and port from server URL
    if "://" in server_url:
        server_url = server_url.split("://")[1]

    host_port = server_url.split("/")[0]  # Remove path if present
    if ":" in host_port:
        host, p = host_port.split(":")
        port = int(p)
    else:
        host = host_port
        port = 8000

    # Add http:// prefix if not present
    if not host.startswith("http"):
        host = f"http://{host}"

    # Run GSM8K evaluation.
    # request_timeout=None disables the per-session timeout so slow backends
    # (e.g. CPU-only) can still finish all questions without hitting the
    # default 600 s aiohttp limit.
    results = evaluate_gsm8k(
        num_questions=eval_config["num_questions"],
        num_shots=eval_config["num_fewshot"],
        host=host,
        port=port,
        seed=seed,
        request_timeout=eval_config.get("request_timeout", None),
    )

    return results


def _print_run_result(run_idx: int, results: dict) -> None:
    print(
        f"  Run {run_idx + 1}: "
        f"accuracy={results['accuracy']:.4f}  "
        f"invalid={results['invalid_rate']:.3f}  "
        f"latency={results['latency']:.1f}s  "
        f"tok/s={results['tokens_per_second']:.1f}  "
        f"q/s={results['questions_per_second']:.3f}"
    )


def _print_stats(label: str, values: list[float], fmt: str = ".4f") -> None:
    if len(values) == 1:
        print(f"    {label}: {values[0]:{fmt}}")
        return
    mean = statistics.mean(values)
    std  = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"    {label}: {mean:{fmt}} ± {std:{fmt}}"
        f"  [min={min(values):{fmt}}, max={max(values):{fmt}}]"
    )


def test_gsm8k_correctness(config_filename):
    """Test GSM8K correctness for a given model configuration."""
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    if (
        not current_platform.is_cuda()
        and "Qwen3-30B-A3B-MXFP4A16" in eval_config["model_name"]
    ):
        pytest.skip(
            "Skipping Qwen3-30B-A3B-MXFP4A16 on non-CUDA platforms. "
            "Marlin kernels are not supported."
        )

    num_runs: int = eval_config.get("num_runs", 1)
    # Use a fixed sequence of seeds so runs are reproducible yet distinct
    base_seed: int = eval_config.get("seed", 42)
    seeds = [base_seed + i for i in range(num_runs)]

    # Parse server arguments from config (use shlex to handle quoted strings)
    server_args_str = eval_config.get("server_args", "")
    server_args = shlex.split(server_args_str) if server_args_str else []

    # Add standard server arguments
    server_args.extend(
        [
            "--trust-remote-code",
            "--disable-uvicorn-access-log",
        ]
    )

    env_dict = eval_config.get("env", None)

    print(f"\n{'='*70}")
    print(f"GSM8K evaluation — {eval_config['model_name']}")
    print(f"  Questions : {eval_config['num_questions']}")
    print(f"  Few-shot  : {eval_config['num_fewshot']}")
    print(f"  Runs      : {num_runs}  (seeds: {seeds})")
    print(f"  Threshold : {eval_config['accuracy_threshold']:.4f}  (tol ±{TOL:.4f})")
    print(f"  Server    : {' '.join(server_args)}")
    print(f"{'='*70}")

    all_results: list[dict] = []

    # Launch the server once; run all evaluation passes against it.
    with RemoteOpenAIServer(
        eval_config["model_name"],
        server_args,
        env_dict=env_dict,
        max_wait_seconds=eval_config.get("startup_max_wait_seconds", 600),
    ) as remote_server:
        server_url = remote_server.url_for("v1")
        print(f"Server started at: {server_url}\n")

        for run_idx, seed in enumerate(seeds):
            print(f"--- Run {run_idx + 1}/{num_runs}  (seed={seed}) ---")
            results = run_gsm8k_eval(eval_config, server_url, seed=seed)
            all_results.append(results)
            _print_run_result(run_idx, results)

    # ------------------------------------------------------------------ #
    # Aggregate statistics
    # ------------------------------------------------------------------ #
    accuracies      = [r["accuracy"]              for r in all_results]
    invalid_rates   = [r["invalid_rate"]          for r in all_results]
    latencies       = [r["latency"]               for r in all_results]
    toks_per_sec    = [r["tokens_per_second"]      for r in all_results]
    q_per_sec       = [r["questions_per_second"]   for r in all_results]
    total_out_toks  = [r["total_output_tokens"]    for r in all_results]

    print(f"\n{'='*70}")
    print(f"Summary — {eval_config['model_name']}  ({num_runs} run(s))")
    print(f"{'='*70}")
    _print_stats("Accuracy          ", accuracies,    fmt=".4f")
    _print_stats("Invalid rate      ", invalid_rates, fmt=".4f")
    _print_stats("Latency (s)       ", latencies,     fmt=".1f")
    _print_stats("Tokens/s          ", toks_per_sec,  fmt=".2f")
    _print_stats("Questions/s       ", q_per_sec,     fmt=".4f")
    _print_stats("Total output toks ", total_out_toks, fmt=".0f")
    print(f"{'='*70}\n")

    # Pass/fail decision: use the mean accuracy across runs
    mean_accuracy    = statistics.mean(accuracies)
    expected_metric  = eval_config["accuracy_threshold"]

    assert mean_accuracy >= expected_metric - TOL, (
        f"GSM8K mean accuracy too low: {mean_accuracy:.4f} < "
        f"{expected_metric:.4f} - {TOL:.4f} = {expected_metric - TOL:.4f}"
    )

    print(f"✅ GSM8K test passed for {eval_config['model_name']}")


def main():
    """Main entry point for direct script execution."""
    parser = argparse.ArgumentParser(
        description="GSM8K evaluation for vLLM models"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to model configuration file (YAML)"
    )
    args = parser.parse_args()

    args.config_file = Path(args.config_file.replace('config_file=', ''))
    if not args.config_file.exists():
        print(f"Error: Config file not found: {args.config_file}", file=sys.stderr)
        return 1

    try:
        test_gsm8k_correctness(args.config_file)
        return 0
    except AssertionError as e:
        print(f"Test failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
