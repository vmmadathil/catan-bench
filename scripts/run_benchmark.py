#!/usr/bin/env python3
"""CLI entry point for the Catan LLM Benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
import os

import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from catan_bench.runner import run_benchmark
from catan_bench.analysis import generate_reports
from catan_bench.providers.anthropic import AnthropicProvider
from catan_bench.providers.google import GoogleProvider

PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_providers(config: dict, model_filter: list[str] | None = None) -> dict:
    """Build provider instances from config."""
    providers = {}
    for name, model_cfg in config.get("models", {}).items():
        if model_filter and name not in model_filter:
            continue
        provider_name = model_cfg["provider"]
        model_id = model_cfg["model_id"]
        cls = PROVIDER_CLASSES.get(provider_name)
        if cls is None:
            logging.warning(f"Unknown provider '{provider_name}' for model '{name}', skipping")
            continue
        providers[name] = cls(model_id=model_id)
    return providers


def main():
    parser = argparse.ArgumentParser(
        description="Catan LLM Benchmark — pit frontier LLMs against each other in Settlers of Catan"
    )
    parser.add_argument(
        "--config", "-c",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"),
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=None,
        help="Number of games to run (overrides config)",
    )
    parser.add_argument(
        "--no-trade",
        action="store_true",
        help="Disable domestic trading",
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        help="Models to include (space-separated names from config)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for reports (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Starting random seed (overrides config)",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=None,
        help="Max concurrent games (default: 4, use 1 for sequential)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    num_games = args.games or config.get("num_games", 24)
    enable_trade = not args.no_trade and config.get("enable_trade", True)
    output_dir = args.output or config.get("output_dir", "results")
    seed_start = args.seed if args.seed is not None else config.get("seed_start", 42)
    max_parallel = args.parallel if args.parallel is not None else config.get("max_parallel", 4)
    log_level = args.log_level or config.get("log_level", "INFO")

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("catan_bench")

    # Suppress noisy HTTP loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    # Build providers
    providers = build_providers(config, args.models)
    if len(providers) < 2:
        logger.error(f"Need at least 2 models, got {len(providers)}: {list(providers.keys())}")
        logger.error("Check your config and API keys. Use --models to filter.")
        sys.exit(1)

    logger.info(f"Models: {list(providers.keys())}")
    logger.info(f"Games: {num_games} | Trading: {enable_trade} | Seed: {seed_start} | Parallel: {max_parallel}")

    # Pad to 4 players if fewer models (duplicate the list)
    if len(providers) < 4:
        model_names = list(providers.keys())
        while len(model_names) < 4:
            model_names.append(model_names[len(model_names) % len(providers)])
        # Create duplicate provider entries with suffixed names
        padded = {}
        seen = {}
        for name in model_names:
            count = seen.get(name, 0)
            seen[name] = count + 1
            if count > 0:
                padded_name = f"{name}#{count + 1}"
                padded[padded_name] = providers[name]
            else:
                padded[name] = providers[name]
        providers = padded
        logger.info(f"Padded to 4 players: {list(providers.keys())}")

    # Run benchmark
    metrics = run_benchmark(
        providers=providers,
        num_games=num_games,
        enable_trade=enable_trade,
        seed_start=seed_start,
        max_parallel=max_parallel,
        output_dir=output_dir,
    )

    # Generate reports
    report_paths = generate_reports(metrics, output_dir=output_dir)
    logger.info(f"\nReports generated:")
    for name, path in report_paths.items():
        logger.info(f"  {name}: {path}")

    # Print summary to stdout
    if "summary" in report_paths:
        with open(report_paths["summary"]) as f:
            print(f.read())


if __name__ == "__main__":
    main()
