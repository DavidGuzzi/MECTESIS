#!/usr/bin/env python
"""
CLI script to run Monte Carlo experiments from YAML configuration files.

Usage:
    python scripts/run_experiment.py experiments/configs/ar1_simple.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mectesis.dgp import AR1
from mectesis.models import ARIMAModel, ChronosModel
from mectesis.simulation import MonteCarloEngine


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dgp(dgp_config: dict, seed: int):
    """Create DGP instance from configuration."""
    dgp_type = dgp_config['type']

    if dgp_type == "AR1":
        return AR1(seed=seed)
    else:
        raise ValueError(f"Unknown DGP type: {dgp_type}")


def create_models(models_config: list) -> list:
    """Create model instances from configuration."""
    models = []

    for model_cfg in models_config:
        model_type = model_cfg['type']
        params = model_cfg.get('params', {})

        if model_type == "arima":
            models.append(ARIMAModel(order=tuple(params['order'])))
        elif model_type == "chronos":
            models.append(ChronosModel(
                model_size=params['model_size'],
                device=params['device']
            ))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return models


def run_experiment(config_path: str):
    """Run Monte Carlo experiment from configuration file."""
    print("="*70)
    print("MECTESIS - Monte Carlo Experiment Runner")
    print("="*70)

    # Load configuration
    print(f"\n[1/4] Loading configuration from: {config_path}")
    config = load_config(config_path)

    exp_name = config['experiment']['name']
    exp_desc = config['experiment']['description']
    print(f"  Experiment: {exp_name}")
    print(f"  Description: {exp_desc}")

    # Extract parameters
    dgp_config = config['dgp']
    sim_config = config['simulation']
    models_config = config['models']

    seed = sim_config['seed']
    T = sim_config['T']
    horizon = sim_config['horizon']
    n_sim = sim_config['n_sim']
    dgp_params = dgp_config['params']

    # Create DGP
    print(f"\n[2/4] Initializing DGP: {dgp_config['type']}")
    dgp = create_dgp(dgp_config, seed=seed)
    print(f"  Parameters: {dgp_params}")

    # Create models
    print(f"\n[3/4] Initializing {len(models_config)} model(s):")
    models = create_models(models_config)
    for model in models:
        print(f"  - {model.name}")

    # Run simulation
    print(f"\n[4/4] Running Monte Carlo simulation...")
    engine = MonteCarloEngine(dgp=dgp, models=models, seed=seed)

    results = engine.run_monte_carlo(
        n_sim=n_sim,
        T=T,
        horizon=horizon,
        dgp_params=dgp_params,
        verbose=True
    )

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for model_name, metrics_df in results.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        print(metrics_df.to_string(index=False))

    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("="*70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo forecasting experiments from YAML config"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    try:
        run_experiment(args.config)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
