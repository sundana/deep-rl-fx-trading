"""Apply best hyperparameters from Optuna tuning results to config.yaml.

Usage
-----
# Apply best PPO params:
python apply_best_params.py --algo PPO

# Apply best RecurrentPPO params:
python apply_best_params.py --algo RecurrentPPO

# Dry-run (preview without writing):
python apply_best_params.py --algo PPO --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


# Params written directly into config.yaml train section
_TRAIN_KEYS = [
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
]

_RPPO_KEYS = ["lstm_hidden_size", "n_lstm_layers"]


def load_best_params(algo: str, tuning_dir: Path) -> dict:
    path = tuning_dir / f"best_params_{algo.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No tuning results for {algo} at {path}.\n"
            f"Run: python tune.py --algo {algo}"
        )
    with path.open() as f:
        return json.load(f)


def apply_to_config(
    best: dict,
    algo: str,
    config_path: Path,
    dry_run: bool = False,
) -> None:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    params = best["params"]
    attrs  = best.get("user_attrs", {})
    score  = best.get("score", "?")

    print(f"\n{'='*55}")
    print(f" Applying best {algo} params  (score={score:.4f})")
    print(f"{'='*55}")
    print(f"  Sharpe:       {attrs.get('sharpe', '?')}")
    print(f"  Total return: {attrs.get('total_return', '?'):.2%}")
    print(f"  Max drawdown: {attrs.get('max_drawdown', '?'):.2%}")
    print(f"  # trades:     {attrs.get('n_trades', '?')}")
    print()

    changes: list[str] = []

    # Always set the algo
    if cfg["train"].get("algo") != algo:
        changes.append(f"  algo:  {cfg['train'].get('algo')}  →  {algo}")
        cfg["train"]["algo"] = algo

    # Scalar hyperparameters
    for key in _TRAIN_KEYS:
        if key not in params:
            continue
        old = cfg["train"].get(key, "<unset>")
        new = params[key]
        if old != new:
            changes.append(f"  {key}:  {old}  →  {new}")
            cfg["train"][key] = new

    # RecurrentPPO-specific keys
    if algo.upper() == "RECURRENTPPO":
        for key in _RPPO_KEYS:
            if key not in params:
                continue
            old = cfg["train"].get(key, "<unset>")
            new = params[key]
            if old != new:
                changes.append(f"  {key}:  {old}  →  {new}")
                cfg["train"][key] = new

    # net_arch (stored as dict in JSON, must become nested YAML)
    if "net_arch" in params:
        old = cfg["train"].get("net_arch", "<unset>")
        new = params["net_arch"]
        if old != new:
            changes.append(f"  net_arch:  {old}  →  {new}")
            cfg["train"]["net_arch"] = new

    if not changes:
        print("  No changes needed — config already matches best params.")
        return

    print("  Changes to config.yaml:")
    for line in changes:
        print(line)

    if dry_run:
        print("\n  [dry-run] config.yaml NOT written.")
        return

    # Preserve YAML formatting as much as possible
    with config_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n  config.yaml updated → {config_path}")
    print("  Run `python train.py` to train with the new parameters.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply best Optuna tuning results to config.yaml"
    )
    parser.add_argument("--algo", choices=["PPO", "RecurrentPPO"], required=True)
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--tuning-dir",  default="results/tuning")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without modifying config.yaml"
    )
    args = parser.parse_args()

    best = load_best_params(args.algo, Path(args.tuning_dir))
    apply_to_config(
        best,
        algo=args.algo,
        config_path=Path(args.config),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
