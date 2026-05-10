#!/usr/bin/env python3
"""Run training experiments from a JSON config file.

Example:

    python src/run_experiments.py --config experiments.json --dry-run
    python src/run_experiments.py --config experiments.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPEATABLE_ARGS = {"good_dir", "defect_dir"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand experiments.json and run src/main.py train for each job."
    )
    parser.add_argument("--config", default="experiments.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", default="src/main.py")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Run only the first N expanded jobs. Useful for testing.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="1-based expanded job index to start from.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose output folder already contains test_metrics.json.",
    )
    parser.add_argument(
        "--summary-dir",
        default="output/experiment_summaries",
        help="Where to write the launcher summary files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path) as handle:
        config = json.load(handle)

    jobs = expand_jobs(config)
    if args.start_at < 1:
        raise ValueError("--start-at is 1-based and must be >= 1")

    indexed_jobs = list(enumerate(jobs, start=1))
    indexed_jobs = [
        (index, job) for index, job in indexed_jobs if index >= args.start_at
    ]
    if args.max_runs is not None:
        indexed_jobs = indexed_jobs[: args.max_runs]

    if not indexed_jobs:
        print("No jobs selected.")
        return

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = summary_dir / f"experiment_summary_{stamp}.csv"
    summary_json = summary_dir / f"experiment_summary_{stamp}.json"

    print(f"Expanded jobs: {len(jobs)}")
    print(f"Selected jobs: {len(indexed_jobs)}")
    if args.dry_run:
        print("Dry run. No training commands will be executed.")

    results: List[Dict[str, Any]] = []
    for index, job in indexed_jobs:
        command = build_train_command(args.python, args.train_script, job)
        run_dir = Path(str(job["output_dir"])) / str(job["run_name"])
        metrics_path = run_dir / "test_metrics.json"

        if args.skip_existing and metrics_path.exists():
            print(f"[{index}/{len(jobs)}] SKIP existing: {job['run_name']}")
            result = make_result(index, job, command, status="skipped")
            result.update(read_metrics(metrics_path))
            results.append(result)
            write_summaries(results, summary_csv, summary_json)
            continue

        print(f"[{index}/{len(jobs)}] {job['run_name']}")
        print(" ".join(shell_quote(part) for part in command))

        result = make_result(index, job, command, status="dry-run")
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            result["return_code"] = completed.returncode
            result["status"] = "completed" if completed.returncode == 0 else "failed"
            if metrics_path.exists():
                result.update(read_metrics(metrics_path))

            write_summaries(results + [result], summary_csv, summary_json)
            if completed.returncode != 0:
                print(f"Stopping because job failed: {job['run_name']}")
                results.append(result)
                break

        results.append(result)
        write_summaries(results, summary_csv, summary_json)

    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


def expand_jobs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    defaults = dict(config.get("defaults", {}))
    experiments = config.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("'experiments' must be a list")

    jobs: List[Dict[str, Any]] = []
    for experiment in experiments:
        name = experiment["name"]
        base_args = dict(defaults)
        base_args.update(experiment.get("args", {}))
        grid = experiment.get("grid", {})

        for combo_index, combo in enumerate(expand_grid(grid), start=1):
            job = dict(base_args)
            job.update(combo)
            job["run_name"] = experiment.get(
                "run_name", make_run_name(name, combo_index, combo)
            )
            validate_job(job, name)
            jobs.append(job)

    return jobs


def expand_grid(grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return

    keys = list(grid.keys())
    values = []
    for key in keys:
        value = grid[key]
        if not isinstance(value, list):
            raise ValueError(f"Grid value for '{key}' must be a list")
        if not value:
            raise ValueError(f"Grid value for '{key}' is empty")
        values.append(value)

    for combo_values in itertools.product(*values):
        yield dict(zip(keys, combo_values))


def make_run_name(experiment_name: str, combo_index: int, combo: Dict[str, Any]) -> str:
    if not combo:
        return sanitize(f"{experiment_name}_{combo_index:02d}")

    parts = [experiment_name, f"{combo_index:02d}"]
    for key, value in combo.items():
        parts.append(f"{short_key(key)}{format_value(value)}")
    return sanitize("_".join(parts))


def short_key(key: str) -> str:
    mapping = {
        "learning_rate": "lr",
        "batch_size": "bs",
        "image_size": "img",
        "balance_strategy": "bal",
        "freeze_backbone_epochs": "freeze",
    }
    return mapping.get(key, key)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return str(value).replace(" ", "")


def sanitize(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def validate_job(job: Dict[str, Any], experiment_name: str) -> None:
    required = ["run_name", "output_dir"]
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"Experiment '{experiment_name}' missing: {missing}")


def build_train_command(python: str, train_script: str, job: Dict[str, Any]) -> List[str]:
    command = [python, train_script, "train"]
    for key, value in job.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")

        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue

        if isinstance(value, list):
            if key not in REPEATABLE_ARGS:
                raise ValueError(
                    f"List value is only supported for repeatable args: {REPEATABLE_ARGS}. "
                    f"Got '{key}'."
                )
            for item in value:
                command.extend([flag, str(item)])
            continue

        command.extend([flag, str(value)])
    return command


def make_result(
    index: int, job: Dict[str, Any], command: Sequence[str], status: str
) -> Dict[str, Any]:
    result = {
        "index": index,
        "status": status,
        "return_code": "",
        "run_name": job["run_name"],
        "output_dir": job["output_dir"],
        "command": " ".join(shell_quote(part) for part in command),
    }
    for key, value in sorted(job.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[f"arg_{key}"] = value
        else:
            result[f"arg_{key}"] = json.dumps(value)
    return result


def read_metrics(metrics_path: Path) -> Dict[str, Any]:
    with open(metrics_path) as handle:
        metrics = json.load(handle)
    return {f"metric_{key}": value for key, value in sorted(metrics.items())}


def write_summaries(
    results: Sequence[Dict[str, Any]], csv_path: Path, json_path: Path
) -> None:
    with open(json_path, "w") as handle:
        json.dump(list(results), handle, indent=2, sort_keys=True)

    fieldnames = sorted({key for result in results for key in result.keys()})
    preferred = [
        "index",
        "status",
        "return_code",
        "run_name",
        "metric_auprc",
        "metric_auroc",
        "metric_f1",
        "metric_recall",
        "metric_precision",
        "metric_balanced_accuracy",
        "metric_loss",
        "arg_model",
        "arg_learning_rate",
        "arg_batch_size",
        "arg_image_size",
        "arg_balance_strategy",
        "output_dir",
        "command",
    ]
    ordered = [key for key in preferred if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(results)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-=/:.,")
    if all(char in safe_chars for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
