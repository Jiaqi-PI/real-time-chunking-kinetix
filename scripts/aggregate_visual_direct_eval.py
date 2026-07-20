from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


EXPECTED_SHARDS = 16
EXPECTED_PER_SHARD = 16
EXPECTED_TOTAL = EXPECTED_SHARDS * EXPECTED_PER_SHARD
def parse_mode(mode: str) -> tuple[int, int]:
    match = re.fullmatch(r"d(\d+)_h(\d+)", mode)
    if match is None:
        raise RuntimeError(
            f"Invalid mode={mode!r}; expected format d<delay>_h<horizon>"
        )
    return int(match.group(1)), int(match.group(2))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def aggregate_mode(root: Path, mode: str) -> dict:
    mode_root = root / mode
    rows: list[dict[str, str]] = []
    summaries: list[dict] = []

    for shard_id in range(EXPECTED_SHARDS):
        shard_root = mode_root / f"shard_{shard_id:02d}"
        episodes_path = shard_root / "episodes.csv"
        summary_path = shard_root / "summary.json"
        if not episodes_path.is_file():
            raise FileNotFoundError(episodes_path)
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)

        shard_rows = read_csv(episodes_path)
        if len(shard_rows) != EXPECTED_PER_SHARD:
            raise RuntimeError(
                f"Expected {EXPECTED_PER_SHARD} rows in {episodes_path}, "
                f"got {len(shard_rows)}"
            )
        for row in shard_rows:
            if int(row["shard_id"]) != shard_id:
                raise RuntimeError(
                    f"Shard id mismatch in {episodes_path}: {row['shard_id']}"
                )
        summary = json.loads(summary_path.read_text())
        if summary.get("mode") != mode:
            raise RuntimeError(
                f"Mode mismatch in {summary_path}: {summary.get('mode')}"
            )
        if int(summary.get("shard_id", -1)) != shard_id:
            raise RuntimeError(
                f"Shard id mismatch in {summary_path}: "
                f"{summary.get('shard_id')}"
            )
        if int(summary.get("num_evals", -1)) != EXPECTED_PER_SHARD:
            raise RuntimeError(
                f"num_evals mismatch in {summary_path}: "
                f"{summary.get('num_evals')}"
            )
        rows.extend(shard_rows)
        summaries.append(summary)

    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} episodes for mode={mode}, got {len(rows)}"
        )

    delays = {int(summary["inference_delay"]) for summary in summaries}
    horizons = {int(summary["execute_horizon"]) for summary in summaries}
    checkpoints = {summary["checkpoint_dir"] for summary in summaries}
    if len(delays) != 1 or len(horizons) != 1 or len(checkpoints) != 1:
        raise RuntimeError(
            f"Inconsistent shard metadata for mode={mode}: "
            f"delays={delays}, horizons={horizons}, checkpoints={checkpoints}"
        )

    solved = np.asarray(
        [float(row["returned_episode_solved"]) for row in rows],
        dtype=np.float64,
    )
    if not np.isin(solved, [0.0, 1.0]).all():
        raise RuntimeError(
            f"Non-binary success values for mode={mode}: "
            f"{np.unique(solved)}"
        )
    returns = np.asarray(
        [float(row["returned_episode_returns"]) for row in rows],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [float(row["returned_episode_lengths"]) for row in rows],
        dtype=np.float64,
    )

    episodes_out = mode_root / "episodes_256.csv"
    with episodes_out.open("w", newline="") as file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    inference_delay = next(iter(delays))
    execute_horizon = next(iter(horizons))
    checkpoint_dir = next(iter(checkpoints))
    expected_delay, expected_horizon = parse_mode(mode)
    if (inference_delay, execute_horizon) != (
        expected_delay,
        expected_horizon,
    ):
        raise RuntimeError(
            f"Unexpected setting for mode={mode}: "
            f"delay={inference_delay}, horizon={execute_horizon}"
        )
    checkpoint_step = Path(checkpoint_dir).name
    if checkpoint_step == "29999":
        checkpoint_label = "30k"
    elif checkpoint_step == "15000":
        checkpoint_label = "15k"
    else:
        checkpoint_label = f"step_{checkpoint_step}"

    result_row = {
        "returned_episode_lengths": float(lengths.mean()),
        "returned_episode_returns": float(returns.mean()),
        "returned_episode_solved": float(solved.mean()),
        "delay": inference_delay,
        "method": f"pi05_visual_direct_{checkpoint_label}_{mode}",
        "level": "grasp_easy",
        "execute_horizon": execute_horizon,
    }
    with (mode_root / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_row.keys()))
        writer.writeheader()
        writer.writerow(result_row)

    summary = {
        "mode": mode,
        "num_evals": EXPECTED_TOTAL,
        "num_solved": int(np.rint(solved).sum()),
        "success_rate": float(solved.mean()),
        "mean_return": float(returns.mean()),
        "mean_episode_length": float(lengths.mean()),
        "inference_delay": inference_delay,
        "execute_horizon": execute_horizon,
        "checkpoint_dir": checkpoint_dir,
        "num_shards": EXPECTED_SHARDS,
        "episodes_per_shard": EXPECTED_PER_SHARD,
        "success_selection": "first_done_index",
    }
    (mode_root / "summary_256.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    print(
        f"MODE={mode} solved={summary['num_solved']}/{EXPECTED_TOTAL} "
        f"success_rate={summary['success_rate']:.8f}",
        flush=True,
    )
    return result_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["d0_h8", "d2_h4"],
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    combined_rows = [aggregate_mode(root, mode) for mode in args.modes]
    with (root / "combined_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(combined_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"WROTE={root / 'combined_results.csv'}", flush=True)
    print("KINETIX_VISUAL_DIRECT_EVAL_AGGREGATE_OK", flush=True)


if __name__ == "__main__":
    main()
