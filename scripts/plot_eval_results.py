import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("rtc_diag_eval_base_64/results.csv")
out_dir = Path("rtc_diag/figures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

print("columns:", df.columns.tolist())
print(df.head())

# 自动识别 solve rate 列
if "returned_episode_solved" in df.columns:
    solved_col = "returned_episode_solved"
elif "solved" in df.columns:
    solved_col = "solved"
else:
    raise ValueError(f"Cannot find solve column. Columns are: {df.columns.tolist()}")

# 1. solve rate vs delay
g = (
    df.groupby(["method", "delay"])[solved_col]
    .mean()
    .reset_index()
)

plt.figure()
for method, sub in g.groupby("method"):
    sub = sub.sort_values("delay")
    plt.plot(sub["delay"], sub[solved_col], marker="o", label=method)

plt.xlabel("Inference delay")
plt.ylabel("Mean solve rate")
plt.title("Solve rate vs inference delay")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "solve_rate_vs_delay_64.png", dpi=200)
plt.close()

# 2. heatmap: delay x execute_horizon
for method, sub in df.groupby("method"):
    pivot = sub.pivot_table(
        index="delay",
        columns="execute_horizon",
        values=solved_col,
        aggfunc="mean",
    )

    plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("execute_horizon")
    plt.ylabel("delay")
    plt.title(f"{method} solve rate")
    plt.colorbar(label="solve rate")
    plt.tight_layout()
    plt.savefig(out_dir / f"heatmap_{method}_64.png", dpi=200)
    plt.close()

# 3. method summary bar
summary = df.groupby("method")[solved_col].mean().reset_index()

plt.figure()
plt.bar(summary["method"], summary[solved_col])
plt.ylabel("Mean solve rate")
plt.title("Overall solve rate by method")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(out_dir / "overall_solve_rate_by_method_64.png", dpi=200)
plt.close()

summary.to_csv(Path("rtc_diag/metrics") / "eval_summary_64.csv", index=False)

print("Saved figures to", out_dir)
print("Saved summary to rtc_diag/metrics/eval_summary_64.csv")
