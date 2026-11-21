"""Turn results/*.jsonl into tables and charts for the report.

  python analyze.py results/*.jsonl --out report
  python analyze.py results/*.jsonl --valid results/canonical.jsonl   # only tasks whose
                                                                       # reference solution passes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from executor import CATEGORY_LABELS

COLORS = ["#2a6fdb", "#e07a1f", "#1b998b", "#c1436d", "#7a5cc7", "#6b8f2a"]
CAT_COLORS = {"syntax": "#c1436d", "api": "#e07a1f", "wrong_result": "#2a6fdb",
              "runtime": "#7a5cc7", "timeout": "#6b6b6b", "no_code": "#b0b0b0"}


def load_runs(paths, valid_from=None) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["first_category"] = r["attempts"][0]["category"]
                r["n_attempts"] = len(r["attempts"])
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["run"] != "canonical"] if valid_from else df
    if valid_from:
        ok = load_runs([valid_from])
        df = df[df["task_id"].isin(set(ok.loc[ok["passed"], "task_id"]))]
    return df.reset_index(drop=True)


def pass_at_k(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative % of tasks solved within k attempts, one row per run."""
    out = {}
    for run, g in df.groupby("run"):
        kmax = int(g["max_attempts"].max())
        out[run] = {k: 100 * ((g["passed_at"].notna()) & (g["passed_at"] <= k)).mean()
                    for k in range(1, kmax + 1)}
    return pd.DataFrame(out).T.rename_axis("run")


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in df.groupby("run"):
        first = (g["passed_at"] == 1).mean() * 100
        final = g["passed"].mean() * 100
        failed_first = g[g["passed_at"] != 1]
        rows.append({
            "run": run, "model": g["model"].iloc[0], "feedback": g["feedback"].iloc[0],
            "tasks": len(g),
            "pass@1 (no repair) %": round(first, 1),
            "with repair %": round(final, 1),
            "gain (pts)": round(final - first, 1),
            "% of first-try failures fixed": round(100 * failed_first["passed"].mean(), 1)
            if len(failed_first) else 0.0,
            "avg attempts": round(g["n_attempts"].mean(), 2),
        })
    return pd.DataFrame(rows).set_index("run")


def fix_rate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """For tasks that failed on attempt 1: how many of each error type got fixed later."""
    failed = df[df["first_category"] != "passed"]
    t = failed.groupby(["run", "first_category"])["passed"].agg(["count", "sum"])
    t.columns = ["failed on try 1", "fixed later"]
    t["fix rate %"] = (100 * t["fixed later"] / t["failed on try 1"]).round(1)
    return t.reset_index().replace({"first_category": CATEGORY_LABELS})


def by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["run", "difficulty"])
    t = pd.DataFrame({"tasks": g.size(),
                      "pass@1 %": (g["passed_at"].apply(lambda s: (s == 1).mean()) * 100).round(1),
                      "with repair %": (g["passed"].mean() * 100).round(1)})
    return t.reset_index()


# ---------------------------------------------------------------- charts
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_pass_at_k(df):
    t = pass_at_k(df)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (run, row) in enumerate(t.iterrows()):
        row = row.dropna()
        ax.plot(row.index, row.values, marker="o", lw=2, color=COLORS[i % len(COLORS)], label=run)
        ax.annotate(f"{row.values[-1]:.0f}%", (row.index[-1], row.values[-1]),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=9)
    ax.set_xlabel("Attempts allowed (1 = no repair)")
    ax.set_ylabel("Tasks solved (%)")
    ax.set_title("Pass rate as the model is allowed to repair its code")
    ax.set_xticks(list(t.columns))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_first_errors(df):
    """What went wrong on the first try, and how much of it got fixed (stacked bars per run)."""
    failed = df[df["first_category"] != "passed"]
    runs = sorted(df["run"].unique())
    cats = [c for c in CAT_COLORS if c in set(failed["first_category"])]
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.6 * len(runs) * max(1, len(cats)) ** 0.5))
    y = range(len(runs))
    left = [0] * len(runs)
    for c in cats:
        fixed = [((failed["run"] == r) & (failed["first_category"] == c) & failed["passed"]).sum() for r in runs]
        unfixed = [((failed["run"] == r) & (failed["first_category"] == c) & ~failed["passed"]).sum() for r in runs]
        ax.barh(y, fixed, left=left, color=CAT_COLORS[c], label=f"{CATEGORY_LABELS[c]} - fixed")
        left = [l + f for l, f in zip(left, fixed)]
        ax.barh(y, unfixed, left=left, color=CAT_COLORS[c], alpha=0.35, hatch="//",
                edgecolor="white", label=f"{CATEGORY_LABELS[c]} - not fixed")
        left = [l + u for l, u in zip(left, unfixed)]
    ax.set_yticks(list(y), runs)
    ax.set_xlabel("Tasks that failed on the first try")
    ax.set_title("First-try errors and how many the repair loop fixed")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    _style(ax)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_by_difficulty(df):
    t = by_difficulty(df)
    order = [d for d in ("basic", "intermediate", "difficult") if d in set(t["difficulty"])] or \
        sorted(t["difficulty"].unique())
    runs = sorted(t["run"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / max(1, len(runs))
    for i, run in enumerate(runs):
        sub = t[t["run"] == run].set_index("difficulty").reindex(order)
        xs = [j + i * w for j in range(len(order))]
        ax.bar(xs, sub["pass@1 %"], w * 0.95, color=COLORS[i % len(COLORS)], alpha=0.45)
        ax.bar(xs, sub["with repair %"] - sub["pass@1 %"], w * 0.95, bottom=sub["pass@1 %"],
               color=COLORS[i % len(COLORS)], label=run)
    ax.set_xticks([j + w * (len(runs) - 1) / 2 for j in range(len(order))], order)
    ax.set_ylabel("Tasks solved (%)")
    ax.set_ylim(0, 100)
    ax.set_title("By difficulty (light = first try, dark = added by repair)")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--out", default="report")
    p.add_argument("--valid", default=None, help="canonical run: keep only tasks it passes")
    a = p.parse_args(argv)

    df = load_runs(a.files, valid_from=a.valid)
    if df.empty:
        raise SystemExit("No results found.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "Summary": summary_table(df),
        "Pass rate within k attempts (%)": pass_at_k(df).round(1),
        "Fix rate by first-try error type": fix_rate_by_category(df),
        "By difficulty": by_difficulty(df),
    }
    def as_md(t):
        try:
            return t.to_markdown()
        except ImportError:  # to_markdown needs the optional 'tabulate' package
            return "```\n" + t.to_string() + "\n```"

    md = ["# Results\n"]
    for title, t in tables.items():
        md += [f"## {title}\n", as_md(t), ""]
        print(f"\n== {title} ==\n{t.to_string()}")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    tables["Summary"].to_csv(out / "summary.csv")

    for name, fn in [("pass_at_k.png", plot_pass_at_k), ("first_try_errors.png", plot_first_errors),
                     ("by_difficulty.png", plot_by_difficulty)]:
        fig = fn(df)
        fig.savefig(out / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
    print(f"\nSaved tables and charts to {out}/")


if __name__ == "__main__":
    main()
