from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_benchmark import PLOT_RC, SCENARIO_ORDER, collect_results, latest_results


TIER_ORDER = ["Quick", "Normal", "Long"]
PIPELINE_ORDER = ["ml", "dl"]


def _tier_label(path: Path) -> str:
    name = path.name.lower()
    if "quick" in name:
        return "Quick"
    if "normal" in name:
        return "Normal"
    if "long" in name:
        return "Long"
    return path.name


def _numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "train_seconds",
        "window_size",
        "train_overlap",
        "test_overlap",
        "epochs",
        "batch_size",
        "num_workers",
        "bayes_n_iter",
        "bayes_cv",
        "bayes_n_points",
        "n_jobs",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def collect_latest_runs(input_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for input_dir in input_dirs:
        latest = latest_results(collect_results(input_dir))
        if latest.empty:
            continue
        latest = _numeric_columns(latest)
        latest["tier"] = _tier_label(input_dir)
        latest["benchmark_dir"] = str(input_dir.resolve())
        frames.append(latest)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["tier_rank"] = combined["tier"].map(
        lambda value: TIER_ORDER.index(value) if value in TIER_ORDER else len(TIER_ORDER)
    )
    combined["pipeline_rank"] = combined["pipeline"].map(
        lambda value: PIPELINE_ORDER.index(value)
        if value in PIPELINE_ORDER
        else len(PIPELINE_ORDER)
    )
    combined = combined.sort_values(
        by=["pipeline_rank", "scenario_label", "tier_rank", "model_label"]
    ).drop(columns=["tier_rank", "pipeline_rank"])
    return combined.reset_index(drop=True)


def _best_row(part: pd.DataFrame) -> pd.Series:
    ordered = part.sort_values(
        by=["macro_f1", "balanced_accuracy", "accuracy", "train_seconds"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    return ordered.iloc[0]


def best_per_scenario(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (pipeline, tier, scenario), part in df.groupby(
        ["pipeline", "tier", "scenario"], dropna=False
    ):
        best = _best_row(part)
        rows.append(
            {
                "pipeline": pipeline,
                "tier": tier,
                "dataset": best["dataset"],
                "scenario": scenario,
                "scenario_label": best["scenario_label"],
                "model": best["model"],
                "model_label": best["model_label"],
                "macro_f1": best["macro_f1"],
                "balanced_accuracy": best["balanced_accuracy"],
                "accuracy": best["accuracy"],
                "train_seconds": best["train_seconds"],
            }
        )

    best = pd.DataFrame(rows)
    best["tier_rank"] = best["tier"].map(
        lambda value: TIER_ORDER.index(value) if value in TIER_ORDER else len(TIER_ORDER)
    )
    scenario_rank = {scenario: idx for idx, scenario in enumerate(SCENARIO_ORDER)}
    best["scenario_rank"] = best["scenario"].map(
        lambda value: scenario_rank.get(value, len(scenario_rank))
    )
    best = best.sort_values(
        by=["pipeline", "scenario_rank", "tier_rank"]
    ).drop(columns=["tier_rank", "scenario_rank"])
    return best.reset_index(drop=True)


def tier_average_summary(best_df: pd.DataFrame) -> pd.DataFrame:
    if best_df.empty:
        return pd.DataFrame()

    grouped = (
        best_df.groupby(["pipeline", "tier"], as_index=False)
        .agg(
            scenarios=("scenario", "nunique"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_accuracy=("accuracy", "mean"),
            total_train_seconds=("train_seconds", "sum"),
            mean_train_seconds=("train_seconds", "mean"),
        )
        .copy()
    )
    grouped["tier_rank"] = grouped["tier"].map(
        lambda value: TIER_ORDER.index(value) if value in TIER_ORDER else len(TIER_ORDER)
    )
    grouped = grouped.sort_values(by=["pipeline", "tier_rank"]).drop(columns=["tier_rank"])
    return grouped.reset_index(drop=True)


def build_tier_table(best_df: pd.DataFrame, pipeline: str) -> pd.DataFrame:
    part = best_df[best_df["pipeline"] == pipeline].copy()
    if part.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        scenario_part = part[part["scenario"] == scenario]
        if scenario_part.empty:
            continue

        first = scenario_part.iloc[0]
        row: dict[str, Any] = {
            "Dataset": first["dataset"],
            "Scenario": first["scenario_label"],
        }
        macro_by_tier: dict[str, float] = {}
        for tier in TIER_ORDER:
            tier_part = scenario_part[scenario_part["tier"] == tier]
            if tier_part.empty:
                row[f"{tier}"] = "--"
                row[f"{tier} Macro-F1"] = np.nan
                row[f"{tier} Train [s]"] = np.nan
                continue

            best = tier_part.iloc[0]
            macro_by_tier[tier] = float(best["macro_f1"])
            row[f"{tier}"] = f"{best['model_label']} / {best['macro_f1']:.3f}"
            row[f"{tier} Macro-F1"] = best["macro_f1"]
            row[f"{tier} Train [s]"] = best["train_seconds"]

        quick_f1 = row.get("Quick Macro-F1")
        normal_f1 = row.get("Normal Macro-F1")
        long_f1 = row.get("Long Macro-F1")
        row["Delta Q->N"] = (
            normal_f1 - quick_f1
            if pd.notna(normal_f1) and pd.notna(quick_f1)
            else np.nan
        )
        row["Delta N->L"] = (
            long_f1 - normal_f1
            if pd.notna(long_f1) and pd.notna(normal_f1)
            else np.nan
        )
        row["Delta Q->L"] = (
            long_f1 - quick_f1
            if pd.notna(long_f1) and pd.notna(quick_f1)
            else np.nan
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _format_metric(value: Any, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_tier_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = df.copy()

    for idx, row in table.iterrows():
        numeric_values = [
            row[column]
            for column in ("Quick Macro-F1", "Normal Macro-F1", "Long Macro-F1")
            if pd.notna(row[column])
        ]
        row_max = max(numeric_values) if numeric_values else None
        for tier in TIER_ORDER:
            metric_column = f"{tier} Macro-F1"
            if row[tier] == "--" or row_max is None:
                continue
            if np.isclose(float(row[metric_column]), float(row_max)):
                table.loc[idx, tier] = f"\\textbf{{{row[tier]}}}"

    for column in ["Quick Macro-F1", "Normal Macro-F1", "Long Macro-F1"]:
        if column in table.columns:
            table = table.drop(columns=[column])

    for column in ["Delta Q->N", "Delta N->L", "Delta Q->L"]:
        if column in table.columns:
            table[column] = table[column].map(_format_metric)

    for column in ["Quick Train [s]", "Normal Train [s]", "Long Train [s]"]:
        if column in table.columns:
            table[column] = table[column].map(lambda value: _format_metric(value, 1))

    latex = table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        na_rep="--",
    )
    path.write_text(latex)


def write_average_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = df.copy().rename(
        columns={
            "pipeline": "Pipeline",
            "tier": "Tier",
            "scenarios": "Scenarios",
            "mean_macro_f1": "Mean Macro-F1",
            "mean_balanced_accuracy": "Mean Bal. Acc.",
            "mean_accuracy": "Mean Acc.",
            "total_train_seconds": "Total Train [s]",
            "mean_train_seconds": "Mean Train [s]",
        }
    )
    table["Pipeline"] = table["Pipeline"].str.upper()
    for column in [
        "Mean Macro-F1",
        "Mean Bal. Acc.",
        "Mean Acc.",
        "Total Train [s]",
        "Mean Train [s]",
    ]:
        decimals = 1 if "[s]" in column else 3
        table[column] = table[column].map(lambda value: _format_metric(value, decimals))
    table["Scenarios"] = table["Scenarios"].map(
        lambda value: "--" if pd.isna(value) else f"{int(value)}"
    )

    latex = table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        na_rep="--",
    )
    path.write_text(latex)


def plot_tier_progression(best_df: pd.DataFrame, metric: str, output_base: Path, metric_label: str) -> None:
    if best_df.empty:
        return

    output_base.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.9, 6.2), constrained_layout=True)
        for ax, pipeline, title in zip(
            axes,
            PIPELINE_ORDER,
            ["ML Best-by-Scenario", "DL Best-by-Scenario"],
            strict=True,
        ):
            part = best_df[best_df["pipeline"] == pipeline].copy()
            if part.empty:
                ax.set_axis_off()
                continue

            tier_positions = np.arange(len(TIER_ORDER), dtype=float)
            for scenario in SCENARIO_ORDER:
                scenario_part = part[part["scenario"] == scenario].copy()
                if scenario_part.empty:
                    continue
                scenario_part["tier_rank"] = scenario_part["tier"].map(TIER_ORDER.index)
                scenario_part = scenario_part.sort_values(by="tier_rank")
                x = scenario_part["tier_rank"].to_numpy(dtype=float)
                y = scenario_part[metric].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.4,
                    label=scenario_part.iloc[0]["scenario_label"],
                )

            ax.set_title(title)
            ax.set_xticks(tier_positions)
            ax.set_xticklabels(TIER_ORDER)
            ax.set_ylabel(metric_label)
            if metric == "train_seconds":
                ax.set_yscale("log")
            else:
                ax.set_ylim(0.0, 1.05)
            ax.legend(loc="best", frameon=False, ncol=2)

        for ext in ("png", "pdf"):
            fig.savefig(output_base.with_suffix(f".{ext}"))
        plt.close(fig)


def write_manifest(output_dir: Path, input_dirs: list[Path], latest_rows: int, best_rows: int) -> None:
    manifest = {
        "input_dirs": [str(path.resolve()) for path in input_dirs],
        "output_dir": str(output_dir.resolve()),
        "tiers": TIER_ORDER,
        "latest_rows": latest_rows,
        "best_rows": best_rows,
        "primary_metric": "macro_f1",
        "support_metrics": ["balanced_accuracy", "accuracy"],
        "notes": [
            "Each input run is reduced to the latest result per (pipeline, scenario, model).",
            "Tier comparison tables use the best model per (pipeline, tier, scenario) based on macro-F1.",
            "Plots are exported at 300 dpi with the same serif font stack as summarize_benchmark.py.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple benchmark runs and build tier-level CSV, LaTeX tables, and plots."
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="Benchmark artifact directories, e.g. artifacts/benchmarks/phm_quick_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the comparison package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dirs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    latest = collect_latest_runs(input_dirs)
    if latest.empty:
        raise SystemExit("No benchmark results found in the provided input directories.")

    best = best_per_scenario(latest)
    averages = tier_average_summary(best)

    write_csv(latest, output_dir / "combined_latest_results.csv")
    write_csv(best, output_dir / "best_by_scenario.csv")
    write_csv(averages, output_dir / "tier_averages.csv")

    for pipeline in PIPELINE_ORDER:
        tier_table = build_tier_table(best, pipeline)
        write_csv(tier_table, tables_dir / f"{pipeline}_tier_best_macro_f1.csv")
        if not tier_table.empty:
            write_tier_latex(
                tier_table,
                tables_dir / f"{pipeline}_tier_best_macro_f1.tex",
                caption=f"Best-model macro-F1 comparison across benchmark tiers for the {pipeline.upper()} pipeline.",
                label=f"tab:{pipeline}-tier-comparison",
            )

    if not averages.empty:
        write_average_latex(
            averages,
            tables_dir / "tier_averages.tex",
            caption="Tier-level averages computed from the best model per scenario.",
            label="tab:tier-averages",
        )

    plot_tier_progression(best, "macro_f1", plots_dir / "macro_f1_progression", "Macro-F1")
    plot_tier_progression(
        best,
        "train_seconds",
        plots_dir / "runtime_progression",
        "Train Time [s]",
    )

    write_manifest(output_dir, input_dirs, len(latest), len(best))
    print(f"Saved comparison package to: {output_dir}")


if __name__ == "__main__":
    main()
