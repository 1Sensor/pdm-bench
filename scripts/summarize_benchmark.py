from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_SUFFIX = "_test_summary.json"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

ML_TRAIN_SECONDS_RE = re.compile(
    r"Model (?P<model>[A-Za-z0-9_]+) finished training in (?P<seconds>[-+0-9.eE]+) seconds"
)
DL_TRAIN_SECONDS_RE = re.compile(
    r"✅\s+(?P<model>[A-Za-z0-9_]+)\s+ready in\s+(?P<seconds>[-+0-9.eE]+)s"
)

SCENARIO_LABELS = {
    "cwru_cross_load": "CWRU Cross-Load",
    "cwru_cross_fs": "CWRU Cross-FS",
    "cwru_cross_fault_instance": "CWRU Cross-Fault Instance",
    "pu_cross_operating_condition": "PU Cross-Operating Condition",
    "pu_cross_damage_provenance": "PU Cross-Damage Provenance",
    "pu_cross_bearing_instance": "PU Cross-Bearing Instance",
}
SCENARIO_ORDER = [
    "cwru_cross_load",
    "cwru_cross_fs",
    "cwru_cross_fault_instance",
    "pu_cross_operating_condition",
    "pu_cross_damage_provenance",
    "pu_cross_bearing_instance",
]

MODEL_LABELS = {
    "LogisticRegression": "LR",
    "RF": "RF",
    "XGBoost": "XGBoost",
    "KNN": "KNN",
    "SVM": "SVM",
    "mlp": "MLP",
    "cnn1d": "1D CNN",
    "fft": "1D FFT",
    "stft": "2D STFT CNN",
}
MODEL_ORDER = {
    "ml": ["LogisticRegression", "RF", "XGBoost", "KNN", "SVM"],
    "dl": ["mlp", "cnn1d", "fft", "stft", "cnn2d"],
}

MODEL_COLORS = {
    "LR": "#22333b",
    "RF": "#7a4f2d",
    "XGBoost": "#3f7d20",
    "KNN": "#6c757d",
    "SVM": "#5a189a",
    "MLP": "#6d597a",
    "1D CNN": "#005f73",
    "1D FFT": "#ca6702",
    "2D STFT CNN": "#ae2012",
}

FONT_STACK = [
    "TeX Gyre Termes",
    "Nimbus Roman",
    "STIX Two Text",
    "Times New Roman",
    "DejaVu Serif",
]

PLOT_RC = {
    "font.family": "serif",
    "font.serif": FONT_STACK,
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_timestamp(timestamp: str | None) -> pd.Timestamp:
    if not timestamp:
        return pd.NaT
    return pd.to_datetime(timestamp, format=TIMESTAMP_FORMAT, errors="coerce")


def _pretty_scenario(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())


def _pretty_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _infer_input_representation(pipeline: str, model: str, cfg: dict[str, Any]) -> str:
    if pipeline == "ml":
        return str(((cfg.get("features") or {}).get("mode")) or "")
    if model == "stft":
        return "stft"
    if model == "cnn1d":
        return "raw"
    return model


def _infer_pipeline(cfg: dict[str, Any], run_dir: Path) -> str:
    train_cfg = cfg.get("train") or {}
    if "classifier_names" in train_cfg:
        return "ml"
    if "models" in cfg:
        return "dl"
    if "ml" in run_dir.parts:
        return "ml"
    return "dl"


def _parse_train_seconds(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}

    train_seconds: dict[str, float] = {}
    for line in log_path.read_text().splitlines():
        ml_match = ML_TRAIN_SECONDS_RE.search(line)
        if ml_match:
            train_seconds[ml_match.group("model")] = float(ml_match.group("seconds"))
            continue

        dl_match = DL_TRAIN_SECONDS_RE.search(line)
        if dl_match:
            train_seconds[dl_match.group("model")] = float(dl_match.group("seconds"))

    return train_seconds


def _n_samples_from_summary(summary: dict[str, Any]) -> int | None:
    confusion = summary.get("confusion_matrix") or []
    if confusion:
        return int(sum(sum(int(cell) for cell in row) for row in confusion))

    per_class = summary.get("per_class_metrics") or {}
    if not per_class:
        return None

    total = 0
    for metrics in per_class.values():
        support = metrics.get("support")
        if support is None:
            return None
        total += int(support)
    return total


def _sort_key(values: pd.Series, ordered: list[str]) -> pd.Series:
    mapping = {value: idx for idx, value in enumerate(ordered)}
    return values.map(lambda x: mapping.get(x, len(mapping)))


def collect_results(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for cfg_path in sorted(input_dir.rglob("pipeline_config.json")):
        if "mlruns" in cfg_path.parts:
            continue

        run_dir = cfg_path.parent
        results_dir = run_dir / "results"
        if not results_dir.exists():
            continue

        summary_paths = sorted(results_dir.glob(f"*{SUMMARY_SUFFIX}"))
        if not summary_paths:
            continue

        cfg = _read_json(cfg_path)
        run_meta_path = run_dir / "run_meta.json"
        run_meta = _read_json(run_meta_path) if run_meta_path.exists() else {}
        pipeline = _infer_pipeline(cfg, run_dir)
        train_seconds_map = _parse_train_seconds(run_dir / "training_log.txt")
        scenario = str(((cfg.get("run") or {}).get("name")) or run_dir.name)
        dataset = str(((cfg.get("dataset") or {}).get("loader")) or "").upper()
        timestamp = run_meta.get("timestamp")

        for summary_path in summary_paths:
            model = summary_path.name[: -len(SUMMARY_SUFFIX)]
            summary = _read_json(summary_path)
            metrics = summary.get("metrics") or {}
            per_class = summary.get("per_class_metrics") or {}

            row = {
                "benchmark": input_dir.name,
                "pipeline": pipeline,
                "dataset": dataset,
                "scenario": scenario,
                "scenario_label": _pretty_scenario(scenario),
                "model": model,
                "model_label": _pretty_model(model),
                "input_representation": _infer_input_representation(pipeline, model, cfg),
                "task_type": summary.get("task_type"),
                "split": summary.get("split"),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "train_seconds": train_seconds_map.get(model),
                "n_classes": len(per_class) if per_class else None,
                "n_samples": _n_samples_from_summary(summary),
                "timestamp": timestamp,
                "timestamp_dt": _parse_timestamp(timestamp),
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "window_size": ((cfg.get("windowing") or {}).get("size")),
                "train_overlap": ((cfg.get("windowing") or {}).get("train_overlap")),
                "test_overlap": ((cfg.get("windowing") or {}).get("test_overlap")),
                "feature_mode": ((cfg.get("features") or {}).get("mode")),
                "epochs": ((cfg.get("train") or {}).get("epochs")),
                "batch_size": ((cfg.get("train") or {}).get("batch_size")),
                "num_workers": ((cfg.get("train") or {}).get("num_workers")),
                "device": ((cfg.get("train") or {}).get("device")),
                "bayes_n_iter": ((cfg.get("train") or {}).get("bayes_n_iter")),
                "bayes_cv": ((cfg.get("train") or {}).get("bayes_cv")),
                "bayes_n_points": ((cfg.get("train") or {}).get("bayes_n_points")),
                "n_jobs": ((cfg.get("train") or {}).get("n_jobs")),
                "train_query": ((cfg.get("dataset") or {}).get("train_query")),
                "test_query": ((cfg.get("dataset") or {}).get("test_query")),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["pipeline", "scenario", "model", "timestamp_dt", "run_dir"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    return df


def latest_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    latest = (
        df.sort_values(
            by=["timestamp_dt", "run_dir"],
            ascending=[True, True],
        )
        .groupby(["pipeline", "scenario", "model"], as_index=False, dropna=False)
        .tail(1)
        .copy()
    )
    return sort_results(latest)


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["scenario_order"] = _sort_key(out["scenario"], SCENARIO_ORDER)
    out["model_order"] = out.apply(
        lambda row: _sort_key(
            pd.Series([row["model"]]),
            MODEL_ORDER.get(str(row["pipeline"]), []),
        ).iloc[0],
        axis=1,
    )
    out = out.sort_values(
        by=["pipeline", "scenario_order", "model_order", "model_label"]
    ).drop(columns=["scenario_order", "model_order"])
    return out.reset_index(drop=True)


def build_macro_wide_table(df: pd.DataFrame, pipeline: str) -> pd.DataFrame:
    part = sort_results(df[df["pipeline"] == pipeline])
    if part.empty:
        return pd.DataFrame()

    scenario_order = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario in set(part["scenario"].tolist())
    ]
    model_order = [
        model
        for model in MODEL_ORDER.get(pipeline, [])
        if model in set(part["model"].tolist())
    ]
    if not model_order:
        model_order = sorted(part["model"].unique().tolist())

    pivot = (
        part.pivot(index="scenario", columns="model", values="macro_f1")
        .reindex(index=scenario_order, columns=model_order)
        .rename(index=SCENARIO_LABELS, columns=MODEL_LABELS)
    )
    pivot.index.name = "Scenario"
    return pivot.reset_index()


def build_support_table(df: pd.DataFrame, pipeline: str) -> pd.DataFrame:
    part = sort_results(df[df["pipeline"] == pipeline])
    if part.empty:
        return pd.DataFrame()

    table = part[
        [
            "dataset",
            "scenario_label",
            "model_label",
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "train_seconds",
        ]
    ].copy()
    return table.rename(
        columns={
            "dataset": "Dataset",
            "scenario_label": "Scenario",
            "model_label": "Model",
            "macro_f1": "Macro-F1",
            "balanced_accuracy": "Balanced Acc.",
            "accuracy": "Accuracy",
            "train_seconds": "Train Time [s]",
        }
    )


def build_setup_inventory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    inventory = sort_results(df)[
        [
            "pipeline",
            "dataset",
            "scenario_label",
            "model_label",
            "input_representation",
            "window_size",
            "train_overlap",
            "test_overlap",
            "feature_mode",
            "epochs",
            "batch_size",
            "bayes_n_iter",
            "bayes_cv",
            "bayes_n_points",
            "n_jobs",
            "device",
        ]
    ].copy()
    inventory = inventory.rename(
        columns={
            "pipeline": "Pipeline",
            "dataset": "Dataset",
            "scenario_label": "Scenario",
            "model_label": "Model",
            "input_representation": "Input",
            "window_size": "Window Size",
            "train_overlap": "Train Overlap",
            "test_overlap": "Test Overlap",
            "feature_mode": "Feature Mode",
            "epochs": "Epochs",
            "batch_size": "Batch Size",
            "bayes_n_iter": "Bayes Iter",
            "bayes_cv": "Bayes CV",
            "bayes_n_points": "Bayes n_points",
            "n_jobs": "n_jobs",
            "device": "Device",
        }
    )
    return inventory.drop_duplicates().reset_index(drop=True)


def _format_number(value: Any, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{float(value):.{decimals}f}"


def _format_seconds(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{float(value):.2f}"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    bold_row_max: bool = False,
    seconds_columns: set[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seconds_columns = seconds_columns or set()

    table = df.copy()
    value_columns = [col for col in table.columns if col != "Scenario"]
    if bold_row_max and value_columns:
        numeric = table[value_columns].apply(pd.to_numeric, errors="coerce")
        row_max = numeric.max(axis=1, skipna=True)
        for col in value_columns:
            formatted: list[str] = []
            for idx, value in numeric[col].items():
                if pd.isna(value):
                    formatted.append("--")
                elif np.isclose(float(value), float(row_max.iloc[idx]), equal_nan=False):
                    formatted.append(f"\\textbf{{{_format_number(value)}}}")
                else:
                    formatted.append(_format_number(value))
            table[col] = formatted
    else:
        for col in table.columns:
            if col in {"Scenario", "Dataset", "Model", "Pipeline", "Input", "Device", "Feature Mode"}:
                continue
            decimals = 2 if col in seconds_columns or "[s]" in col else 3
            table[col] = table[col].map(lambda x: _format_number(x, decimals=decimals))

    latex = table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        na_rep="--",
    )
    path.write_text(latex)


def write_inventory_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = df.copy()
    for col in table.columns:
        if col in {"Pipeline", "Dataset", "Scenario", "Model", "Input", "Device", "Feature Mode"}:
            continue
        if col in {"Window Size", "Epochs", "Batch Size", "Bayes Iter", "Bayes CV", "Bayes n_points", "n_jobs"}:
            table[col] = table[col].map(
                lambda x: "--" if pd.isna(x) else f"{int(float(x))}"
            )
        else:
            table[col] = table[col].map(lambda x: _format_number(x, decimals=2))

    latex = table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        na_rep="--",
    )
    path.write_text(latex)


def _pipeline_model_order(df: pd.DataFrame, pipeline: str) -> list[tuple[str, str]]:
    part = df[df["pipeline"] == pipeline]
    present = set(part["model"].tolist())
    ordered = [
        (model, _pretty_model(model))
        for model in MODEL_ORDER.get(pipeline, [])
        if model in present
    ]
    remaining = sorted(present - {model for model, _ in ordered})
    ordered.extend((model, _pretty_model(model)) for model in remaining)
    return ordered


def _pipeline_scenario_order(df: pd.DataFrame, pipeline: str) -> list[tuple[str, str]]:
    part = df[df["pipeline"] == pipeline]
    present = set(part["scenario"].tolist())
    ordered = [
        (scenario, _pretty_scenario(scenario))
        for scenario in SCENARIO_ORDER
        if scenario in present
    ]
    remaining = sorted(present - {scenario for scenario, _ in ordered})
    ordered.extend((scenario, _pretty_scenario(scenario)) for scenario in remaining)
    return ordered


def plot_overview(
    df: pd.DataFrame,
    metric: str,
    output_base: Path,
    metric_label: str,
    log_scale: bool = False,
) -> None:
    if df.empty:
        return

    output_base.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.9, 6.6), constrained_layout=True)
        for ax, pipeline, title in zip(
            axes,
            ["ml", "dl"],
            ["ML Benchmarks", "DL Benchmarks"],
            strict=True,
        ):
            part = sort_results(df[df["pipeline"] == pipeline])
            if part.empty:
                ax.set_axis_off()
                continue

            scenario_pairs = _pipeline_scenario_order(part, pipeline)
            model_pairs = _pipeline_model_order(part, pipeline)
            scenario_order = [scenario for scenario, _ in scenario_pairs]
            scenario_labels = [label for _, label in scenario_pairs]
            model_order = [model for model, _ in model_pairs]
            model_labels = [label for _, label in model_pairs]

            pivot = (
                part.pivot(index="scenario", columns="model", values=metric)
                .reindex(index=scenario_order, columns=model_order)
                .rename(index=SCENARIO_LABELS, columns=MODEL_LABELS)
            )

            y = np.arange(len(scenario_labels), dtype=float)
            total_height = 0.82
            bar_height = total_height / max(1, len(model_labels))
            offsets = (
                np.arange(len(model_labels), dtype=float) - (len(model_labels) - 1) / 2.0
            ) * bar_height

            for offset, model_label in zip(offsets, model_labels, strict=True):
                if model_label not in pivot.columns:
                    continue
                values = pivot[model_label].to_numpy(dtype=float)
                bars = ax.barh(
                    y + offset,
                    values,
                    height=bar_height * 0.92,
                    label=model_label,
                    color=MODEL_COLORS.get(model_label, "#4f5d75"),
                )
                for bar, value in zip(bars, values, strict=True):
                    if np.isnan(value):
                        continue
                    if log_scale:
                        label_x = value * 1.03
                        text = f"{value:.1f}"
                    else:
                        label_x = min(value + 0.01, 1.01)
                        text = f"{value:.3f}"
                    ax.text(
                        label_x,
                        bar.get_y() + bar.get_height() / 2.0,
                        text,
                        va="center",
                        ha="left",
                        fontsize=7,
                    )

            ax.set_title(title)
            ax.set_yticks(y)
            ax.set_yticklabels(scenario_labels)
            ax.invert_yaxis()
            ax.set_xlabel(metric_label)
            if log_scale:
                ax.set_xscale("log")
                ax.set_xlim(left=max(0.1, float(part[metric].min(skipna=True)) * 0.8))
            else:
                ax.set_xlim(0.0, 1.05)
            ax.legend(loc="upper right" if log_scale else "lower right", frameon=False)

        for ext in ("png", "pdf"):
            fig.savefig(output_base.with_suffix(f".{ext}"))
        plt.close(fig)


def write_manifest(output_dir: Path, input_dir: Path, all_rows: int, latest_rows_count: int) -> None:
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "all_rows": all_rows,
        "latest_rows": latest_rows_count,
        "primary_metric": "macro_f1",
        "support_metrics": ["balanced_accuracy", "accuracy"],
        "font_stack": FONT_STACK,
        "notes": [
            "Metrics are read from saved *_test_summary.json files.",
            "Train time is parsed from training_log.txt.",
            "Nested mlruns directories are ignored.",
            "Paper-facing tables and plots use the latest run per (pipeline, scenario, model).",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark artifacts into CSV, LaTeX tables, and paper-style plots."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Benchmark artifact directory, e.g. artifacts/benchmarks/phm_quick_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the summary package. Defaults to <input_dir>/summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "summary")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    all_results = collect_results(input_dir)
    if all_results.empty:
        raise SystemExit(f"No benchmark summaries found under: {input_dir}")

    latest = latest_results(all_results)

    write_csv(all_results, output_dir / "all_results.csv")
    write_csv(latest, output_dir / "latest_results.csv")

    for pipeline in ("ml", "dl"):
        macro_wide = build_macro_wide_table(latest, pipeline)
        support = build_support_table(latest, pipeline)

        write_csv(macro_wide, tables_dir / f"{pipeline}_macro_f1_wide.csv")
        write_csv(support, tables_dir / f"{pipeline}_metrics_long.csv")

        if not macro_wide.empty:
            write_latex_table(
                macro_wide,
                tables_dir / f"{pipeline}_macro_f1_wide.tex",
                caption=f"Macro-F1 by scenario for the {pipeline.upper()} benchmark subset. Best value per scenario is bolded.",
                label=f"tab:{pipeline}-macro-f1",
                bold_row_max=True,
            )

        if not support.empty:
            write_latex_table(
                support,
                tables_dir / f"{pipeline}_metrics_long.tex",
                caption=f"Supporting metrics for the {pipeline.upper()} benchmark subset.",
                label=f"tab:{pipeline}-support-metrics",
                seconds_columns={"Train Time [s]"},
            )

    inventory = build_setup_inventory(latest)
    write_csv(inventory, tables_dir / "setup_inventory.csv")
    write_inventory_latex(
        inventory,
        tables_dir / "setup_inventory.tex",
        caption="Benchmark setup inventory for the latest run of each pipeline/scenario/model combination.",
        label="tab:benchmark-setup-inventory",
    )

    plot_overview(latest, "macro_f1", plots_dir / "macro_f1_overview", "Macro-F1")
    plot_overview(
        latest,
        "train_seconds",
        plots_dir / "runtime_overview",
        "Train Time [s]",
        log_scale=True,
    )

    write_manifest(output_dir, input_dir, len(all_results), len(latest))

    print(f"Saved summary package to: {output_dir}")


if __name__ == "__main__":
    main()
