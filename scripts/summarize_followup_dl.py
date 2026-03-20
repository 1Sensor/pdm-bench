from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

MODEL_LABELS = {
    "mlp": "MLP",
    "cnn1d": "1D CNN",
    "fft": "1D FFT",
    "stft": "2D STFT CNN",
}

SCENARIO_ORDER = [
    "cwru_cross_load",
    "cwru_cross_fs",
    "cwru_cross_fault_instance",
    "pu_cross_operating_condition",
    "pu_cross_damage_provenance",
    "pu_cross_bearing_instance",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _pretty_scenario(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def _pretty_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _tier_from_study(study: str) -> str:
    lower = study.lower()
    if "_quick_" in lower:
        return "Quick"
    if "_normal_" in lower:
        return "Normal"
    if "_long_" in lower:
        return "Long"
    return study


def _parse_train_seconds(log_path: Path, model: str) -> float | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text().splitlines():
        match = DL_TRAIN_SECONDS_RE.search(line)
        if match and match.group("model") == model:
            return float(match.group("seconds"))
    return None


def collect_followup_results(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for cfg_path in sorted(input_dir.rglob("pipeline_config.json")):
        if "mlruns" in cfg_path.parts:
            continue

        run_dir = cfg_path.parent
        rel = run_dir.relative_to(input_dir)
        if len(rel.parts) < 4:
            continue

        study = rel.parts[0]
        cfg = _read_json(cfg_path)
        models = cfg.get("models") or []
        if len(models) != 1:
            continue
        model = str(models[0])

        summary_path = run_dir / "results" / f"{model}_test_summary.json"
        if not summary_path.exists():
            continue

        summary = _read_json(summary_path)
        metrics = summary.get("metrics") or {}
        scenario = str(((cfg.get("run") or {}).get("name")) or "")
        dataset = str(((cfg.get("dataset") or {}).get("loader")) or "").upper()
        train_cfg = cfg.get("train") or {}
        windowing_cfg = cfg.get("windowing") or {}

        rows.append(
            {
                "study": study,
                "tier": _tier_from_study(study),
                "dataset": dataset,
                "scenario": scenario,
                "scenario_label": _pretty_scenario(scenario),
                "model": model,
                "model_label": _pretty_model(model),
                "seed": train_cfg.get("random_state"),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "accuracy": metrics.get("accuracy"),
                "train_seconds": _parse_train_seconds(run_dir / "training_log.txt", model),
                "epochs": train_cfg.get("epochs"),
                "train_overlap": windowing_cfg.get("train_overlap"),
                "test_overlap": windowing_cfg.get("test_overlap"),
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_columns = [
        "seed",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "train_seconds",
        "epochs",
        "train_overlap",
        "test_overlap",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    scenario_rank = {scenario: idx for idx, scenario in enumerate(SCENARIO_ORDER)}
    tier_rank = {"Quick": 0, "Normal": 1, "Long": 2}
    df["scenario_rank"] = df["scenario"].map(lambda value: scenario_rank.get(value, 999))
    df["tier_rank"] = df["tier"].map(lambda value: tier_rank.get(value, 999))
    return (
        df.sort_values(by=["scenario_rank", "tier_rank", "study", "seed"])
        .drop(columns=["scenario_rank", "tier_rank"])
        .reset_index(drop=True)
    )


def aggregate_followup(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["study", "tier", "dataset", "scenario", "scenario_label", "model", "model_label"],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(s.tolist()))),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            train_seconds_mean=("train_seconds", "mean"),
            train_seconds_std=("train_seconds", "std"),
            epochs=("epochs", "first"),
            train_overlap=("train_overlap", "first"),
            test_overlap=("test_overlap", "first"),
        )
        .copy()
    )

    merged = grouped.merge(
        reference_df,
        on=["tier", "scenario", "model"],
        how="left",
        validate="one_to_one",
    )
    merged["delta_vs_ref_macro_f1"] = merged["macro_f1_mean"] - merged["ref_macro_f1"]
    merged["delta_vs_ref_balanced_accuracy"] = (
        merged["balanced_accuracy_mean"] - merged["ref_balanced_accuracy"]
    )
    merged["delta_vs_ref_accuracy"] = merged["accuracy_mean"] - merged["ref_accuracy"]
    merged["runtime_ratio_vs_ref"] = (
        merged["train_seconds_mean"] / merged["ref_train_seconds"]
    )

    scenario_rank = {scenario: idx for idx, scenario in enumerate(SCENARIO_ORDER)}
    tier_rank = {"Quick": 0, "Normal": 1, "Long": 2}
    merged["scenario_rank"] = merged["scenario"].map(
        lambda value: scenario_rank.get(value, 999)
    )
    merged["tier_rank"] = merged["tier"].map(lambda value: tier_rank.get(value, 999))
    return (
        merged.sort_values(by=["scenario_rank", "tier_rank", "study"])
        .drop(columns=["scenario_rank", "tier_rank"])
        .reset_index(drop=True)
    )


def load_reference(reference_csv: Path) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    ref = ref[ref["pipeline"] == "dl"][
        ["tier", "scenario", "model", "macro_f1", "balanced_accuracy", "accuracy", "train_seconds"]
    ].copy()
    return ref.rename(
        columns={
            "macro_f1": "ref_macro_f1",
            "balanced_accuracy": "ref_balanced_accuracy",
            "accuracy": "ref_accuracy",
            "train_seconds": "ref_train_seconds",
        }
    )


def _fmt(value: Any, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{float(value):.{decimals}f}"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_followup_latex(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = df[
        [
            "scenario_label",
            "model_label",
            "tier",
            "seeds",
            "macro_f1_mean",
            "macro_f1_std",
            "ref_macro_f1",
            "delta_vs_ref_macro_f1",
        ]
    ].copy()
    table = table.rename(
        columns={
            "scenario_label": "Scenario",
            "model_label": "Model",
            "tier": "Tier",
            "seeds": "Seeds",
            "macro_f1_mean": "Mean Macro-F1",
            "macro_f1_std": "Std Macro-F1",
            "ref_macro_f1": "Reference Macro-F1",
            "delta_vs_ref_macro_f1": "Delta vs Ref",
        }
    )
    for column in ["Mean Macro-F1", "Std Macro-F1", "Reference Macro-F1", "Delta vs Ref"]:
        table[column] = table[column].map(_fmt)

    latex = table.to_latex(
        index=False,
        escape=False,
        caption="DL follow-up summary across repeated-seed runs. Reference values are the single-run benchmark results from the matching tier/model/scenario configuration.",
        label="tab:dl-followup-summary",
        na_rep="--",
    )
    path.write_text(latex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeated-seed DL follow-up experiments."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Follow-up artifact root, e.g. artifacts/benchmarks/phm_followup_dl",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("artifacts/benchmarks/phm_tier_comparison/combined_latest_results.csv"),
        help="CSV with the benchmark-tier reference results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the follow-up summary package. Defaults to <input_dir>/summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "summary")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = collect_followup_results(input_dir)
    if all_runs.empty:
        raise SystemExit(f"No follow-up runs found under: {input_dir}")

    reference = load_reference(args.reference_csv.resolve())
    aggregate = aggregate_followup(all_runs, reference)

    write_csv(all_runs, output_dir / "all_followup_runs.csv")
    write_csv(aggregate, output_dir / "followup_aggregate.csv")
    write_followup_latex(aggregate, output_dir / "followup_summary.tex")

    manifest = {
        "input_dir": str(input_dir),
        "reference_csv": str(args.reference_csv.resolve()),
        "output_dir": str(output_dir),
        "n_runs": len(all_runs),
        "n_studies": int(all_runs["study"].nunique()),
        "notes": [
            "Each study groups repeated-seed runs for one fixed scenario/model/profile combination.",
            "Reference values come from the matching row in the tier-comparison package.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved follow-up summary package to: {output_dir}")


if __name__ == "__main__":
    main()
