#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

PIPELINES="all"
DATASETS="all"
DRY_RUN=0
TRACKING=0
SMOKE=0
QUICK=0
NORMAL=0
LONG=0
CHECK_CONFIGS=0
OUTPUT_ROOT="artifacts/benchmarks/phm_benchmark"
EXPERIMENT_NAME="phm-benchmark"

ML_TASKS=(
  "cwru_cross_load"
  "cwru_cross_fs"
  "cwru_cross_fault_instance"
  "pu_cross_operating_condition"
  "pu_cross_damage_provenance"
  "pu_cross_bearing_instance"
)

CWRU_ML_TASKS=(
  "cwru_cross_load"
  "cwru_cross_fs"
  "cwru_cross_fault_instance"
)

PU_ML_TASKS=(
  "pu_cross_operating_condition"
  "pu_cross_damage_provenance"
  "pu_cross_bearing_instance"
)

CWRU_DL_TASKS=(
  "cwru_cross_load/mlp"
  "cwru_cross_load/cnn1d"
  "cwru_cross_load/fft"
  "cwru_cross_load/stft"
  "cwru_cross_fs/mlp"
  "cwru_cross_fs/cnn1d"
  "cwru_cross_fs/fft"
  "cwru_cross_fs/stft"
  "cwru_cross_fault_instance/mlp"
  "cwru_cross_fault_instance/cnn1d"
  "cwru_cross_fault_instance/fft"
  "cwru_cross_fault_instance/stft"
)

PU_DL_TASKS=(
  "pu_cross_operating_condition/mlp"
  "pu_cross_operating_condition/cnn1d"
  "pu_cross_operating_condition/fft"
  "pu_cross_operating_condition/stft"
  "pu_cross_damage_provenance/mlp"
  "pu_cross_damage_provenance/cnn1d"
  "pu_cross_damage_provenance/fft"
  "pu_cross_damage_provenance/stft"
  "pu_cross_bearing_instance/mlp"
  "pu_cross_bearing_instance/cnn1d"
  "pu_cross_bearing_instance/fft"
  "pu_cross_bearing_instance/stft"
)

CWRU_DL_TASKS_SMOKE=(
  "cwru_cross_load/mlp"
)

PU_DL_TASKS_SMOKE=(
  "pu_cross_operating_condition/mlp"
)

CWRU_DL_TASKS_QUICK=(
  "cwru_cross_load/cnn1d"
  "cwru_cross_load/stft"
  "cwru_cross_fs/cnn1d"
  "cwru_cross_fs/stft"
  "cwru_cross_fault_instance/cnn1d"
  "cwru_cross_fault_instance/stft"
)

PU_DL_TASKS_QUICK=(
  "pu_cross_operating_condition/cnn1d"
  "pu_cross_operating_condition/stft"
  "pu_cross_damage_provenance/cnn1d"
  "pu_cross_damage_provenance/stft"
  "pu_cross_bearing_instance/cnn1d"
  "pu_cross_bearing_instance/stft"
)

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [--pipelines all|ml|dl] [--datasets all|cwru|pu] [--tracking] [--experiment-name NAME] [--output-root PATH] [--smoke] [--quick] [--normal] [--long] [--check-configs] [--dry-run]

Runs the PHM benchmark matrix using the checked-in Hydra task presets.

Examples:
  $SCRIPT_NAME
  $SCRIPT_NAME --pipelines ml
  $SCRIPT_NAME --pipelines dl --datasets pu
  $SCRIPT_NAME --tracking
  $SCRIPT_NAME --tracking --experiment-name phm-europe-2026
  $SCRIPT_NAME --output-root artifacts/benchmarks/phm_europe_2026
  $SCRIPT_NAME --smoke
  $SCRIPT_NAME --quick
  $SCRIPT_NAME --normal
  $SCRIPT_NAME --long
  $SCRIPT_NAME --check-configs
  $SCRIPT_NAME --dry-run
EOF
}

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

check_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

require_workspace() {
  local script_dir
  local repo_root

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(dirname "$script_dir")"

  if [[ -z "${WORKSPACE:-}" && -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
  fi

  if [[ -z "${WORKSPACE:-}" ]]; then
    WORKSPACE="$repo_root"
    export WORKSPACE
    echo "WORKSPACE is not set; defaulting to repo root: $WORKSPACE"
  fi
}

check_ml_task() {
  local task="$1"
  shift
  local output_dir="${OUTPUT_ROOT}/ml"
  local cmd=(
    uv run python -m pdm_bench.pipelines.ml.pipeline
    --cfg job
    "task=${task}"
    "run.output_dir=${output_dir}"
  )
  if [[ "$TRACKING" -eq 1 ]]; then
    cmd+=("tracking.enabled=true" "tracking.experiment_name=${EXPERIMENT_NAME}")
  fi
  while [[ $# -gt 0 ]]; do
    cmd+=("$1")
    shift
  done
  check_cmd "${cmd[@]}"
}

run_ml_smoke_task() {
  local task="$1"
  local output_dir="$2"
  local cmd=(
    uv run python -m pdm_bench.pipelines.ml.pipeline
    "task=${task}"
    "run.output_dir=${output_dir}"
    "train.classifier_names=[LogisticRegression]"
    "train.bayes_n_iter=1"
    "train.bayes_cv=2"
  )
  if [[ "$TRACKING" -eq 1 ]]; then
    cmd+=("tracking.enabled=true" "tracking.experiment_name=${EXPERIMENT_NAME}")
  fi
  run_cmd "${cmd[@]}"
}

run_ml_quick_overrides() {
  printf '%s\n' \
    "train.classifier_names=[LogisticRegression,RF,XGBoost]" \
    "train.bayes_n_iter=8" \
    "train.bayes_cv=3" \
    "train.bayes_n_points=2"
}

run_ml_normal_overrides() {
  printf '%s\n' \
    "train.classifier_names=[LogisticRegression,RF,XGBoost]" \
    "train.bayes_n_iter=30" \
    "train.bayes_cv=5" \
    "train.bayes_n_points=2"
}

run_ml_long_overrides() {
  printf '%s\n' \
    "train.classifier_names=[LogisticRegression,RF,XGBoost,KNN]" \
    "train.bayes_n_iter=30" \
    "train.bayes_cv=5" \
    "train.bayes_n_points=2" \
    "windowing.train_overlap=0.75" \
    "windowing.test_overlap=0.5" \
    "features.time_features=[pp,zp,rms,cf,std,kurt,sf,eo,FM4,FM6,FM8,clf,ii,FN4]" \
    "features.freq_features=[mf,fc,rms_f,std_f]"
}

run_ml_quick_sweeper() {
  local sweeper="$1"
  local output_dir="$2"
  shift 2

  mapfile -t quick_overrides < <(run_ml_quick_overrides)
  local cmd=(
    uv run python -m pdm_bench.pipelines.ml.pipeline
    -m "hydra/sweeper=${sweeper}"
    "run.output_dir=${output_dir}"
  )
  cmd+=("${quick_overrides[@]}")
  if [[ "$TRACKING" -eq 1 ]]; then
    cmd+=("tracking.enabled=true" "tracking.experiment_name=${EXPERIMENT_NAME}")
  fi
  while [[ $# -gt 0 ]]; do
    cmd+=("$1")
    shift
  done
  run_cmd "${cmd[@]}"
}

run_ml_profile_sweeper() {
  local sweeper="$1"
  local output_dir="$2"
  local overrides_fn="$3"
  shift 3

  mapfile -t profile_overrides < <("$overrides_fn")
  local cmd=(
    uv run python -m pdm_bench.pipelines.ml.pipeline
    -m "hydra/sweeper=${sweeper}"
    "run.output_dir=${output_dir}"
  )
  cmd+=("${profile_overrides[@]}")
  if [[ "$TRACKING" -eq 1 ]]; then
    cmd+=("tracking.enabled=true" "tracking.experiment_name=${EXPERIMENT_NAME}")
  fi
  while [[ $# -gt 0 ]]; do
    cmd+=("$1")
    shift
  done
  run_cmd "${cmd[@]}"
}

check_ml_tasks() {
  local -n tasks_ref=$1
  shift
  local task
  for task in "${tasks_ref[@]}"; do
    check_ml_task "$task" "$@"
  done
}

run_ml() {
  local output_dir="${OUTPUT_ROOT}/ml"

  if [[ "$CHECK_CONFIGS" -eq 1 ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
      case "$DATASETS" in
        cwru)
          check_ml_task "cwru_cross_load" \
            "train.classifier_names=[LogisticRegression]" \
            "train.bayes_n_iter=1" \
            "train.bayes_cv=2"
          ;;
        pu)
          check_ml_task "pu_cross_operating_condition" \
            "train.classifier_names=[LogisticRegression]" \
            "train.bayes_n_iter=1" \
            "train.bayes_cv=2"
          ;;
        all)
          check_ml_task "cwru_cross_load" \
            "train.classifier_names=[LogisticRegression]" \
            "train.bayes_n_iter=1" \
            "train.bayes_cv=2"
          check_ml_task "pu_cross_operating_condition" \
            "train.classifier_names=[LogisticRegression]" \
            "train.bayes_n_iter=1" \
            "train.bayes_cv=2"
          ;;
      esac
      return
    fi

    if [[ "$QUICK" -eq 1 ]]; then
      mapfile -t quick_overrides < <(run_ml_quick_overrides)
      case "$DATASETS" in
        cwru)
          check_ml_tasks CWRU_ML_TASKS "${quick_overrides[@]}"
          ;;
        pu)
          check_ml_tasks PU_ML_TASKS \
            "${quick_overrides[@]}" \
            "windowing.train_overlap=0.1" \
            "windowing.test_overlap=0.1"
          ;;
        all)
          check_ml_tasks CWRU_ML_TASKS "${quick_overrides[@]}"
          check_ml_tasks PU_ML_TASKS \
            "${quick_overrides[@]}" \
            "windowing.train_overlap=0.1" \
            "windowing.test_overlap=0.1"
          ;;
      esac
      return
    fi

    if [[ "$NORMAL" -eq 1 ]]; then
      mapfile -t normal_overrides < <(run_ml_normal_overrides)
      case "$DATASETS" in
        cwru)
          check_ml_tasks CWRU_ML_TASKS "${normal_overrides[@]}"
          ;;
        pu)
          check_ml_tasks PU_ML_TASKS \
            "${normal_overrides[@]}" \
            "windowing.train_overlap=0.1" \
            "windowing.test_overlap=0.1"
          ;;
        all)
          check_ml_tasks CWRU_ML_TASKS "${normal_overrides[@]}"
          check_ml_tasks PU_ML_TASKS \
            "${normal_overrides[@]}" \
            "windowing.train_overlap=0.1" \
            "windowing.test_overlap=0.1"
          ;;
      esac
      return
    fi

    if [[ "$LONG" -eq 1 ]]; then
      mapfile -t long_overrides < <(run_ml_long_overrides)
      case "$DATASETS" in
        cwru)
          check_ml_tasks CWRU_ML_TASKS "${long_overrides[@]}"
          ;;
        pu)
          check_ml_tasks PU_ML_TASKS "${long_overrides[@]}"
          ;;
        all)
          check_ml_tasks ML_TASKS "${long_overrides[@]}"
          ;;
      esac
      return
    fi

    case "$DATASETS" in
      cwru) check_ml_tasks CWRU_ML_TASKS ;;
      pu) check_ml_tasks PU_ML_TASKS ;;
      all) check_ml_tasks ML_TASKS ;;
    esac
    return
  fi

  if [[ "$SMOKE" -eq 1 ]]; then
    case "$DATASETS" in
      cwru) run_ml_smoke_task "cwru_cross_load" "$output_dir" ;;
      pu) run_ml_smoke_task "pu_cross_operating_condition" "$output_dir" ;;
      all)
        run_ml_smoke_task "cwru_cross_load" "$output_dir"
        run_ml_smoke_task "pu_cross_operating_condition" "$output_dir"
        ;;
      *)
        echo "Unsupported dataset selection for ML: $DATASETS" >&2
        exit 1
        ;;
    esac
    return
  fi

  if [[ "$QUICK" -eq 1 ]]; then
    case "$DATASETS" in
      cwru)
        run_ml_quick_sweeper "ml_cwru_benchmark" "$output_dir"
        ;;
      pu)
        run_ml_quick_sweeper \
          "ml_pu_benchmark" \
          "$output_dir" \
          "windowing.train_overlap=0.1" \
          "windowing.test_overlap=0.1"
        ;;
      all)
        run_ml_quick_sweeper "ml_cwru_benchmark" "$output_dir"
        run_ml_quick_sweeper \
          "ml_pu_benchmark" \
          "$output_dir" \
          "windowing.train_overlap=0.1" \
          "windowing.test_overlap=0.1"
        ;;
      *)
        echo "Unsupported dataset selection for ML: $DATASETS" >&2
        exit 1
        ;;
    esac
    return
  fi

  if [[ "$NORMAL" -eq 1 ]]; then
    case "$DATASETS" in
      cwru)
        run_ml_profile_sweeper "ml_cwru_benchmark" "$output_dir" run_ml_normal_overrides
        ;;
      pu)
        run_ml_profile_sweeper \
          "ml_pu_benchmark" \
          "$output_dir" \
          run_ml_normal_overrides \
          "windowing.train_overlap=0.1" \
          "windowing.test_overlap=0.1"
        ;;
      all)
        run_ml_profile_sweeper "ml_cwru_benchmark" "$output_dir" run_ml_normal_overrides
        run_ml_profile_sweeper \
          "ml_pu_benchmark" \
          "$output_dir" \
          run_ml_normal_overrides \
          "windowing.train_overlap=0.1" \
          "windowing.test_overlap=0.1"
        ;;
      *)
        echo "Unsupported dataset selection for ML: $DATASETS" >&2
        exit 1
        ;;
    esac
    return
  fi

  if [[ "$LONG" -eq 1 ]]; then
    case "$DATASETS" in
      cwru)
        run_ml_profile_sweeper "ml_cwru_benchmark" "$output_dir" run_ml_long_overrides
        ;;
      pu)
        run_ml_profile_sweeper "ml_pu_benchmark" "$output_dir" run_ml_long_overrides
        ;;
      all)
        run_ml_profile_sweeper "ml_benchmark" "$output_dir" run_ml_long_overrides
        ;;
      *)
        echo "Unsupported dataset selection for ML: $DATASETS" >&2
        exit 1
        ;;
    esac
    return
  fi

  local sweeper
  case "$DATASETS" in
    cwru) sweeper="ml_cwru_benchmark" ;;
    pu) sweeper="ml_pu_benchmark" ;;
    all) sweeper="ml_benchmark" ;;
    *)
      echo "Unsupported dataset selection for ML: $DATASETS" >&2
      exit 1
      ;;
  esac

  local cmd=(
    uv run python -m pdm_bench.pipelines.ml.pipeline
    -m "hydra/sweeper=${sweeper}"
    "run.output_dir=${output_dir}"
  )
  if [[ "$TRACKING" -eq 1 ]]; then
    cmd+=("tracking.enabled=true" "tracking.experiment_name=${EXPERIMENT_NAME}")
  fi

  run_cmd "${cmd[@]}"
}

run_dl_tasks() {
  local -n tasks_ref=$1
  local output_dir="$2"
  local task
  for task in "${tasks_ref[@]}"; do
    local cmd=(
      uv run python -m pdm_bench.pipelines.dl.pipeline
      "task=${task}"
      "run.output_dir=${output_dir}"
    )
    if [[ "$SMOKE" -eq 1 ]]; then
      cmd+=(
        "train.epochs=1"
        "train.batch_size=64"
        "train.num_workers=0"
      )
    elif [[ "$QUICK" -eq 1 ]]; then
      cmd+=(
        "train.epochs=10"
        "train.batch_size=64"
        "train.num_workers=0"
      )
    elif [[ "$NORMAL" -eq 1 ]]; then
      cmd+=(
        "train.epochs=50"
        "train.batch_size=64"
        "train.num_workers=0"
      )
    elif [[ "$LONG" -eq 1 ]]; then
      cmd+=(
        "train.epochs=50"
        "train.batch_size=64"
        "train.num_workers=0"
        "windowing.train_overlap=0.75"
        "windowing.test_overlap=0.5"
      )
    fi
    if [[ "$TRACKING" -eq 1 ]]; then
      cmd+=("tracking.enabled=true" "tracking.experiment_name=${EXPERIMENT_NAME}")
    fi
    run_cmd "${cmd[@]}"
  done
}

check_dl_tasks() {
  local -n tasks_ref=$1
  shift
  local extras=("$@")
  local output_dir="${OUTPUT_ROOT}/dl"
  local task
  for task in "${tasks_ref[@]}"; do
    local cmd=(
      uv run python -m pdm_bench.pipelines.dl.pipeline
      --cfg job
      "task=${task}"
      "run.output_dir=${output_dir}"
    )
    if [[ "$TRACKING" -eq 1 ]]; then
      cmd+=("tracking.enabled=true" "tracking.experiment_name=${EXPERIMENT_NAME}")
    fi
    cmd+=("${extras[@]}")
    check_cmd "${cmd[@]}"
  done
}

run_dl() {
  local output_dir="${OUTPUT_ROOT}/dl"

  if [[ "$CHECK_CONFIGS" -eq 1 ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
      case "$DATASETS" in
        cwru) check_dl_tasks CWRU_DL_TASKS_SMOKE "train.epochs=1" "train.batch_size=64" "train.num_workers=0" ;;
        pu) check_dl_tasks PU_DL_TASKS_SMOKE "train.epochs=1" "train.batch_size=64" "train.num_workers=0" ;;
        all)
          check_dl_tasks CWRU_DL_TASKS_SMOKE "train.epochs=1" "train.batch_size=64" "train.num_workers=0"
          check_dl_tasks PU_DL_TASKS_SMOKE "train.epochs=1" "train.batch_size=64" "train.num_workers=0"
          ;;
      esac
      return
    fi

    if [[ "$QUICK" -eq 1 ]]; then
      case "$DATASETS" in
        cwru) check_dl_tasks CWRU_DL_TASKS_QUICK "train.epochs=10" "train.batch_size=64" "train.num_workers=0" ;;
        pu) check_dl_tasks PU_DL_TASKS_QUICK "train.epochs=10" "train.batch_size=64" "train.num_workers=0" ;;
        all)
          check_dl_tasks CWRU_DL_TASKS_QUICK "train.epochs=10" "train.batch_size=64" "train.num_workers=0"
          check_dl_tasks PU_DL_TASKS_QUICK "train.epochs=10" "train.batch_size=64" "train.num_workers=0"
          ;;
      esac
      return
    fi

    if [[ "$NORMAL" -eq 1 ]]; then
      case "$DATASETS" in
        cwru) check_dl_tasks CWRU_DL_TASKS_QUICK "train.epochs=50" "train.batch_size=64" "train.num_workers=0" ;;
        pu) check_dl_tasks PU_DL_TASKS_QUICK "train.epochs=50" "train.batch_size=64" "train.num_workers=0" ;;
        all)
          check_dl_tasks CWRU_DL_TASKS_QUICK "train.epochs=50" "train.batch_size=64" "train.num_workers=0"
          check_dl_tasks PU_DL_TASKS_QUICK "train.epochs=50" "train.batch_size=64" "train.num_workers=0"
          ;;
      esac
      return
    fi

    if [[ "$LONG" -eq 1 ]]; then
      case "$DATASETS" in
        cwru) check_dl_tasks CWRU_DL_TASKS "train.epochs=50" "train.batch_size=64" "train.num_workers=0" "windowing.train_overlap=0.75" "windowing.test_overlap=0.5" ;;
        pu) check_dl_tasks PU_DL_TASKS "train.epochs=50" "train.batch_size=64" "train.num_workers=0" "windowing.train_overlap=0.75" "windowing.test_overlap=0.5" ;;
        all)
          check_dl_tasks CWRU_DL_TASKS "train.epochs=50" "train.batch_size=64" "train.num_workers=0" "windowing.train_overlap=0.75" "windowing.test_overlap=0.5"
          check_dl_tasks PU_DL_TASKS "train.epochs=50" "train.batch_size=64" "train.num_workers=0" "windowing.train_overlap=0.75" "windowing.test_overlap=0.5"
          ;;
      esac
      return
    fi

    case "$DATASETS" in
      cwru) check_dl_tasks CWRU_DL_TASKS ;;
      pu) check_dl_tasks PU_DL_TASKS ;;
      all)
        check_dl_tasks CWRU_DL_TASKS
        check_dl_tasks PU_DL_TASKS
        ;;
    esac
    return
  fi

  if [[ "$SMOKE" -eq 1 ]]; then
    case "$DATASETS" in
      cwru) run_dl_tasks CWRU_DL_TASKS_SMOKE "$output_dir" ;;
      pu) run_dl_tasks PU_DL_TASKS_SMOKE "$output_dir" ;;
      all)
        run_dl_tasks CWRU_DL_TASKS_SMOKE "$output_dir"
        run_dl_tasks PU_DL_TASKS_SMOKE "$output_dir"
        ;;
      *)
        echo "Unsupported dataset selection for DL: $DATASETS" >&2
        exit 1
        ;;
    esac
    return
  fi

  if [[ "$QUICK" -eq 1 ]]; then
    case "$DATASETS" in
      cwru) run_dl_tasks CWRU_DL_TASKS_QUICK "$output_dir" ;;
      pu) run_dl_tasks PU_DL_TASKS_QUICK "$output_dir" ;;
      all)
        run_dl_tasks CWRU_DL_TASKS_QUICK "$output_dir"
        run_dl_tasks PU_DL_TASKS_QUICK "$output_dir"
        ;;
      *)
        echo "Unsupported dataset selection for DL: $DATASETS" >&2
        exit 1
        ;;
    esac
    return
  fi

  if [[ "$NORMAL" -eq 1 ]]; then
    case "$DATASETS" in
      cwru) run_dl_tasks CWRU_DL_TASKS_QUICK "$output_dir" ;;
      pu) run_dl_tasks PU_DL_TASKS_QUICK "$output_dir" ;;
      all)
        run_dl_tasks CWRU_DL_TASKS_QUICK "$output_dir"
        run_dl_tasks PU_DL_TASKS_QUICK "$output_dir"
        ;;
      *)
        echo "Unsupported dataset selection for DL: $DATASETS" >&2
        exit 1
        ;;
    esac
    return
  fi

  if [[ "$LONG" -eq 1 ]]; then
    case "$DATASETS" in
      cwru) run_dl_tasks CWRU_DL_TASKS "$output_dir" ;;
      pu) run_dl_tasks PU_DL_TASKS "$output_dir" ;;
      all)
        run_dl_tasks CWRU_DL_TASKS "$output_dir"
        run_dl_tasks PU_DL_TASKS "$output_dir"
        ;;
      *)
        echo "Unsupported dataset selection for DL: $DATASETS" >&2
        exit 1
        ;;
    esac
    return
  fi

  case "$DATASETS" in
    cwru) run_dl_tasks CWRU_DL_TASKS "$output_dir" ;;
    pu) run_dl_tasks PU_DL_TASKS "$output_dir" ;;
    all)
      run_dl_tasks CWRU_DL_TASKS "$output_dir"
      run_dl_tasks PU_DL_TASKS "$output_dir"
      ;;
    *)
      echo "Unsupported dataset selection for DL: $DATASETS" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipelines)
      PIPELINES="${2:-}"
      shift 2
      ;;
    --datasets)
      DATASETS="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --quick)
      QUICK=1
      shift
      ;;
    --normal)
      NORMAL=1
      shift
      ;;
    --long)
      LONG=1
      shift
      ;;
    --check-configs)
      CHECK_CONFIGS=1
      shift
      ;;
    --tracking)
      TRACKING=1
      shift
      ;;
    --experiment-name)
      EXPERIMENT_NAME="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$PIPELINES" in
  all|ml|dl) ;;
  *)
    echo "Unsupported pipeline selection: $PIPELINES" >&2
    usage >&2
    exit 1
    ;;
esac

case "$DATASETS" in
  all|cwru|pu) ;;
  *)
    echo "Unsupported dataset selection: $DATASETS" >&2
    usage >&2
    exit 1
    ;;
esac

profiles_selected=$((SMOKE + QUICK + NORMAL + LONG))
if [[ "$profiles_selected" -gt 1 ]]; then
  echo "--smoke, --quick, --normal, and --long are mutually exclusive." >&2
  usage >&2
  exit 1
fi

require_workspace

if [[ "$PIPELINES" == "all" || "$PIPELINES" == "ml" ]]; then
  run_ml
fi

if [[ "$PIPELINES" == "all" || "$PIPELINES" == "dl" ]]; then
  run_dl
fi
