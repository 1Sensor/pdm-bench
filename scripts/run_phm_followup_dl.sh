#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

DRY_RUN=0
TRACKING=0
CHECK_CONFIGS=0
SCOPE="all"
SEEDS_CSV="41,42,43"
OUTPUT_ROOT="artifacts/benchmarks/phm_followup_dl"
EXPERIMENT_NAME="phm-followup-dl"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [--scope all|winners|ablation] [--seeds 41,42,43] [--tracking] [--experiment-name NAME] [--output-root PATH] [--check-configs] [--dry-run]

Runs targeted DL follow-up experiments after the broad benchmark tiers.

Scopes:
  winners   Repeat the current best DL setting per scenario across several seeds.
  ablation  Compare quick vs long STFT settings on PU Cross-Damage Provenance.
  all       Run the union of both scopes without duplicating shared jobs.

Examples:
  $SCRIPT_NAME
  $SCRIPT_NAME --scope winners --tracking --experiment-name phm-followup-dl
  $SCRIPT_NAME --scope ablation --seeds 41,42,43 --dry-run
  $SCRIPT_NAME --check-configs
EOF
}

run_cmd() {
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

append_profile_overrides() {
  local profile="$1"
  local -n out_ref=$2

  case "$profile" in
    quick)
      out_ref+=(
        "train.epochs=10"
        "train.batch_size=64"
        "train.num_workers=0"
      )
      ;;
    normal)
      out_ref+=(
        "train.epochs=50"
        "train.batch_size=64"
        "train.num_workers=0"
      )
      ;;
    long)
      out_ref+=(
        "train.epochs=50"
        "train.batch_size=64"
        "train.num_workers=0"
        "windowing.train_overlap=0.75"
        "windowing.test_overlap=0.5"
      )
      ;;
    *)
      echo "Unsupported follow-up profile: $profile" >&2
      exit 1
      ;;
  esac
}

parse_seeds() {
  local raw="$1"
  IFS=',' read -r -a SEEDS <<<"$raw"
  if [[ "${#SEEDS[@]}" -eq 0 ]]; then
    echo "At least one seed is required." >&2
    exit 1
  fi
  local seed
  for seed in "${SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
      echo "Invalid seed: $seed" >&2
      exit 1
    fi
  done
}

should_include_spec() {
  local kind="$1"
  case "$SCOPE" in
    all) return 0 ;;
    winners) [[ "$kind" == "winners" ]] ;;
    ablation) [[ "$kind" == "ablation" ]] ;;
    *)
      echo "Unsupported scope: $SCOPE" >&2
      exit 1
      ;;
  esac
}

emit_specs() {
  local -A seen=()
  local spec id kind

  while IFS='|' read -r id kind _; do
    [[ -z "$id" ]] && continue
    if should_include_spec "$kind"; then
      seen["$id"]=1
    fi
  done <<'EOF'
cwru_cross_load_long_cnn1d|winners|cwru_cross_load/cnn1d|long|cwru_cross_load_long_cnn1d
cwru_cross_fs_normal_stft|winners|cwru_cross_fs/stft|normal|cwru_cross_fs_normal_stft
cwru_cross_fault_instance_long_mlp|winners|cwru_cross_fault_instance/mlp|long|cwru_cross_fault_instance_long_mlp
pu_cross_operating_condition_normal_cnn1d|winners|pu_cross_operating_condition/cnn1d|normal|pu_cross_operating_condition_normal_cnn1d
pu_cross_damage_provenance_quick_stft|winners|pu_cross_damage_provenance/stft|quick|pu_cross_damage_provenance_quick_stft
pu_cross_bearing_instance_long_cnn1d|winners|pu_cross_bearing_instance/cnn1d|long|pu_cross_bearing_instance_long_cnn1d
pu_cross_damage_provenance_long_stft|ablation|pu_cross_damage_provenance/stft|long|pu_cross_damage_provenance_long_stft
EOF

  while IFS='|' read -r id kind task profile study; do
    [[ -z "$id" ]] && continue
    if [[ -n "${seen[$id]:-}" ]]; then
      printf '%s|%s|%s|%s\n' "$id" "$task" "$profile" "$study"
    fi
  done <<'EOF'
cwru_cross_load_long_cnn1d|winners|cwru_cross_load/cnn1d|long|cwru_cross_load_long_cnn1d
cwru_cross_fs_normal_stft|winners|cwru_cross_fs/stft|normal|cwru_cross_fs_normal_stft
cwru_cross_fault_instance_long_mlp|winners|cwru_cross_fault_instance/mlp|long|cwru_cross_fault_instance_long_mlp
pu_cross_operating_condition_normal_cnn1d|winners|pu_cross_operating_condition/cnn1d|normal|pu_cross_operating_condition_normal_cnn1d
pu_cross_damage_provenance_quick_stft|winners|pu_cross_damage_provenance/stft|quick|pu_cross_damage_provenance_quick_stft
pu_cross_bearing_instance_long_cnn1d|winners|pu_cross_bearing_instance/cnn1d|long|pu_cross_bearing_instance_long_cnn1d
pu_cross_damage_provenance_long_stft|ablation|pu_cross_damage_provenance/stft|long|pu_cross_damage_provenance_long_stft
EOF
}

run_followup() {
  local seed
  local spec_line

  while IFS='|' read -r _id task profile study; do
    [[ -z "$task" ]] && continue
    for seed in "${SEEDS[@]}"; do
      local output_dir="${OUTPUT_ROOT}/${study}"
      local cmd=(
        uv run python -m pdm_tools.main.pipelines.dl.pipeline
      )

      if [[ "$CHECK_CONFIGS" -eq 1 ]]; then
        cmd+=(--cfg job)
      fi

      cmd+=(
        "task=${task}"
        "run.output_dir=${output_dir}"
        "+train.random_state=${seed}"
      )

      append_profile_overrides "$profile" cmd

      if [[ "$TRACKING" -eq 1 ]]; then
        cmd+=("tracking.enabled=true" "tracking.experiment_name=${EXPERIMENT_NAME}")
      fi

      run_cmd "${cmd[@]}"
    done
  done < <(emit_specs)
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    --seeds)
      SEEDS_CSV="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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
    --check-configs)
      CHECK_CONFIGS=1
      shift
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

case "$SCOPE" in
  all|winners|ablation) ;;
  *)
    echo "Unsupported scope: $SCOPE" >&2
    usage >&2
    exit 1
    ;;
esac

parse_seeds "$SEEDS_CSV"
require_workspace
run_followup
