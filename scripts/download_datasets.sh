#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

SCRIPT_NAME="$(basename "$0")"
DOWNLOAD_CWRU=0
DOWNLOAD_PU=0
INSTALL_KAGGLE=0


usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [--all] [--cwru] [--pu] [--install-kaggle]

Download the benchmark datasets into ./datasets/ so the checked-in Hydra configs
resolve without extra path edits.

Options:
  --all   Download both benchmark datasets (default)
  --cwru  Download only the CWRU dataset from Kaggle
  --pu    Download only the PU dataset from BearingDataCenter
  --install-kaggle  Install the Kaggle CLI automatically (prefers uv, falls back to pip --user)
  -h, --help  Show this help text

Requirements:
  CWRU: kaggle CLI configured locally plus ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
  PU: aria2c plus either unrar or 7z, and wget or curl
EOF
}


parse_args() {
  if [[ $# -eq 0 ]]; then
    DOWNLOAD_CWRU=1
    DOWNLOAD_PU=1
    return
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --all)
        DOWNLOAD_CWRU=1
        DOWNLOAD_PU=1
        ;;
      --cwru)
        DOWNLOAD_CWRU=1
        ;;
      --pu)
        DOWNLOAD_PU=1
        ;;
      --install-kaggle)
        INSTALL_KAGGLE=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage
        exit 1
        ;;
    esac
    shift
  done
}


resolve_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  dirname "$script_dir"
}


load_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}


require_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    return 0
  fi
  echo "Error: required command not found: $name" >&2
  return 1
}


append_user_bin_to_path() {
  local user_base
  user_base="$(python3 - <<'PY'
import site
print(site.USER_BASE)
PY
)"
  export PATH="$user_base/bin:$PATH"
}


install_kaggle_cli() {
  append_user_bin_to_path

  if command -v kaggle >/dev/null 2>&1; then
    return 0
  fi

  if command -v uv >/dev/null 2>&1; then
    echo "Installing kaggle CLI with uv tool install ..."
    uv tool install --force kaggle || return 1
    append_user_bin_to_path
    command -v kaggle >/dev/null 2>&1 && return 0
  fi

  echo "Installing kaggle CLI with pip --user ..."
  if ! python3 -m pip install --user kaggle; then
    cat >&2 <<'EOF'
Error: automatic pip installation failed.

If you're on a system with an externally-managed Python, try:
  uv tool install kaggle

Or install pipx and run:
  pipx install kaggle
EOF
    return 1
  fi

  append_user_bin_to_path
  command -v kaggle >/dev/null 2>&1
}


ensure_kaggle_cli() {
  if command -v kaggle >/dev/null 2>&1; then
    return 0
  fi

  if [[ "$INSTALL_KAGGLE" -eq 0 ]]; then
    cat >&2 <<'EOF'
Error: kaggle CLI not found.

Install it with one of:
  uv tool install kaggle
  python3 -m pip install --user kaggle
  ./scripts/download_datasets.sh --cwru --install-kaggle

Then place your Kaggle API credentials at:
  ~/.kaggle/kaggle.json
EOF
    return 1
  fi

  install_kaggle_cli || return 1

  if ! command -v kaggle >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Error: kaggle CLI was installed, but it is not on PATH for this shell.
Try re-running with:
  export PATH="$HOME/.local/bin:$PATH"

Or install directly with:
  uv tool install kaggle

If you prefer pip:
  export PATH="$(python3 - <<'PY'
import site
print(site.USER_BASE + "/bin")
PY
):$PATH"
EOF
    return 1
  fi

  return 0
}


ensure_kaggle_credentials() {
  local kaggle_json="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    return 0
  fi
  if [[ -f "$kaggle_json" ]]; then
    return 0
  fi

  cat >&2 <<EOF
Error: Kaggle credentials not found.

Accepted auth for the kaggle CLI:
  - environment variables: KAGGLE_USERNAME and KAGGLE_KEY
  - credentials file: $kaggle_json

Expected file if you use file-based auth:
  $kaggle_json

Create a Kaggle API token in your Kaggle account settings and place kaggle.json there.
EOF
  return 1
}


fetch_url() {
  local url="$1"
  if command -v wget >/dev/null 2>&1; then
    wget -qO- "$url"
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url"
    return
  fi

  echo "Error: wget or curl is required." >&2
  return 1
}


extract_rar() {
  local archive="$1"
  local dest_dir="$2"

  if command -v unrar >/dev/null 2>&1; then
    unrar x -o+ -inul "$archive" "$dest_dir"
    return
  fi
  if command -v 7z >/dev/null 2>&1; then
    7z x -y "-o$dest_dir" "$archive" >/dev/null
    return
  fi

  echo "Error: unrar or 7z is required for PU extraction." >&2
  return 1
}


download_cwru() {
  local outdir="$1"
  local dataset_id="javadseraj/cwru-bearing-fault-data-set"
  local target_dir="$outdir/javadseraj-cwru-bearing-fault-data-set"

  ensure_kaggle_cli || return 1
  ensure_kaggle_credentials || return 1

  if [[ -d "$target_dir/Datasets/CWRU" ]]; then
    echo "CWRU dataset already present at $target_dir"
    return 0
  fi

  mkdir -p "$target_dir"
  echo "Downloading CWRU dataset into $target_dir ..."
  kaggle datasets download -d "$dataset_id" -p "$target_dir" --unzip || return 1
  echo "CWRU dataset ready: $target_dir"
}


download_pu() {
  local outdir="$1"
  local base_url="https://groups.uni-paderborn.de/kat/BearingDataCenter/"
  local pu_dir="$outdir/paderborn-university-bearing-dataset"
  local rar_dir="$pu_dir/rars"
  local url_list

  require_cmd aria2c || return 1

  if find "$pu_dir" -type f ! -path "$rar_dir/*" -print -quit 2>/dev/null | grep -q .; then
    echo "PU dataset already present at $pu_dir"
    return 0
  fi

  mkdir -p "$rar_dir"
  if ! url_list="$(
    fetch_url "$base_url" \
    | grep -oE 'href="[^"]+\.rar"' \
    | sed -E 's/href="([^"]+)"/\1/' \
    | sed -E "s#^#${base_url}#"
  )"; then
    return 1
  fi

  if [[ -z "$url_list" ]]; then
    echo "No PU RAR links found at $base_url" >&2
    return 1
  fi

  echo "Downloading PU archives into $rar_dir ..."
  printf '%s\n' "$url_list" | aria2c -c -x 8 -s 8 -j 8 -d "$rar_dir" --input-file=- || return 1

  echo "Extracting PU archives into $pu_dir ..."
  for archive in "$rar_dir"/*.rar; do
    extract_rar "$archive" "$pu_dir" || return 1
    rm -f "$archive"
  done

  echo "PU dataset ready: $pu_dir"
}


main() {
  local repo_root
  local outdir
  local failures=()

  parse_args "$@"

  repo_root="$(resolve_repo_root)"
  load_env_file "$repo_root/.env"
  outdir="$repo_root/datasets"
  mkdir -p "$outdir"

  if [[ "$DOWNLOAD_CWRU" -eq 1 ]]; then
    if ! download_cwru "$outdir"; then
      failures+=("cwru")
    fi
  fi

  if [[ "$DOWNLOAD_PU" -eq 1 ]]; then
    if ! download_pu "$outdir"; then
      failures+=("pu")
    fi
  fi

  if [[ "${#failures[@]}" -gt 0 ]]; then
    echo >&2
    echo "Dataset download completed with failures: ${failures[*]}" >&2
    exit 1
  fi
}


main "$@"
