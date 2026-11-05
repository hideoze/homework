#!/usr/bin/env bash
# Rebuild on this CPU, check functionality, then produce host-local A/B/C evidence.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash run_server_validation.sh [NEW_OUTPUT_DIR] [spmmprof run options]'
  echo 'Example: bash run_server_validation.sh results/server_4t --threads 4'
  echo 'A successful process exit does not imply that all performance acceptance gates passed.'
  exit 0
fi

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ required")'
for tool in g++ make; do
  command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 2; }
done
if ! command -v objdump >/dev/null; then
  echo 'Warning: objdump unavailable; static SIMD instruction evidence will be unavailable.' >&2
fi

out="${1:-results/server_$(date +%Y%m%d_%H%M%S)}"
if (( $# > 0 )); then shift; fi
for arg in "$@"; do
  if [[ "$arg" == "--out" || "$arg" == --out=* ]]; then
    echo 'Use the first positional argument for the output directory, not --out.' >&2
    exit 2
  fi
done
if [[ -e "$out" ]]; then
  echo "Refusing to overwrite existing output: $out" >&2
  exit 2
fi
# Clean before creating logs, so no foreign -march=native binary is reused.
make clean
mkdir -p -- "$out"
out="$(cd -- "$out" && pwd)"
make -j2 2>&1 | tee "$out/build.log"
"$PYTHON" -m unittest discover -s tests -v 2>&1 | tee "$out/tests.log"
"$PYTHON" spmmprof.py run --out "$out" "$@" 2>&1 | tee "$out/run.log"
echo "Results: $out"
echo "Read $out/evaluation.md and $out/profile_report.md; completion is not an all-pass assertion."
