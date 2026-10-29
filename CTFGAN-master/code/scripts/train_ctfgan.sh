#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-$CODE_DIR/cfg/ctfgan_bird.yml}"
if [[ $# -gt 0 ]]; then shift; fi
exec python "$CODE_DIR/ctfgan.py" train --cfg "$CFG" "$@"
