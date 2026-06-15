#!/bin/bash

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent directory has prepare_verifiers_data.py
VERIFIERS_DIR="$(dirname "$SCRIPT_DIR")"

python "$VERIFIERS_DIR/prepare_verifiers_data.py" --env-module wordle