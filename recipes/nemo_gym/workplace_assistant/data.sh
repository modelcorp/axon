#!/bin/bash
# Download and prepare data for the Workplace Assistant environment.
# Run from $AXON_DIR. Requires NeMo Gym installed at $HOME/Gym.
set -euo pipefail

NEMO_GYM_DIR="${NEMO_GYM_DIR:-$HOME/Gym}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Downloading Workplace Assistant dataset ==="
echo "NeMo Gym dir: $NEMO_GYM_DIR"

# Download + validate via NeMo Gym's ng_prepare_data
(
    cd "$NEMO_GYM_DIR"
    source .venv/bin/activate
    echo "hf_token: ${HF_TOKEN:?Set HF_TOKEN env var}" >> env.yaml
    config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
    resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

    ng_prepare_data "+config_paths=[${config_paths}]" \
        +output_dirpath=data/workplace_assistant \
        +mode=train_preparation \
        +should_download=true \
        +data_source=huggingface
)

echo "=== Preparing parquet for Axon ==="

python "${SCRIPT_DIR}/prepare_nemo_gym_data.py" \
    --input "${NEMO_GYM_DIR}/data/workplace_assistant/train.jsonl" \
    --test-input "${NEMO_GYM_DIR}/data/workplace_assistant/validation.jsonl" \
    --resource-server workplace_assistant

echo "=== Done ==="
echo "Output: data/nemo_gym/workplace_assistant/{train,test}.parquet"
