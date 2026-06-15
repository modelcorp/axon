#!/bin/bash
# Qwen3-30B-A3B (MoE) — 1 node × 8 H100s, disaggregated, one-off pipeline mode.
MODEL_PATH=Qwen/Qwen3-30B-A3B
NUM_NODES=1
HYBRID_ENGINE=False

TENSOR_MODEL_PARALLEL_SIZE=4
EXPERT_MODEL_PARALLEL_SIZE=4
PIPELINE_MODEL_PARALLEL_SIZE=1
SAMPLER_TENSOR_MODEL_PARALLEL_SIZE=4

GPU_MEMORY_UTILIZATION=0.85

# Enable Axon's one-off-pipeline mode (overlapped sample / train under disagg).
source "$(dirname "$0")/_common.sh" enable_one_off_pipeline=True
