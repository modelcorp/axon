#!/bin/bash
set -x

# =============================================================================
# Formal Math (Lean 4) RL Training with axon
#
# Prerequisites:
#   1. Install kimina-client: pip install kimina-client
#   2. Create Docker network: docker network create formal_math
#   3. Prepare data: python recipes/formal_math/data.py
#   4. Ensure Docker is accessible (for Kimina lean-server containers)
#
# The reward function automatically spawns Kimina Docker containers on each
# Ray node to verify Lean 4 proofs during training.
# =============================================================================

AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model - default to Qwen3-8B, can override via MODEL_PATH env var
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}

python3 -m axon.driver.train_agent_ppo \
    advantage=grpo \
    train_files=${AXON_DIR}/data/formal_math/train/flc.parquet \
    val_files=${AXON_DIR}/data/formal_math/test/minif2f.parquet \
    train_batch_size=32 \
    max_prompt_length=2048 \
    max_seq_length=8192 \
    model_path=${MODEL_PATH} \
    hybrid_engine=True \
    actor.optimizer_args.lr=1e-6 \
    actor.fsdp.use_remove_padding=True \
    mini_batch_size=8 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=6144 \
    loss=ppo \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.param_offload=False \
    actor.optimizer_offload=False \
    sampler.tensor_model_parallel_size=1 \
    sampler.name=vllm \
    decoding.temperature=1.0 \
    decoding.n=8 \
    sampler.gpu_memory_utilization=0.85 \
    sampler.enforce_eager=False \
    ref.param_offload=True \
    validation.before_train=False \
    validation.steps=20 \
    validation.decoding.n=1 \
    validation.decoding.temperature=0.6 \
    validation.decoding.top_p=0.95 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='formal-math-lean4' \
    experiment_name='formal-math-qwen3-8b' \
    num_gpus_per_node=8 \
    num_nodes=1 \
    save_steps=20 \
    total_epochs=1000 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:FormalMathEnvironment \
    +program.env_args.max_turns=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:FormalMathAgent
