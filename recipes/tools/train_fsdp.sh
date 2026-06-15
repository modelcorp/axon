#!/bin/bash
set -x

# Calculator Tool RL Training Script
#
# This script trains an RL agent to use tools (calculator, python interpreter)
# to solve math problems.
#
# Prerequisites:
#   1. Generate math data: python recipes/math/data.py
#
# Usage:
#   bash recipes/tools/train_fsdp.sh

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
# SCRIPT_DIR points to recipes/tools/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurable Arguments
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"
loss_mode="${LOSS_MODE:-gspo}"
engine="${ENGINE:-vllm}"
hybrid_engine="${HYBRID_ENGINE:-True}"

# Tools configuration
# Format: "name=relative_path:ClassName name2=relative_path2:ClassName2"
TOOLS="calculator=math_tools/calculator.py:CalculatorTool python=code_tools/python_interpreter.py:PythonInterpreter"

# Reward function
REWARD_FN="axon.utils.rewards.math_reward:math_reward_fn"

# Build tool_map arguments dynamically
TOOL_MAP_ARGS=""
for tool in $TOOLS; do
    tool_name="${tool%%=*}"
    tool_path="${tool#*=}"
    TOOL_MAP_ARGS="${TOOL_MAP_ARGS} +program.env_args.tool_map.${tool_name}=\"${SCRIPT_DIR}/${tool_path}\""
    TOOL_MAP_ARGS="${TOOL_MAP_ARGS} +program.agent_args.tool_map.${tool_name}=\"${SCRIPT_DIR}/${tool_path}\""
done

# Extract model name for experiment naming
model_name=$(basename "$model_path")

gpu_memory_utilization=0.85
max_token_len_per_gpu=16384

# Set algorithm-specific parameters
if [ "$loss_mode" = "gspo" ]; then
    clip_ratio_low=0.003
    clip_ratio_high=0.005
    token_reduce=mean
    batch_reduce=step-mean
    experiment_name="${model_name}-calculator_agent-gspo"
else
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    token_reduce=mean-norm
    batch_reduce=step-mean
    experiment_name="${model_name}-calculator_agent-grpo++"
fi

python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    advantage=rloo \
    train_files=${AXON_DIR}/data/math/train/deepscaler.parquet \
    val_files=${AXON_DIR}/data/math/test/math.parquet \
    train_batch_size=64 \
    max_prompt_length=2048 \
    max_seq_length=8192 \
    model_path=${model_path} \
    hybrid_engine=${hybrid_engine} \
    actor.fsdp.use_remove_padding=True \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.strategy="fsdp2" \
    loss=${loss_mode} \
    loss_args.token_reduce=${token_reduce} \
    loss_args.batch_reduce=${batch_reduce} \
    loss_args.clip_ratio_low=${clip_ratio_low} \
    loss_args.clip_ratio_high=${clip_ratio_high} \
    loss_args.kl_coef=0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=0.01 \
    mini_batch_size=64 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.tensor_model_parallel_size=1 \
    sampler.enable_prefix_caching=False \
    sampler.name=${engine} \
    decoding.temperature=0.7 \
    decoding.n=8 \
    sampler.gpu_memory_utilization=${gpu_memory_utilization} \
    actor.forward_use_dynamic_bsz=True \
    actor.forward_max_token_len_per_gpu=${max_token_len_per_gpu} \
    ref.param_offload=True \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=4 \
    validation.decoding.temperature=0.7 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='calculator-agent' \
    experiment_name=${experiment_name} \
    num_gpus_per_node=8 \
    num_nodes=1 \
    save_steps=1000 \
    total_epochs=100 \
    +engine_args.disable_thinking=True \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:ToolEnvironment \
    ${TOOL_MAP_ARGS} \
    +program.env_args.reward_fn="${REWARD_FN}" \
    +program.env_args.max_turns=5 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:ToolAgent \
    +program.accumulate_history=True \
    +program.accumulate_thinking=True
