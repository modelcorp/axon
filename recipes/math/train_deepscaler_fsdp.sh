#!/bin/bash
set -x

# Parse command line arguments
context="8k"  # default value
model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # default value
while [[ $# -gt 0 ]]; do
    case $1 in
        --context)
            context="$2"
            shift 2
            ;;
        --model)
            model_path="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


if [ "$context" == "8k" ]; then
   num_nodes=1
   max_seq_length=$((8192 + 2048))
   exp_name="deepscaler-8k-run"
   temperature=1.0
elif [ "$context" == "16k" ]; then
   num_nodes=1
   max_seq_length=$((16384 + 2048))
   exp_name="deepscaler-16k-run"
   temperature=1.2
elif [ "$context" == "24k" ]; then
   num_nodes=2
   max_seq_length=$((24576 + 2048))
   exp_name="deepscaler-24k-run"
   temperature=1.4
else
   echo "Invalid context: $context"
   exit 1
fi

# Train over a single node, 8 A100-80GB GPUs.
python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    advantage=grpo \
    train_files=${AXON_DIR}/data/math/train/deepscaler.parquet \
    val_files=${AXON_DIR}/data/math/test/aime24.parquet \
    train_batch_size=128 \
    max_prompt_length=2048 \
    max_seq_length=$max_seq_length \
    model_path=$model_path \
    actor.fsdp.enable_gradient_checkpointing=True \
    hybrid_engine=True \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=0.01 \
    actor.fsdp.use_remove_padding=True \
    loss="ppo" \
    loss_args.token_reduce=mean \
    loss_args.batch_reduce=step-mean \
    loss_args.clip_ratio_low=0.2 \
    loss_args.clip_ratio_high=0.2 \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    mini_batch_size=64 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=32768 \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.grad_norm_threshold=1000 \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.tensor_model_parallel_size=1 \
    sampler.enable_prefix_caching=False \
    sampler.name="vllm" \
    sampler.enforce_eager=False \
    decoding.temperature=$temperature \
    decoding.n=8 \
    sampler.gpu_memory_utilization=0.85 \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=16 \
    validation.decoding.temperature=0.6 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name="deepscaler" \
    experiment_name="$exp_name" \
    num_gpus_per_node=8 \
    enable_ray_collective=False \
    num_nodes=$num_nodes \
    save_steps=20 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:MathEnvironment \
    +program.env_args.max_turns=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:MathAgent \
    total_epochs=1000000