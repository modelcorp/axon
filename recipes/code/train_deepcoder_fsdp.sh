#!/bin/bash
set -x

ulimit -n 1048576 

# Set ulysses_sequence_parallel_size based on response length
RESPONSE_LENGTH=16384
ULYSSES_SEQ_PARALLEL=1
if [ "$RESPONSE_LENGTH" -eq 16384 ]; then
    ULYSSES_SEQ_PARALLEL=1
elif [ "$RESPONSE_LENGTH" -eq 32768 ]; then
    ULYSSES_SEQ_PARALLEL=2
fi

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Train over 4 nodes, 8 A100-80GB GPUs per node.
python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    advantage=loop \
    train_files=${AXON_DIR}/data/code/train/deepcoder.parquet \
    val_files=${AXON_DIR}/data/code/test/test_livecodebench.parquet \
    train_batch_size=128 \
    max_prompt_length=2048 \
    max_seq_length=$RESPONSE_LENGTH \
    model_path=$MODEL_PATH \
    actor.fsdp.use_remove_padding=True \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.fsdp.enable_activation_offload=False \
    actor.fsdp.use_fused_kernels=True \
    actor.strategy="fsdp2" \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=0.01 \
    mini_batch_size=64 \
    loss=ppo \
    loss_args.token_reduce=mean \
    loss_args.batch_reduce=step-mean \
    loss_args.clip_ratio_low=0.2 \
    loss_args.clip_ratio_high=0.28 \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=32768 \
    actor.fsdp.ulysses_sequence_parallel_size=$ULYSSES_SEQ_PARALLEL \
    actor.fsdp.grad_norm_threshold=1000 \
    actor.fsdp.offload_policy=False \
    actor.fsdp.model_dtype="fp32" \
    actor.fsdp.fsdp_size=-1 \
    actor.fsdp.reshard_after_forward=False \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.name=vllm \
    sampler.tensor_model_parallel_size=8 \
    sampler.enable_prefix_caching=False \
    sampler.enforce_eager=False \
    decoding.temperature=1.0 \
    decoding.n=8 \
    sampler.gpu_memory_utilization=0.85 \
    ref.param_offload=True \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=2 \
    validation.decoding.temperature=0.6 \
    validation.decoding.top_p=0.95 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='deepcoder' \
    experiment_name='14b-16k-deepcoder' \
    num_gpus_per_node=8 \
    num_nodes=1 \
    save_steps=10 \
    total_epochs=100 \
    engine_endpoint.enable=True \
    overlong_filter=True \
    use_sampler_logprobs=False \
    partial_rollout.enable=False \
    partial_rollout.n_iters=2 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:CompetitionCodingEnv \
    +program.env_args.max_turns=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:CompetitionCodingAgent \
    +program.accumulate_history=True \
    +program.accumulate_thinking=True
