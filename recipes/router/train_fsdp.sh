set -x

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurable Arguments
model_path="Qwen/Qwen3-8B"  # Default model
loss_mode="gspo"  # gspo, ppo
engine="vllm"  # vllm, sglang
hybrid_engine=True

# Extract model name from model path for experiment naming
model_name=$(basename "$model_path")

gpu_memory_utilization=0.85
max_token_len_per_gpu=24000
tensor_model_parallel_size=1
pipeline_model_parallel_size=1
data_parallel_size=1

# Set algorithm-specific parameters
if [ "$loss_mode" = "gspo" ]; then
    clip_ratio_low=0.0003
    clip_ratio_high=0.0004
    token_reduce=mean
    batch_reduce=step-mean
    experiment_name="${model_name}-router_agent-gspo"
    use_sampler_logprobs=False
else
    # ppo algorithm
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    token_reduce=mean-norm
    batch_reduce=step-mean
    experiment_name="${model_name}-router_agent-grpo++"
    use_sampler_logprobs=False
fi

# Note: The router environment uses math data. You can use the math data directly
# or run `python recipes/math/data.py` to prepare router-specific datasets.
# Default paths use the math dataset.

python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    advantage=loop \
    train_files=${AXON_DIR}/data/math/train/math.parquet \
    val_files=${AXON_DIR}/data/math/test/math.parquet \
    train_batch_size=64 \
    max_prompt_length=2048 \
    max_seq_length=8192\
    loss=${loss_mode} \
    loss_args.token_reduce=${token_reduce} \
    loss_args.batch_reduce=${batch_reduce} \
    loss_args.clip_ratio_low=${clip_ratio_low} \
    loss_args.clip_ratio_high=${clip_ratio_high} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    model_path=${model_path} \
    hybrid_engine=${hybrid_engine} \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=0.01 \
    actor.fsdp.use_remove_padding=True \
    mini_batch_size=64 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.grad_norm_threshold=1000 \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.tensor_model_parallel_size=${tensor_model_parallel_size} \
    sampler.pipeline_model_parallel_size=${pipeline_model_parallel_size} \
    sampler.data_parallel_size=${data_parallel_size} \
    sampler.enable_prefix_caching=False \
    sampler.name=${engine} \
    sampler.enforce_eager=False \
    decoding.temperature=0.7 \
    decoding.n=8 \
    sampler.gpu_memory_utilization=${gpu_memory_utilization} \
    actor.forward_micro_batch_size_per_gpu=1 \
    ref.param_offload=True \
    ref.forward_micro_batch_size_per_gpu=1 \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=1 \
    validation.decoding.temperature=0.7 \
    validation.decoding.top_p=0.8 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='axon-router-agent' \
    experiment_name=${experiment_name} \
    num_gpus_per_node=8 \
    num_nodes=1 \
    save_steps=10000 \
    total_epochs=1000  \
    engine_endpoint.enable=True \
    overlong_filter=False\
    use_sampler_logprobs=False \
    partial_rollout.enable=False \
    partial_rollout.n_iters=2 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:RouterEnv \
    +program.env_args.max_turns=2 \
    +program.env_args.max_expert_calls=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:RouterAgent \
    +program.accumulate_history=True \
    +program.accumulate_thinking=True
