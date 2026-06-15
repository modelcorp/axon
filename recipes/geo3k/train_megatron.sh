set -x
# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_OFFLOAD=True # True, False
LOSS_MODE=gspo # gspo, grpo
# sudo sysctl -w kernel.yama.ptrace_scope=0
ENGINE=vllm # vllm, sglang
MAX_PROMPT_LENGTH=2048
MAX_SEQ_LENGTH=4096
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
ENABLE_PARTIAL_ROLLOUT=False  # Default enable partial sampler
HYBRID_ENGINE=True
STEP_MODE=False # Enable step-level mode (non-cumulative context)

ACTOR_MAX_TOKEN_LEN=32768
INFER_MAX_TOKEN_LEN=32768

# Define offload variables (preserving frozenlake variable names)
ACTOR_PARAM_OFFLOAD=${ALL_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ALL_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ALL_OFFLOAD}
REF_PARAM_OFFLOAD=${ALL_OFFLOAD}

# Default configuration (for 8B and other models)
PIPELINE_MODEL_PARALLEL_SIZE=1
VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE=null
TENSOR_MODEL_PARALLEL_SIZE=4
EXPERT_MODEL_PARALLEL_SIZE=1
EXPERT_TENSOR_PARALLEL_SIZE=1
SAMPLER_TENSOR_MODEL_PARALLEL_SIZE=4
NUM_NODES=1
SAMPLER_TRAINER_GPU_RATIO=1
WEIGHT_DECAY=0.01
GPU_MEMORY_UTILIZATION=0.85
USE_DIST_CHECKPOINTING=False

MODEL_NAME=$(basename "$MODEL_PATH")

# Set algorithm-specific parameters
if [ "$LOSS_MODE" = "gspo" ]; then
    CLIP_RATIO_LOW=0.003
    CLIP_RATIO_HIGH=0.005
    TOKEN_REDUCE=mean
    BATCH_REDUCE=step-mean
    EXPERIMENT_NAME=${MODEL_NAME}-geo3k_agent-gspo-megatron
    USE_SAMPLER_LOGPROBS=False
elif [ "$LOSS_MODE" = "ppo" ]; then
    # ppo algorithm
    CLIP_RATIO_LOW=0.2
    CLIP_RATIO_HIGH=0.28
    TOKEN_REDUCE=mean-norm
    BATCH_REDUCE=step-mean
    EXPERIMENT_NAME=${MODEL_NAME}-geo3k_agent-grpo-megatron
    USE_SAMPLER_LOGPROBS=False
fi

if [ "$STEP_MODE" = "True" ]; then
    ACCUMULATE_HISTORY=False
    USE_MULTISTEP_PROMPT=False
else
    ACCUMULATE_HISTORY=True
    USE_MULTISTEP_PROMPT=True
fi


python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    strategy=megatron \
    advantage=grpo \
    train_files=${AXON_DIR}/data/geo3k/train.parquet \
    val_files=${AXON_DIR}/data/geo3k/test.parquet \
    train_batch_size=64 \
    max_prompt_length=${MAX_PROMPT_LENGTH} \
    max_seq_length=${MAX_SEQ_LENGTH} \
    model_path=${MODEL_PATH} \
    hybrid_engine=${HYBRID_ENGINE} \
    actor.use_fused_kernels=False \
    loss=${LOSS_MODE} \
    loss_args.clip_ratio_low=${CLIP_RATIO_LOW} \
    loss_args.clip_ratio_high=${CLIP_RATIO_HIGH} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    loss_args.token_reduce=${TOKEN_REDUCE} \
    loss_args.batch_reduce=${BATCH_REDUCE} \
    actor.use_dynamic_bsz=True \
    mini_batch_size=128 \
    actor.max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN} \
    actor.grad_clip=1.0 \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=${WEIGHT_DECAY} \
    actor.lr_scheduler_args.lr_warmup_steps=0 \
    actor.micro_batch_size_per_gpu=1 \
    actor.megatron.use_mbridge=True \
    actor.megatron.use_dist_checkpointing=${USE_DIST_CHECKPOINTING} \
    actor.megatron.pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL_SIZE} \
    actor.megatron.virtual_pipeline_model_parallel_size=${VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE} \
    actor.megatron.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    actor.megatron.expert_model_parallel_size=${EXPERT_MODEL_PARALLEL_SIZE} \
    actor.megatron.expert_tensor_parallel_size=${EXPERT_TENSOR_PARALLEL_SIZE} \
    actor.megatron.context_parallel_size=1 \
    actor.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor.megatron.vanilla_mbridge=True \
    actor.megatron.use_remove_padding=True \
    actor.megatron.sequence_parallel=True \
    +actor.optimizer_args.override_optimizer_args.optimizer_offload_fraction=1 \
    +actor.optimizer_args.override_optimizer_args.overlap_cpu_optimizer_d2h_h2d=True \
    +actor.optimizer_args.override_optimizer_args.use_precision_aware_optimizer=True \
    +actor.optimizer_args.override_optimizer_args.optimizer_cpu_offload=True \
    +actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor.megatron.override_transformer_config.moe_permute_fusion=True \
    actor.forward_use_dynamic_bsz=True \
    actor.forward_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN} \
    sampler.name=vllm \
    sampler.offload_sampler=${HYBRID_ENGINE} \
    sampler.enable_prefix_caching=False \
    sampler.enforce_eager=False \
    sampler.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    sampler.tensor_model_parallel_size=${SAMPLER_TENSOR_MODEL_PARALLEL_SIZE} \
    sampler.enable_chunked_prefill=True \
    sampler.max_num_batched_tokens=32768 \
    decoding.temperature=0.7 \
    decoding.n=8 \
    actor.forward_micro_batch_size_per_gpu=1 \
    ref.param_offload=${REF_PARAM_OFFLOAD} \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=1 \
    validation.decoding.temperature=0.7 \
    validation.decoding.top_p=0.8 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='geo3k-agent' \
    experiment_name=${EXPERIMENT_NAME} \
    num_gpus_per_node=8 \
    num_nodes=${NUM_NODES} \
    sampler_trainer_gpu_ratio=${SAMPLER_TRAINER_GPU_RATIO} \
    save_steps=10000 \
    total_epochs=10000 \
    output_dir=${HOME}/checkpoints/${EXPERIMENT_NAME} \
    moe_replay=False \
    use_dummy_batch=False \
    use_sampler_logprobs=${USE_SAMPLER_LOGPROBS} \
    +engine_args.disable_thinking=True \
    partial_rollout.enable=${ENABLE_PARTIAL_ROLLOUT} \
    partial_rollout.n_iters=2 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:Geo3kEnvironment \
    +program.env_args.max_turns=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:Geo3kAgent