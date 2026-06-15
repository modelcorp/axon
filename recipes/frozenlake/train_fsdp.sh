set -x

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurable Arguments
MODEL_PATH="google/gemma-4-31B-it"
# MODEL_PATH="google/gemma-4-26B-A4B-it"
LOSS_MODE="gspo"  # gspo, ppo
ENGINE=vllm  # vllm, sglang
ENABLE_PARTIAL_ROLLOUT=False
HYBRID_ENGINE=True
STEP_MODE=False  # Enable step-level mode (non-cumulative context)

MAX_PROMPT_LENGTH=2048
MAX_SEQ_LENGTH=4096
MAX_TOKEN_LEN_PER_GPU=32768

# Define offload variables
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True

# Sampler
SAMPLER_TENSOR_MODEL_PARALLEL_SIZE=8
SAMPLER_PIPELINE_MODEL_PARALLEL_SIZE=1
SAMPLER_DATA_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.75

# General
NUM_NODES=1
NUM_GPUS_PER_NODE=8
SAMPLER_TRAINER_GPU_RATIO=1
WEIGHT_DECAY=0.01

MODEL_NAME=$(basename "$MODEL_PATH")

PROJECT_NAME='frozenlake-agent'
# Set algorithm-specific parameters
if [ "$LOSS_MODE" = "gspo" ]; then
    CLIP_RATIO_LOW=0.003
    CLIP_RATIO_HIGH=0.005
    TOKEN_REDUCE=mean
    BATCH_REDUCE=step-mean
    EXPERIMENT_NAME="${MODEL_NAME}-frozenlake_agent-gspo"
    USE_SAMPLER_LOGPROBS=False
else
    # ppo algorithm
    CLIP_RATIO_LOW=0.2
    CLIP_RATIO_HIGH=0.28
    TOKEN_REDUCE=mean-norm
    BATCH_REDUCE=step-mean
    EXPERIMENT_NAME="${MODEL_NAME}-frozenlake_agent-grpo++"
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
    advantage=rloo \
    train_files=${AXON_DIR}/data/frozenlake/train.parquet \
    val_files=${AXON_DIR}/data/frozenlake/test.parquet \
    data_sampler=threshold_masking_curriculum \
    data_sampler_args.threshold=0.9 \
    train_batch_size=64 \
    max_prompt_length=${MAX_PROMPT_LENGTH} \
    max_seq_length=${MAX_SEQ_LENGTH} \
    prompt_truncation=null \
    model_path=${MODEL_PATH} \
    hybrid_engine=${HYBRID_ENGINE} \
    actor.fsdp.use_remove_padding=True \
    actor.fsdp.enable_activation_offload=False \
    actor.fsdp.use_fused_kernels=True \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.strategy="fsdp2" \
    loss=${LOSS_MODE} \
    loss_args.token_reduce=${TOKEN_REDUCE} \
    loss_args.batch_reduce=${BATCH_REDUCE} \
    loss_args.clip_ratio_low=${CLIP_RATIO_LOW} \
    loss_args.clip_ratio_high=${CLIP_RATIO_HIGH} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=${WEIGHT_DECAY} \
    mini_batch_size=64 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.grad_norm_threshold=1000 \
    actor.fsdp.offload_policy=False \
    actor.fsdp.model_dtype="fp32" \
    actor.fsdp.fsdp_size=-1 \
    actor.fsdp.reshard_after_forward=True \
    actor.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor.forward_use_dynamic_bsz=True \
    actor.forward_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor.forward_micro_batch_size_per_gpu=1 \
    sampler.name=${ENGINE} \
    sampler.tensor_model_parallel_size=${SAMPLER_TENSOR_MODEL_PARALLEL_SIZE} \
    sampler.pipeline_model_parallel_size=${SAMPLER_PIPELINE_MODEL_PARALLEL_SIZE} \
    sampler.data_parallel_size=${SAMPLER_DATA_PARALLEL_SIZE} \
    sampler.enable_prefix_caching=False \
    sampler.enforce_eager=False \
    sampler.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    decoding.temperature=0.7 \
    decoding.n=8 \
    ref.param_offload=${REF_PARAM_OFFLOAD} \
    ref.forward_micro_batch_size_per_gpu=1 \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=4 \
    validation.decoding.temperature=0.7 \
    validation.decoding.top_p=0.8 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name=${PROJECT_NAME} \
    experiment_name=${EXPERIMENT_NAME} \
    num_gpus_per_node=${NUM_GPUS_PER_NODE} \
    sampler_trainer_gpu_ratio=${SAMPLER_TRAINER_GPU_RATIO} \
    enable_ray_collective=False \
    num_nodes=${NUM_NODES} \
    save_steps=10000 \
    total_epochs=1000 \
    use_sampler_logprobs=${USE_SAMPLER_LOGPROBS} \
    +engine_args.disable_thinking=True \
    partial_rollout.enable=${ENABLE_PARTIAL_ROLLOUT} \
    partial_rollout.n_iters=2 \
    stepwise_advantage_mode="broadcast" \
    max_steps=10 \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:FrozenLakeEnv \
    +program.env_args.max_turns=8 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:FrozenLakeAgent \
    +program.agent_args.use_multistep_prompt=${USE_MULTISTEP_PROMPT} \
    +program.accumulate_history=${ACCUMULATE_HISTORY} \
    +program.accumulate_thinking=True
