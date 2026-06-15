# Shared training-loop body for the FrozenLake recipe.
#
# Source this from a per-variant script after setting at least:
#   MODEL_PATH                          — HF model id or local path (required)
#   NUM_NODES                           — node count (default 1)
#   NUM_GPUS_PER_NODE                   — per-node GPU count (default 8)
#   HYBRID_ENGINE                       — True / False (default True)
#   TENSOR_MODEL_PARALLEL_SIZE          — actor TP (required)
#   EXPERT_MODEL_PARALLEL_SIZE          — actor EP (required for MoE)
#   PIPELINE_MODEL_PARALLEL_SIZE        — actor PP (default 1)
#   SAMPLER_TENSOR_MODEL_PARALLEL_SIZE  — sampler TP (required)
#
# Optional overrides: LOSS_MODE (ppo / gspo, default ppo), STEP_MODE,
# ACTOR_MAX_TOKEN_LEN, INFER_MAX_TOKEN_LEN, GPU_MEMORY_UTILIZATION,
# SAMPLER_TRAINER_GPU_RATIO, WEIGHT_DECAY, USE_DIST_CHECKPOINTING,
# ENABLE_PARTIAL_ROLLOUT, ALL_OFFLOAD.

set -x

AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"

# --- Defaults ---------------------------------------------------------------
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TENSOR_MODEL_PARALLEL_SIZE:?TENSOR_MODEL_PARALLEL_SIZE is required}"
: "${SAMPLER_TENSOR_MODEL_PARALLEL_SIZE:?SAMPLER_TENSOR_MODEL_PARALLEL_SIZE is required}"

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
HYBRID_ENGINE=${HYBRID_ENGINE:-True}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE=${VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE:-null}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
SAMPLER_TRAINER_GPU_RATIO=${SAMPLER_TRAINER_GPU_RATIO:-1}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
LOSS_MODE=${LOSS_MODE:-ppo}
ENGINE=${ENGINE:-vllm}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
ACTOR_MAX_TOKEN_LEN=${ACTOR_MAX_TOKEN_LEN:-32768}
INFER_MAX_TOKEN_LEN=${INFER_MAX_TOKEN_LEN:-32768}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.78}
USE_DIST_CHECKPOINTING=${USE_DIST_CHECKPOINTING:-False}
ENABLE_PARTIAL_ROLLOUT=${ENABLE_PARTIAL_ROLLOUT:-False}
PARTIAL_ROLLOUT_N_ITERS=${PARTIAL_ROLLOUT_N_ITERS:-2}
STEP_MODE=${STEP_MODE:-False}
MAX_STEPS=${MAX_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-2000}

# Offload knobs derived from ALL_OFFLOAD unless explicitly set.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-${ALL_OFFLOAD}}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-${ALL_OFFLOAD}}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-${ALL_OFFLOAD}}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-${ALL_OFFLOAD}}

MODEL_NAME=$(basename "$MODEL_PATH")
PROJECT_NAME=${PROJECT_NAME:-frozenlake-agent}

# --- Algorithm-specific knobs -----------------------------------------------
if [ "$LOSS_MODE" = "gspo" ]; then
    CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.003}
    CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.005}
    TOKEN_REDUCE=${TOKEN_REDUCE:-mean}
    BATCH_REDUCE=${BATCH_REDUCE:-step-mean}
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_NAME}-frozenlake-gspo-megatron${EXPERIMENT_SUFFIX:-}}
    USE_SAMPLER_LOGPROBS=${USE_SAMPLER_LOGPROBS:-False}
elif [ "$LOSS_MODE" = "ppo" ]; then
    CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
    CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
    TOKEN_REDUCE=${TOKEN_REDUCE:-mean-program}
    BATCH_REDUCE=${BATCH_REDUCE:-program-mean}
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_NAME}-frozenlake-grpo-megatron${EXPERIMENT_SUFFIX:-}}
    USE_SAMPLER_LOGPROBS=${USE_SAMPLER_LOGPROBS:-False}
else
    echo "Unknown LOSS_MODE=${LOSS_MODE}; expected ppo or gspo." >&2
    exit 1
fi

if [ "$STEP_MODE" = "True" ]; then
    ACCUMULATE_HISTORY=False
    USE_MULTISTEP_PROMPT=False
else
    ACCUMULATE_HISTORY=True
    USE_MULTISTEP_PROMPT=True
fi

# --- Launch -----------------------------------------------------------------
python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    strategy=megatron \
    advantage=loop \
    train_files=${AXON_DIR}/data/frozenlake/train.parquet \
    val_files=${AXON_DIR}/data/frozenlake/test.parquet \
    train_batch_size=64 \
    max_prompt_length=${MAX_PROMPT_LENGTH} \
    max_seq_length=${MAX_SEQ_LENGTH} \
    model_path=${MODEL_PATH} \
    hybrid_engine=${HYBRID_ENGINE} \
    actor.use_fused_kernels=True \
    loss=${LOSS_MODE} \
    loss_args.clip_ratio_low=${CLIP_RATIO_LOW} \
    loss_args.clip_ratio_high=${CLIP_RATIO_HIGH} \
    loss_args.token_reduce=${TOKEN_REDUCE} \
    loss_args.batch_reduce=${BATCH_REDUCE} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    ++loss_args.sampler_is=token \
    ++loss_args.sampler_is_threshold=2.0 \
    actor.use_dynamic_bsz=True \
    mini_batch_size=64 \
    actor.max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN} \
    actor.grad_clip=1.0 \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=${WEIGHT_DECAY} \
    +actor.optimizer_args.override_optimizer_args.optimizer_offload_fraction=1.0 \
    +actor.optimizer_args.override_optimizer_args.overlap_cpu_optimizer_d2h_h2d=True \
    +actor.optimizer_args.override_optimizer_args.optimizer_cpu_offload=True \
    +actor.optimizer_args.override_optimizer_args.use_precision_aware_optimizer=True \
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
    +actor.megatron.override_transformer_config.attention_softmax_in_fp32=True \
    +actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor.megatron.override_transformer_config.memory_efficient_layer_norm=False \
    +actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor.megatron.override_transformer_config.apply_rope_fusion=True \
    actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor.megatron.override_transformer_config.cross_entropy_loss_fusion=True \
    +actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    +actor.megatron.override_transformer_config.moe_enable_deepep=False \
    +actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall \
    +actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=False \
    +actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=False \
    +actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=null \
    actor.forward_use_dynamic_bsz=True \
    actor.forward_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN} \
    sampler.name=${ENGINE} \
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
    validation.decoding.n=4 \
    validation.decoding.temperature=0.7 \
    validation.decoding.top_p=0.8 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name=${PROJECT_NAME} \
    experiment_name=${EXPERIMENT_NAME} \
    num_gpus_per_node=${NUM_GPUS_PER_NODE} \
    num_nodes=${NUM_NODES} \
    sampler_trainer_gpu_ratio=${SAMPLER_TRAINER_GPU_RATIO} \
    save_steps=${SAVE_STEPS} \
    total_epochs=10000 \
    output_dir=${HOME}/axon-runs/ \
    moe_replay=True \
    use_dummy_batch=False \
    use_sampler_logprobs=${USE_SAMPLER_LOGPROBS} \
    +engine_args.disable_thinking=True \
    partial_rollout.enable=${ENABLE_PARTIAL_ROLLOUT} \
    partial_rollout.n_iters=${PARTIAL_ROLLOUT_N_ITERS} \
    stepwise_advantage_mode="broadcast" \
    max_steps=${MAX_STEPS} \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:FrozenLakeEnv \
    +program.env_args.max_turns=8 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:FrozenLakeAgent \
    +program.agent_args.use_multistep_prompt=${USE_MULTISTEP_PROMPT} \
    +program.accumulate_history=${ACCUMULATE_HISTORY} \
    +program.accumulate_thinking=True \
    "$@"
