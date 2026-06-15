#!/bin/bash
set -x

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Sudoku training script (Qwen3-30B-A3B MoE on Megatron).
#
# Reward: sparse binary — solve = +1.0, invalid move = -0.1, else = 0.0.
# No partial credit (avoids fill-easy-cells reward hacking).
#
# Data: curriculum dataset with 5 difficulty tiers (4x4 easy → 9x9 expert).
# Per-row parquet values (size, num_clues, max_turns) override the defaults
# below, so the env_args here are only fallbacks.
#
# Prerequisites:
#   python recipes/sudoku/data.py          # generates data/sudoku/{train,test}.parquet
#
# Override at the command line, e.g.
#   LOSS_MODE=gspo MAX_SEQ_LENGTH=32768 bash train_sudoku_qwen_30b_a3b.sh
# ============================================================================

ALL_OFFLOAD=True
LOSS_MODE=ppo # gspo, ppo
ENGINE=vllm   # vllm, sglang
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
ENABLE_PARTIAL_ROLLOUT=False
HYBRID_ENGINE=True
STEP_MODE=False

# Sudoku env defaults (overridden per-row by the curriculum parquet)
# These only kick in if a parquet row is missing the column.
DEFAULT_SIZE=${DEFAULT_SIZE:-9}
DEFAULT_NUM_CLUES=${DEFAULT_NUM_CLUES:-36}
DEFAULT_MAX_TURNS=${DEFAULT_MAX_TURNS:-200}

# max_steps: the training loop's hard cap on env steps per episode.
# Must be >= the largest max_turns in any curriculum tier (200 for expert).
MAX_STEPS=${MAX_STEPS:-200}

# Optimization
LEARNING_RATE=${LEARNING_RATE:-1e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
DECODING_N=${DECODING_N:-8}

ACTOR_MAX_TOKEN_LEN=32768
INFER_MAX_TOKEN_LEN=32768

# Define offload variables
ACTOR_PARAM_OFFLOAD=${ALL_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ALL_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ALL_OFFLOAD}
REF_PARAM_OFFLOAD=${ALL_OFFLOAD}

# Default configuration for Qwen3-30B-A3B MoE
PIPELINE_MODEL_PARALLEL_SIZE=2
VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE=null
TENSOR_MODEL_PARALLEL_SIZE=4
EXPERT_MODEL_PARALLEL_SIZE=4
EXPERT_TENSOR_PARALLEL_SIZE=1
SAMPLER_TENSOR_MODEL_PARALLEL_SIZE=8
NUM_NODES=1
SAMPLER_TRAINER_GPU_RATIO=1
WEIGHT_DECAY=0.01
GPU_MEMORY_UTILIZATION=0.78
USE_DIST_CHECKPOINTING=False

MODEL_NAME=$(basename "$MODEL_PATH")

PROJECT_NAME='sudoku-agent'
if [ "$LOSS_MODE" = "gspo" ]; then
    CLIP_RATIO_LOW=0.003
    CLIP_RATIO_HIGH=0.005
    TOKEN_REDUCE=mean
    BATCH_REDUCE=step-mean
    EXPERIMENT_NAME=${MODEL_NAME}-sudoku_agent-gspo-megatron
    USE_SAMPLER_LOGPROBS=False
elif [ "$LOSS_MODE" = "ppo" ]; then
    CLIP_RATIO_LOW=0.2
    CLIP_RATIO_HIGH=0.28
    TOKEN_REDUCE=mean-program
    BATCH_REDUCE=program-mean
    EXPERIMENT_NAME=${MODEL_NAME}-sudoku_agent-grpo-megatron
    USE_SAMPLER_LOGPROBS=False
fi

if [ "$STEP_MODE" = "True" ]; then
    ACCUMULATE_HISTORY=False
else
    ACCUMULATE_HISTORY=True
fi

python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    strategy=megatron \
    advantage=loop \
    train_files=${AXON_DIR}/data/sudoku/train.parquet \
    val_files=${AXON_DIR}/data/sudoku/test.parquet \
    train_batch_size=${TRAIN_BATCH_SIZE} \
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
    actor.use_dynamic_bsz=True \
    mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor.max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN} \
    actor.grad_clip=1.0 \
    actor.optimizer_args.lr=${LEARNING_RATE} \
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
    sampler.name=vllm \
    sampler.offload_sampler=${HYBRID_ENGINE} \
    sampler.enable_prefix_caching=False \
    sampler.enforce_eager=False \
    sampler.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    sampler.tensor_model_parallel_size=${SAMPLER_TENSOR_MODEL_PARALLEL_SIZE} \
    sampler.enable_chunked_prefill=True \
    sampler.max_num_batched_tokens=32768 \
    decoding.temperature=0.7 \
    decoding.n=${DECODING_N} \
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
    num_gpus_per_node=8 \
    num_nodes=${NUM_NODES} \
    sampler_trainer_gpu_ratio=${SAMPLER_TRAINER_GPU_RATIO} \
    save_steps=200 \
    total_epochs=10000 \
    output_dir=${HOME}/axon-runs/ \
    moe_replay=True \
    use_dummy_batch=False \
    use_sampler_logprobs=${USE_SAMPLER_LOGPROBS} \
    +engine_args.disable_thinking=True \
    partial_rollout.enable=${ENABLE_PARTIAL_ROLLOUT} \
    partial_rollout.n_iters=2 \
    stepwise_advantage_mode="broadcast" \
    max_steps=${MAX_STEPS} \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:SudokuEnv \
    +program.env_args.max_turns=${DEFAULT_MAX_TURNS} \
    +program.env_args.size=${DEFAULT_SIZE} \
    +program.env_args.num_clues=${DEFAULT_NUM_CLUES} \
    +program.agent_name=${SCRIPT_DIR}/agent.py:SudokuAgent \
    +program.accumulate_history=${ACCUMULATE_HISTORY} \
    +program.accumulate_thinking=True
