#!/bin/bash
set -x

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_OFFLOAD=True # True, False
LOSS_MODE=ppo # gspo, ppo
ENGINE=vllm # vllm, sglang
MAX_PROMPT_LENGTH=2048
MAX_SEQ_LENGTH=16384
MODEL_NAME=Qwen/Qwen3-30B-A3B

ACTOR_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_SEQ_LENGTH * 2))
INFER_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_SEQ_LENGTH * 6))

PROJECT_NAME='math-agent'
# Set algorithm-specific parameters
if [ "$LOSS_MODE" = "gspo" ]; then
    CLIP_RATIO_LOW=0.003
    CLIP_RATIO_HIGH=0.005
    TOKEN_REDUCE=mean
    BATCH_REDUCE=step-mean
    EXPERIMENT_NAME=${MODEL_NAME}-polaris-gspo-megatron
    USE_SAMPLER_LOGPROBS=False
else
    # ppo algorithm
    CLIP_RATIO_LOW=0.2
    CLIP_RATIO_HIGH=0.28
    TOKEN_REDUCE=mean-norm
    BATCH_REDUCE=step-mean
    EXPERIMENT_NAME=${MODEL_NAME}-polaris-grpo++-megatron
    USE_SAMPLER_LOGPROBS=False
fi

python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    strategy=megatron \
    advantage=loop \
    train_files=${AXON_DIR}/data/math/train/polaris.parquet \
    val_files=${AXON_DIR}/data/math/test/aime25.parquet \
    data_sampler=threshold_masking_curriculum \
    data_sampler_args.threshold=0.9 \
    train_batch_size=64 \
    max_prompt_length=${MAX_PROMPT_LENGTH} \
    max_seq_length=${MAX_SEQ_LENGTH} \
    model_path=${MODEL_NAME} \
    hybrid_engine=True \
    actor.use_fused_kernels=True \
    loss=${LOSS_MODE} \
    loss_args.clip_ratio_low=${CLIP_RATIO_LOW} \
    loss_args.clip_ratio_high=${CLIP_RATIO_HIGH} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    loss_args.token_reduce=${TOKEN_REDUCE} \
    loss_args.batch_reduce=${BATCH_REDUCE} \
    actor.use_dynamic_bsz=True \
    mini_batch_size=64 \
    actor.max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN} \
    actor.grad_clip=1.0 \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=0.1 \
    +actor.optimizer_args.override_optimizer_args.optimizer_offload_fraction=1.0 \
    +actor.optimizer_args.override_optimizer_args.overlap_cpu_optimizer_d2h_h2d=True \
    +actor.optimizer_args.override_optimizer_args.optimizer_cpu_offload=True \
    +actor.optimizer_args.override_optimizer_args.use_precision_aware_optimizer=True \
    actor.lr_scheduler_args.lr_warmup_steps=0 \
    actor.megatron.use_mbridge=True \
    actor.megatron.use_dist_checkpointing=False \
    actor.megatron.pipeline_model_parallel_size=2 \
    actor.megatron.virtual_pipeline_model_parallel_size=null \
    actor.megatron.tensor_model_parallel_size=4 \
    actor.megatron.expert_model_parallel_size=4 \
    actor.megatron.expert_tensor_parallel_size=1 \
    actor.megatron.context_parallel_size=1 \
    actor.param_offload=${ALL_OFFLOAD} \
    actor.optimizer_offload=${ALL_OFFLOAD} \
    actor.megatron.grad_offload=${ALL_OFFLOAD} \
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
    actor.forward_use_dynamic_bsz=True \
    actor.forward_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN} \
    sampler.name=${ENGINE} \
    sampler.enable_prefix_caching=False \
    sampler.enforce_eager=False \
    sampler.gpu_memory_utilization=0.7 \
    sampler.tensor_model_parallel_size=8 \
    sampler.enable_chunked_prefill=True \
    sampler.max_num_batched_tokens=32768 \
    decoding.temperature=1.0 \
    decoding.n=8 \
    ref.param_offload=${ALL_OFFLOAD} \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=4 \
    validation.decoding.temperature=0.78 \
    validation.decoding.top_p=0.8 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name=${PROJECT_NAME} \
    experiment_name=${EXPERIMENT_NAME} \
    num_gpus_per_node=8 \
    num_nodes=1 \
    save_steps=10000 \
    total_epochs=100000 \
    output_dir=${HOME}/axon-runs/ \
    moe_replay=True \
    use_sampler_logprobs=${USE_SAMPLER_LOGPROBS} \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:MathEnvironment \
    +program.env_args.max_turns=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:MathAgent
