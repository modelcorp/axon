set -x

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME=google/gemma-4-E2B-it
loss_mode=ppo # gspo, ppo
engine=vllm # vllm, sglang

# Set algorithm-specific parameters
if [ "$loss_mode" = "gspo" ]; then
    clip_ratio_low=0.0003
    clip_ratio_high=0.0004
    token_reduce=mean
    batch_reduce=step-mean
    experiment_name=${MODEL_NAME}-acereason_math-gspo-fsdp
    use_sampler_logprobs=False
else
    # ppo algorithm
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    token_reduce=mean-norm
    batch_reduce=step-mean
    experiment_name=${MODEL_NAME}-acereason_math-grpo++-fsdp
    use_sampler_logprobs=False
fi


python3 -m axon.driver.train_agent_ppo \
    advantage=loop \
    train_files=${AXON_DIR}/data/math/train/acereason_math.parquet \
    val_files=${AXON_DIR}/data/math/test/aime26.parquet \
    output_dir=${AXON_DIR}/axon-runs/ \
    train_batch_size=64 \
    max_prompt_length=2100 \
    max_seq_length=18484 \
    model_path=${MODEL_NAME} \
    data_sampler=threshold_masking_curriculum \
    data_sampler_args.threshold=0.9 \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.fsdp.use_remove_padding=True \
    actor.fsdp.enable_activation_offload=False \
    actor.fsdp.use_fused_kernels=True \
    actor.fsdp.reshard_after_forward=True \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.entropy_checkpointing=True \
    actor.fsdp.entropy_from_logits_with_chunking=True \
    hybrid_engine=True \
    actor.optimizer_args.lr=1e-6 \
    mini_batch_size=64 \
    loss=ppo \
    loss_args.token_reduce=${token_reduce} \
    loss_args.batch_reduce=${batch_reduce} \
    loss_args.clip_ratio_low=${clip_ratio_low} \
    loss_args.clip_ratio_high=${clip_ratio_high} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0.0 \
    loss_args.sampler_is=token \
    loss_args.sampler_is_threshold=2.0 \
    actor.max_token_len_per_gpu=20000 \
    actor.micro_batch_size_per_gpu=1 \
    actor.use_dynamic_bsz=True \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    actor.strategy="fsdp2" \
    actor.forward_use_dynamic_bsz=True \
    actor.forward_micro_batch_size_per_gpu=1 \
    sampler.tensor_model_parallel_size=1 \
    sampler.enable_prefix_caching=False \
    sampler.name=vllm \
    sampler.enforce_eager=False \
    sampler.gpu_memory_utilization=0.75 \
    decoding.temperature=0.7 \
    decoding.n=8 \
    ref.param_offload=True \
    validation.before_train=True \
    validation.steps=10 \
    validation.decoding.n=4 \
    validation.decoding.top_p=0.95 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='math-agent' \
    experiment_name=${experiment_name} \
    num_gpus_per_node=8 \
    num_nodes=1 \
    save_steps=100 \
    max_checkpoints_to_keep=3 \
    checkpoint_format=both \
    total_epochs=3 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:MathEnvironment \
    +program.env_args.max_turns=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:MathAgent \
#   actor.fsdp.model_dtype=bf16 \
#    +sampler.optimizer="adafactor"
