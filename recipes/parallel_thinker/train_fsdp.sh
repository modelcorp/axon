set -x

# Parallel Thinker Math Training (Solver -> Rewriter -> Selector)
#
# Trains a single model to play all three roles in a collaborative pipeline.
# Reuses data from recipes/math/ — run `python recipes/math/data.py` first.

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurable Arguments
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"
loss_mode="gspo"
engine="vllm"
hybrid_engine=True

# Extract model name from model path for experiment naming
model_name=$(basename "$model_path")

gpu_memory_utilization=0.85
max_token_len_per_gpu=32768
tensor_model_parallel_size=8
pipeline_model_parallel_size=1
data_parallel_size=1

# Set algorithm-specific parameters
if [ "$loss_mode" = "gspo" ]; then
    clip_ratio_low=0.003
    clip_ratio_high=0.005
    token_reduce=mean
    batch_reduce=step-mean
    experiment_name="${model_name}-parallel_thinker-gspo"
else
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    token_reduce=mean-norm
    batch_reduce=step-mean
    experiment_name="${model_name}-parallel_thinker-ppo"
fi

# ---- Prepare data if needed (reuse recipes/math/data.py) ----
DATA_DIR="${DATA_DIR:-$AXON_DIR/data/math}"

python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    advantage=identity \
    train_files=${DATA_DIR}/train/deepscaler.parquet \
    val_files=${DATA_DIR}/test/math.parquet \
    train_batch_size=128 \
    max_prompt_length=2048 \
    max_seq_length=16384 \
    prompt_truncation=null \
    model_path=${model_path} \
    hybrid_engine=${hybrid_engine} \
    actor.fsdp.use_remove_padding=True \
    actor.fsdp.enable_activation_offload=False \
    actor.fsdp.use_fused_kernels=True \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.strategy="fsdp2" \
    loss=${loss_mode} \
    loss_args.token_reduce=${token_reduce} \
    loss_args.batch_reduce=${batch_reduce} \
    loss_args.clip_ratio_low=${clip_ratio_low} \
    loss_args.clip_ratio_high=${clip_ratio_high} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=0.01 \
    mini_batch_size=64 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.grad_norm_threshold=1000 \
    actor.fsdp.offload_policy=False \
    actor.fsdp.model_dtype="fp32" \
    actor.fsdp.fsdp_size=-1 \
    actor.fsdp.reshard_after_forward=False \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.tensor_model_parallel_size=${tensor_model_parallel_size} \
    sampler.pipeline_model_parallel_size=${pipeline_model_parallel_size} \
    sampler.data_parallel_size=${data_parallel_size} \
    sampler.enable_prefix_caching=False \
    sampler.name=${engine} \
    sampler.enforce_eager=False \
    decoding.temperature=1.0 \
    decoding.n=1 \
    sampler.gpu_memory_utilization=${gpu_memory_utilization} \
    actor.forward_use_dynamic_bsz=True \
    actor.forward_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor.forward_micro_batch_size_per_gpu=1 \
    validation.before_train=False \
    validation.steps=10 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger='[console,wandb]' \
    project_name='parallel_thinker_math' \
    experiment_name=${experiment_name} \
    num_gpus_per_node=8 \
    sampler_trainer_gpu_ratio=1 \
    enable_ray_collective=False \
    num_nodes=1 \
    save_steps=20 \
    total_epochs=1000000 \
    use_sampler_logprobs=False \
    drop_zero_advantage_samples=false \
    program.name=${SCRIPT_DIR}/program.py:ParallelThinkerProgram \
    +program.num_parallel=5 \
    +program.correct_reward_weight=1.2 \
    +program.incorrect_reward_weight=0.8 \
    "$@"
