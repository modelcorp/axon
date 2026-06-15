set -x

MINIWOB_URL="file://$HOME/miniwob-plusplus/miniwob/html/miniwob/"
# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MINIWOB_URL=${MINIWOB_URL}

# Parse command line arguments
model_path="Qwen/Qwen3-8B"  # Default model
loss_mode="ppo"  # Default loss mode``
engine="vllm"  # Default engine

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model_path="$2"
            shift 2
            ;;
        --loss)
            loss_mode="$2"
            shift 2
            ;;
        --engine)
            engine="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL_PATH] [--loss LOSS_MODE] [--engine ENGINE]"
            exit 1
            ;;
    esac
done

# Validate loss mode
case $loss_mode in
    "gspo"|"ppo")
        ;;
    *)
        echo "Invalid loss mode. Use: gspo or ppo"
        exit 1
        ;;
esac

# Validate engine
case $engine in
    "vllm"|"sglang")
        ;;
    *)
        echo "Invalid engine. Use: vllm or sglang"
        exit 1
        ;;
esac

# Extract model name from model path for experiment naming
model_name=$(basename "$model_path")

# Set algorithm-specific parameters
if [ "$loss_mode" = "gspo" ]; then
    clip_ratio_low=0.0003
    clip_ratio_high=0.0004
    token_reduce=mean
    batch_reduce=step-mean
    experiment_name="${model_name}-miniwob_agent-gspo"
    use_sampler_logprobs=False
else
    # ppo algorithm
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    token_reduce=mean-norm
    batch_reduce=step-mean
    experiment_name="${model_name}-miniwob_agent-grpo++"
    use_sampler_logprobs=False
fi

# For step-level PPO:
# accumulate_thinking=False
# accumulate_history=False

python3 -m axon.driver.train_agent_ppo \
    advantage=loop \
    train_files=${AXON_DIR}/data/miniwob/train/miniwob.parquet \
    val_files=${AXON_DIR}/data/miniwob/test/miniwob.parquet \
    train_batch_size=16 \
    max_prompt_length=6144 \
    max_seq_length=8192 \
    loss=${loss_mode} \
    loss_args.token_reduce=${token_reduce} \
    loss_args.batch_reduce=${batch_reduce} \
    loss_args.clip_ratio_low=${clip_ratio_low} \
    loss_args.clip_ratio_high=${clip_ratio_high} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    model_path=${model_path} \
    hybrid_engine=True \
    actor.optimizer_args.lr=1e-6 \
    actor.fsdp.use_remove_padding=True \
    mini_batch_size=16 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=24000 \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.grad_norm_threshold=1000 \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.tensor_model_parallel_size=1 \
    sampler.name=${engine} \
    sampler.enforce_eager=False \
    decoding.temperature=1.0 \
    decoding.n=16 \
    sampler.gpu_memory_utilization=0.85 \
    actor.forward_micro_batch_size_per_gpu=1 \
    ref.param_offload=True \
    ref.forward_micro_batch_size_per_gpu=1 \
    validation.before_train=True \
    validation.steps=10 \
    validation.decoding.n=1 \
    validation.decoding.temperature=0.7 \
    validation.decoding.top_p=0.8 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='miniwob-agent' \
    experiment_name=${experiment_name} \
    num_gpus_per_node=8 \
    enable_ray_collective=False \
    num_nodes=1 \
    save_steps=1000 \
    total_epochs=100 \
    engine_endpoint.enable=False \
    use_sampler_logprobs=False \
    +engine_args.disable_thinking=True \
    partial_rollout.enable=False \
    partial_rollout.n_iters=2 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    max_concurrency=64 \
    +program.env_name=${SCRIPT_DIR}/env.py:BrowserGymEnv \
    +program.env_args.subtask=miniwob \
    +program.env_args.miniwob_url=${MINIWOB_URL} \
    +program.env_args.max_turns=10 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:MiniWobAgent \
    +program.accumulate_history=True \
    +program.accumulate_thinking=True