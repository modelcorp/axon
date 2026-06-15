set -x

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}

python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    strategy=megatron \
    advantage=grpo \
    train_files=${AXON_DIR}/data/math/train/math.parquet \
    val_files=${AXON_DIR}/data/math/test/math.parquet \
    train_batch_size=128 \
    max_prompt_length=2048 \
    max_seq_length=6144 \
    model_path=Qwen/Qwen3-8B  \
    actor.megatron.use_mbridge=True \
    hybrid_engine=True \
    actor.optimizer_args.lr=1e-6 \
    mini_batch_size=128 \
    actor.micro_batch_size_per_gpu=1 \
    actor.megatron.pipeline_model_parallel_size=1 \
    actor.megatron.tensor_model_parallel_size=4 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=20000 \
    loss=ppo \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    sampler.tensor_model_parallel_size=1 \
    sampler.name=vllm \
    decoding.temperature=1.0 \
    decoding.n=4 \
    sampler.gpu_memory_utilization=0.3 \
    actor.forward_micro_batch_size_per_gpu=20 \
    actor.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    ref.param_offload=${REF_PARAM_OFFLOAD} \
    sampler.enforce_eager=False \
    validation.before_train=True \
    validation.steps=10 \
    validation.decoding.n=1 \
    validation.decoding.temperature=0.6 \
    validation.decoding.top_p=0.95 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='math-agent' \
    experiment_name='math-agent-4b' \
    num_gpus_per_node=8 \
    sampler_trainer_gpu_ratio=4 \
    num_nodes=1 \
    save_steps=2000 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:MathEnvironment \
    +program.env_args.max_turns=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:MathAgent \
    total_epochs=30 "${@:1}"
