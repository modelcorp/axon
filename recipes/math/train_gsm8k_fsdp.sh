set -x

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Train over a single node, 8 A100-80GB GPUs.
python3 -m axon.driver.train_agent_ppo \
    advantage=grpo \
    train_files=${AXON_DIR}/data/math/train/math.parquet \
    val_files=${AXON_DIR}/data/math/test/math.parquet \
    train_batch_size=64 \
    max_prompt_length=2048 \
    max_seq_length=4096 \
    model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  \
    hybrid_engine=True \
    actor.optimizer_args.lr=1e-6 \
    actor.fsdp.use_remove_padding=True \
    mini_batch_size=16 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=24000 \
    loss=ppo \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.param_offload=False \
    actor.optimizer_offload=False \
    sampler.tensor_model_parallel_size=1 \
    sampler.name=vllm \
    decoding.temperature=0.6 \
    decoding.n=4 \
    sampler.gpu_memory_utilization=0.85 \
    sampler.enforce_eager=False \
    ref.param_offload=True \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=1 \
    validation.decoding.temperature=0.6 \
    validation.decoding.top_p=0.95 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='math-agent' \
    experiment_name='gsm8k' \
    num_gpus_per_node=8 \
    num_nodes=1 \
    save_steps=10 \
    total_epochs=1000 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:MathEnvironment \
    +program.env_args.max_turns=1 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:MathAgent
