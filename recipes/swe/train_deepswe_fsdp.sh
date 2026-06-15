set -x

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


python3 -m axon.driver.train_agent_ppo \
    advantage=loop \
    train_files=${AXON_DIR}/data/swe/R2E_Gym_Subset.parquet \
    val_files=${AXON_DIR}/data/swe/SWE_Bench_Verified.parquet \
    train_batch_size=8 \
    max_prompt_length=4096 \
    max_seq_length=36864 \
    model_path=Qwen/Qwen3-32B \
    hybrid_engine=True \
    actor.optimizer_args.lr=1e-6 \
    actor.fsdp.use_remove_padding=True \
    mini_batch_size=8 \
    loss=ppo \
    loss_args.token_reduce=mean-norm \
    loss_args.batch_reduce=step-mean \
    loss_args.clip_ratio_high=0.28 \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0.0 \
    actor.use_dynamic_bsz=False \
    actor.micro_batch_size_per_gpu=1 \
    actor.forward_use_dynamic_bsz=True \
    actor.forward_micro_batch_size_per_gpu=1 \
    actor.max_token_len_per_gpu=32000 \
    actor.fsdp.ulysses_sequence_parallel_size=8 \
    actor.micro_batch_size_per_gpu=1 \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.tensor_model_parallel_size=8 \
    sampler.name=vllm \
    sampler.enforce_eager=False \
    decoding.temperature=1.0 \
    decoding.n=8 \
    sampler.gpu_memory_utilization=0.6 \
    ref.param_offload=True \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=1 \
    validation.decoding.temperature=0 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='deepswe-agent' \
    experiment_name='swe-agent-rl' \
    num_gpus_per_node=8 \
    num_nodes=8 \
    save_steps=10 \
    total_epochs=1000 \
    overlong_filter=True \
    program_timeout=5400 \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:SWEEnv \
    +program.env_args.max_turns=50 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:SWEAgent \
    +program.accumulate_history=True \
    +program.accumulate_thinking=True