set -x

# Address of a running WebArena gateway (https://github.com/web-arena-x/webarena).
# Stand up your own gateway and export WEBARENA_URL before running this script.
: "${WEBARENA_URL:?Set WEBARENA_URL to your WebArena gateway, e.g. ws://host:port/send_and_wait}"
# export NCCL_P2P_DISABLE=1
# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m axon.driver.train_agent_ppo \
    advantage=loop \
    train_files=${AXON_DIR}/data/webarena/train.parquet \
    val_files=${AXON_DIR}/data/webarena/test.parquet \
    train_batch_size=4 \
    max_prompt_length=20480 \
    max_seq_length=32768 \
    model_path=Qwen/Qwen3-0.6B \
    hybrid_engine=True \
    actor.optimizer_args.lr=1e-6 \
    actor.fsdp.use_remove_padding=True \
    mini_batch_size=1 \
    loss=ppo \
    loss_args.token_reduce=mean-norm \
    loss_args.batch_reduce=step-mean \
    loss_args.clip_ratio_high=0.28 \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=48000 \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.tensor_model_parallel_size=1 \
    sampler.name=vllm \
    sampler.enforce_eager=False \
    decoding.temperature=0.7 \
    decoding.n=4 \
    sampler.gpu_memory_utilization=0.65 \
    ref.param_offload=True \
    ref.forward_micro_batch_size_per_gpu=1 \
    actor.forward_micro_batch_size_per_gpu=1 \
    validation.before_train=False \
    validation.steps=10 \
    validation.decoding.n=1 \
    validation.decoding.temperature=0.7 \
    validation.decoding.top_p=0.8 \
    validation.decoding.top_k=20 \
    kl_reward_args.kl_coef=0.001 \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='axon-agent' \
    experiment_name='4b-webarena_stepwise' \
    num_gpus_per_node=2 \
    num_nodes=1 \
    save_steps=400 \
    total_epochs=100 \
    engine_endpoint.enable=True \
    use_sampler_logprobs=False \
    +engine_args.disable_thinking=False \
    partial_rollout.enable=False \
    partial_rollout.n_iters=2 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:BrowserGymEnv \
    +program.env_args.subtask=webarena \
    +program.env_args.url=${WEBARENA_URL} \
    +program.env_args.max_turns=3 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:WebArenaAgent \
    +program.accumulate_history=True \
    +program.accumulate_thinking=True