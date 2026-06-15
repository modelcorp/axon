set -x

AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

model_path="${1:-Qwen/Qwen3-8B}"

python3 -m axon.driver.train_agent_ppo \
    mode=eval \
    model_path=${model_path} \
    train_files=${AXON_DIR}/data/frozenlake/test.parquet \
    val_files=${AXON_DIR}/data/frozenlake/test.parquet \
    validation.decoding.n=4 \
    validation.decoding.temperature=0.7 \
    validation.decoding.top_p=0.8 \
    max_prompt_length=2048 \
    max_seq_length=4096 \
    max_steps=10 \
    sampler.name=vllm \
    sampler.load_format=auto \
    sampler.gpu_memory_utilization=0.95 \
    sampler.tensor_model_parallel_size=8 \
    sampler.pipeline_model_parallel_size=1 \
    sampler.data_parallel_size=1 \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:FrozenLakeEnv \
    +program.env_args.max_steps=8 \
    +program.agent_name=${SCRIPT_DIR}/agent.py:FrozenLakeAgent \
    +program.agent_args.use_multistep_prompt=True \
    +program.accumulate_history=True \
    +program.accumulate_thinking=True \
    +engine_args.disable_thinking=True \
    logger=['console'] \
    project_name=frozenlake-eval \
    experiment_name=eval
