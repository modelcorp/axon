# NeMo Gym

Integration with [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) environments — verifiable-reward tasks for tool-use, coding, math, and instruction-following. The default is **Workplace Assistant**: a multi-step agent over a suite of tools spanning Email, Calendar, Analytics, Project Management, and CRM, graded by state-matching — the final database state is compared against ground truth, so any valid tool path counts.

## Two modes

| | **builtin** (default) | **native** |
|---|---|---|
| Agent loop | Axon (`NemoGymProgram`) | NeMo Gym's agent server |
| Servers | head + resource | head + agent + resource |
| Axon HTTP endpoint | not needed | **required** (`engine_endpoint.enable=true`) |
| Training context | full conversation (token-level) | per-step (agent server mediates each turn) |

## Setup

**1. Install NeMo Gym (one-time):**

```bash
cd $HOME && git clone https://github.com/NVIDIA-NeMo/Gym.git && cd Gym
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev
```

**2. Dataset + resource server** (in the Gym venv):

```bash
# env.yaml: hf_token, policy_base_url: http://localhost:9999/v1, policy_api_key: dummy, policy_model_name: dummy
config="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml"
ng_prepare_data "+config_paths=[${config}]" +output_dirpath=data/workplace_assistant \
    +mode=train_preparation +should_download=true +data_source=huggingface
ng_run "+config_paths=[${config}]"     # start the resource server; verify: curl http://localhost:11000/openapi.json
```

**3. Prepare Axon data (new terminal, Axon env):**

```bash
python recipes/nemo_gym/prepare_nemo_gym_data.py \
    --input $HOME/Gym/data/workplace_assistant/train.jsonl \
    --test-input $HOME/Gym/data/workplace_assistant/validation.jsonl \
    --resource-server workplace_assistant
```

## Run

**Builtin:**

```bash
cd recipes/nemo_gym/workplace_assistant
./train_megatron.sh
```

**Native** — enable Axon's endpoint and point NeMo Gym at it (`policy_base_url: http://AXON_HOST:8000/v1` in `env.yaml`):

```bash
python3 -m axon.driver.train_agent_ppo \
    engine_endpoint.enable=true engine_endpoint.port=8000 \
    program.name=nemo_gym +program.mode=native \
    +program.resource_server_name=workplace_assistant +program.agent_server_name=simple_agent
```

Axon autodiscovers the resource server's port from the head server (cached after first use). In native mode, `NemoGymProgram` tags each model call with `user="axon:{session_id}"` so the OpenAI-compatible endpoint routes generation to the right sampler.

## Key parameters

`program.name=nemo_gym` (required); `+program.resource_server_name` (e.g. `workplace_assistant`); `+program.max_steps` (default `10`); `+program.tool_call_format` (match the model). Native adds `+program.mode=native`, `+program.agent_server_name` (default `simple_agent`), `engine_endpoint.enable=true`. Pass `+program.resource_server_url` / `+program.agent_server_url` to skip autodiscovery.

## Data-prep options

`prepare_nemo_gym_data.py` also supports `--max-examples N --shuffle`, `--hf-repo … --hf-train-split … --hf-test-split …` (straight from HuggingFace), and `--info-only`. Default output: `<axon>/data/nemo_gym/{resource_server}/train.parquet`.

## Add a new environment

In NeMo Gym: `ng_init_resources_server +entrypoint=resources_servers/my_env`, implement the tools and `verify()`, then `ng_run`. In Axon: `prepare_nemo_gym_data.py --resource-server my_env`, then train with `program.name=nemo_gym +program.resource_server_name=my_env`.
