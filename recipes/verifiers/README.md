# Verifiers

Integration with the [Verifiers Environments Hub](https://github.com/PrimeIntellect-ai/verifiers) — install any community environment and train against it. The shipped example is `wordle`; any environment from the [Prime Intellect hub](https://app.primeintellect.ai/dashboard/environments) plugs in the same way.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv tool install prime
pip install verifiers
prime env install will/wordle            # or will/wordle@0.1.3 to pin a version
```

Verify: `python -c "import verifiers as vf; print(len(vf.load_environment('wordle').dataset))"`

## Prepare data

```bash
cd recipes/verifiers
python prepare_verifiers_data.py --env-module wordle
# options: --local-dir DIR · --embed-task (self-contained rows, no dataset lookup at train time) · --max-examples N --shuffle · --info-only
```

Each parquet row is a flat dict that `VerifiersProgram` reads directly — `env_module` + `task_idx` in the default reference mode, or an embedded `task` with `--embed-task`.

## Run

```bash
cd recipes/verifiers/wordle/
bash data.sh             # shortcut for the prepare step above
bash train_megatron.sh
```

The training command is a normal Axon run with three program overrides:

```bash
program.name=verifiers +program.env_module=wordle +program.sampling_params={}
```

## Add a new environment

```bash
prime env install <owner>/<env-name>
python recipes/verifiers/prepare_verifiers_data.py --env-module <env-name>
mkdir recipes/verifiers/<env-name>       # mirror wordle/: data.sh + train_megatron.sh
```
