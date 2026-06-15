# Code

Competitive-programming agent (single-turn by default; raise `max_turns` for revise loops). The agent writes a Python solution and the environment runs it against unit tests. Reward is all-or-nothing — full reward only if every test passes, otherwise zero (`reward_bonus_coeff` shapes it).

## Run

```bash
cd recipes/code/
python data.py
huggingface-cli download <your-base-model>
./train_deepcoder_fsdp.sh
```

## Code execution

`code_reward_fn` (`axon/utils/rewards/code_reward.py`) runs the submitted code against the problem's unit tests, dispatching by dataset to the internal runners in `axon/utils/rewards/code_utils/` (TACO, LiveCodeBench, HumanEval+, KodCode). These are multiprocessing-based and process-isolated — not a security sandbox; the `leetcode` path additionally runs under firejail.

(The E2B / `python_interpreter` / `lcb_tool` tools under `recipes/tools/code_tools/` are a separate tool-calling system used by the `tools` recipe — they are *not* invoked by `code_reward_fn`.)

## Files

| File | Purpose |
|---|---|
| `agent.py` | `CompetitionCodingAgent` — formats the problem prompt (code-block extraction happens in the reward fn) |
| `env.py` | `CompetitionCodingEnv` — runs code, computes reward via `code_reward_fn` |
| `data.py` | Competitive-programming dataset prep |

## Algorithm

PPO loss + RLOO (`loop`) advantage, asymmetric clip (`0.2` / `0.28`), `token_reduce: mean`, `batch_reduce: step-mean`, FSDP, one node, `max_turns: 1` by default. Pass `loss=cispo` / `advantage=grpo` on the script's pass-through to vary it.

## Customize

- **Execution backend** — `code_reward_fn` uses the internal `code_utils` runners (process-isolated; firejail on the leetcode path). For OS-level isolation of untrusted code, wire in the E2B tool from `recipes/tools/code_tools/`.
- **Multi-turn** — raise `max_turns` for revise loops.
- **Reward** — `reward_bonus_coeff` for shaping.
