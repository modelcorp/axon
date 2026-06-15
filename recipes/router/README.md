# Router

Multi-turn agent that learns *when to call an expert tool*. Reuses math data. `RouterEnv` overrides `MultiTurnEnvironment.step` directly — a useful pattern reference for fully custom transition logic.

```bash
python recipes/math/data.py     # reuses math data
cd recipes/router/
./train_fsdp.sh
```
