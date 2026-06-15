# Formal Math

Theorem-proving recipe. The reward function automatically spawns a Kimina Lean server in Docker (image `projectnumina/kimina-lean-server:2.0.0`, overridable via `AXON_KIMINA_IMAGE`) to verify proofs — no manual prover server setup needed.

```bash
cd recipes/formal_math/
python data.py
./train_fsdp.sh
```
