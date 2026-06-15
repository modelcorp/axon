# Eval

Eval-only data preparation. Builds parquet files for standard benchmarks (GPQA, IFBench, F1-style QA), each row tagged with an `env_name` matching a reward in `axon.utils.rewards`.

```bash
python recipes/eval/data.py --tasks <gpqa|ifbench|f1|all> --output-dir <path>
```

Point any recipe's `val_files` at the produced parquet and set `mode: eval` (or `validation.steps` during training).
