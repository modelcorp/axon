#!/usr/bin/env python3
"""Prepare eval datasets. Each row's env_name doubles as the reward function name.

Usage:
    python recipes/eval/data.py --tasks math gpqa f1 ifbench
    python recipes/eval/data.py --tasks all
    python recipes/eval/data.py --tasks math --output-dir data/eval
"""

import argparse
import os
import string

import pandas as pd

import axon

OUT = os.path.join(os.path.dirname(os.path.dirname(axon.__file__)), "data", "eval")


def _save(rows, split, name):
    if not rows:
        return print(f"  [skip] {name}: 0 rows")
    dest = os.path.join(OUT, split)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f"{name}.parquet")
    pd.DataFrame(rows).to_parquet(path)
    print(f"  {name}: {len(rows)} -> {path}")


def _row(env_name, question, answer, **extra):
    return {"env_name": env_name, "question": question, "answer": answer, **extra}


# ---- tasks ----


def gpqa():
    from datasets import load_dataset as hf_load

    ds = hf_load("Idavidrein/gpqa", "gpqa_diamond", trust_remote_code=True)["train"]
    rows = []
    for e in ds:
        choices = [e.get("Correct Answer", "")] + [e.get(f"Incorrect Answer {i}", "") for i in range(1, 4)]
        labeled = "\n".join(f"{string.ascii_uppercase[i]}) {c}" for i, c in enumerate(choices))
        rows.append(_row("gpqa", f"{e['Question']}\n\n{labeled}", "A", choices=choices, correct_letter="A"))
    _save(rows, "test", "gpqa_diamond")


def f1():
    from datasets import load_dataset as hf_load

    ds = hf_load("google-research-datasets/nq_open", trust_remote_code=True)["validation"]
    _save([_row("f1", e["question"], e["answer"]) for e in ds], "test", "nq_open")


def ifbench():
    from datasets import load_dataset as hf_load

    ds = hf_load("google/IFEval", trust_remote_code=True)["train"]
    _save(
        [
            _row(
                "ifbench",
                e["prompt"],
                "",
                instruction_id_list=e.get("instruction_id_list", []),
                prompt_text=e["prompt"],
                kwargs=e.get("kwargs", []),
            )
            for e in ds
        ],
        "test",
        "ifbench",
    )


TASKS = {"gpqa": gpqa, "f1": f1, "ifbench": ifbench}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["all"], choices=list(TASKS) + ["all"])
    p.add_argument("--output-dir", default=OUT)
    args = p.parse_args()
    OUT = args.output_dir
    for t in TASKS if "all" in args.tasks else {k: TASKS[k] for k in args.tasks}:
        print(f"[{t}]")
        TASKS[t]()
