"""Worker script for mbridge integration tests. Launched via torchrun by pytest.

Initializes torch.distributed + megatron parallel state, creates a bridge,
loads model weights, runs a forward pass, and writes results to a JSON file.

Supports multi-GPU parallelism (TP, PP, EP).

Usage (not called directly — invoked by test_gpu_mbridge.py):
    torchrun --nproc_per_node=1 tests/models/_gpu_mbridge_worker.py \
        --model-id THUDM/GLM-4-9B-0414 --output /tmp/result.json

    torchrun --nproc_per_node=8 tests/models/_gpu_mbridge_worker.py \
        --model-id Qwen/Qwen3-Next-80B-A3B-Instruct --tp 8 --output /tmp/result.json
"""

import argparse
import datetime
import json
import os
import sys
import traceback

# Prevent tests/models/mbridge/ from shadowing the real mbridge pip package.
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _this_dir]

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def _export_compare_to_hf(bridge, models, *, is_rank0):
    """Stream ``bridge.export_weights`` and compare every yielded tensor to HF.

    ``bridge.load_weights(...)`` + ``bridge.export_weights(...)`` is a pure
    reshape/gather/concat round-trip — there is no compute that can lose
    precision, so every exported tensor MUST be bit-exact equal to the HF
    safetensor it originated from. Any diff, shape mismatch, missing name or
    orphan name is a bug in the bridge's sharding or name mapping.

    All ranks drain the generator (bridge.export_weights does collectives on
    every rank), but only rank 0 opens safetensors and records diffs. Names
    and tensors not under the text-model prefix are ignored on the HF side so
    VL-only checkpoints don't produce spurious "missing" entries for vision
    subtrees the text bridge doesn't export.
    """
    st_io = bridge.safetensor_io  # populated by bridge.load_weights()
    # e.g. "model.language_model." for Gemma4ForConditionalGeneration, "model." for CausalLM
    hf_prefix = bridge._hf_prefix() + "." if hasattr(bridge, "_hf_prefix") else "model."

    compared = []
    shape_mismatches = []
    exported_not_in_hf = []
    hf_names_seen = set()
    hf_all_names = set(st_io.load_hf_weight_names()) if is_rank0 else set()

    # NOTE: don't `del` the loop variable in early-continue branches — pyflakes
    # (ruff F821) doesn't track `continue` after `del` and will flag every
    # subsequent use as undefined. Python rebinds the loop variable each
    # iteration, so letting the reference drop at loop rebind is enough for
    # memory hygiene.
    for name, mc_tensor in bridge.export_weights(models):
        if not is_rank0:
            continue
        if name not in hf_all_names:
            exported_not_in_hf.append(name)
            continue
        hf_names_seen.add(name)
        try:
            hf_t = st_io.load_one_hf_weight(name)
        except Exception as e:
            shape_mismatches.append({"name": name, "issue": f"load_failed: {type(e).__name__}: {e}"})
            continue
        mc_t = mc_tensor.detach().to("cpu", copy=True)
        if tuple(hf_t.shape) != tuple(mc_t.shape):
            shape_mismatches.append({"name": name, "hf_shape": list(hf_t.shape), "mc_shape": list(mc_t.shape)})
            del hf_t, mc_t
            continue
        exact = bool(torch.equal(hf_t, mc_t))
        if exact:
            max_abs = 0.0
            mean_abs = 0.0
        else:
            diff = (hf_t.float() - mc_t.float()).abs()
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            del diff
        compared.append(
            {
                "name": name,
                "shape": list(mc_t.shape),
                "dtype": str(mc_t.dtype),
                "hf_dtype": str(hf_t.dtype),
                "exact": exact,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
            }
        )
        del hf_t, mc_t

    if not is_rank0:
        return None

    # Only flag HF names the bridge was supposed to export (text + lm_head);
    # vision/multimodal subtrees are intentionally not exported by text bridges.
    text_hf_names = {n for n in hf_all_names if n.startswith(hf_prefix) or n == "lm_head.weight"}
    text_hf_not_exported = sorted(text_hf_names - hf_names_seen)

    return {
        "hf_prefix": hf_prefix,
        "n_hf_total": len(hf_all_names),
        "n_hf_text": len(text_hf_names),
        "n_exported_matched": len(compared),
        "n_exact_match": sum(1 for c in compared if c["exact"]),
        "n_value_diff": sum(1 for c in compared if not c["exact"]),
        "n_shape_mismatch": len(shape_mismatches),
        "n_exported_not_in_hf": len(exported_not_in_hf),
        "n_hf_text_not_exported": len(text_hf_not_exported),
        "max_max_abs": max((c["max_abs"] for c in compared), default=0.0),
        "max_mean_abs": max((c["mean_abs"] for c in compared), default=0.0),
        "worst_by_max_abs": sorted(compared, key=lambda c: -c["max_abs"])[:15],
        "shape_mismatches": shape_mismatches[:15],
        "exported_not_in_hf_sample": exported_not_in_hf[:20],
        "hf_text_not_exported_sample": text_hf_not_exported[:20],
    }


def _patch_mbridge_transformer_layers():
    """Patch mbridge TransformerLayer subclasses for megatron-core compatibility.

    Newer megatron-core passes extra kwargs (pg_collection, etc.) to TransformerLayer.__init__
    that older mbridge subclasses don't accept. This adds **kwargs to their __init__ signatures.
    """
    import importlib
    import inspect

    from megatron.core.transformer.transformer_layer import TransformerLayer

    patches = [
        ("mbridge.models.gemma3.transformer_layer", "Gemma3TransformerLayer"),
    ]

    for mod_path, cls_name in patches:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            if not issubclass(cls, TransformerLayer):
                continue
            orig_init = cls.__init__
            sig = inspect.signature(orig_init)
            if "kwargs" in sig.parameters:
                continue  # already accepts **kwargs

            def _make_patched_init(original):
                def patched_init(self, *args, **kwargs):
                    valid = inspect.signature(original).parameters
                    filtered = {k: v for k, v in kwargs.items() if k in valid}
                    return original(self, *args, **filtered)

                return patched_init

            cls.__init__ = _make_patched_init(orig_init)
        except (ImportError, AttributeError):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument("--vpp", type=int, default=None, help="Virtual pipeline parallel size (None disables VPP).")
    parser.add_argument(
        "--generate-tokens",
        type=int,
        default=0,
        help="If >0, run greedy generation for this many new tokens and "
        "record the decoded text in results['checks']['generated_text']. "
        "Uses a chat-templated prompt when the tokenizer supports it.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="User prompt for generation. Ignored unless --generate-tokens > 0.",
    )
    args = parser.parse_args()

    result = {"model_id": args.model_id, "passed": False, "checks": {}, "error": None}
    result["checks"]["tp_size"] = args.tp
    result["checks"]["pp_size"] = args.pp

    try:
        # 1. Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(seconds=600),
            )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # 2. Initialize megatron parallel state
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=args.tp,
                pipeline_model_parallel_size=args.pp,
                virtual_pipeline_model_parallel_size=args.vpp,
                context_parallel_size=args.cp,
                expert_model_parallel_size=args.ep,
                expert_tensor_parallel_size=args.etp,
            )

        # 3. Initialize megatron CUDA RNG
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(42)

        # 4. Load HF config and create bridge
        from mbridge import AutoBridge
        from transformers import AutoConfig, AutoTokenizer

        # Register axon's custom model types
        import axon.models.mbridge  # noqa: F401

        # Patch mbridge TransformerLayer subclasses for megatron-core compat
        # (newer megatron passes pg_collection kwarg that older mbridge doesn't accept)
        _patch_mbridge_transformer_layers()

        hf_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False)
        result["checks"]["config_loaded"] = True
        result["checks"]["model_type"] = hf_config.model_type
        # Some models (e.g. Gemma3 VL) store these in text_config
        _tcfg = getattr(hf_config, "text_config", hf_config)
        result["checks"]["num_hidden_layers"] = _tcfg.num_hidden_layers
        result["checks"]["vocab_size"] = _tcfg.vocab_size

        bridge = AutoBridge.from_config(hf_config, dtype=torch.bfloat16)
        tf_config = bridge.config
        tf_config.bf16 = True
        tf_config.fp16 = False
        result["checks"]["bridge_created"] = True
        result["checks"]["bridge_class"] = type(bridge).__name__

        # 5. Create model via bridge
        import inspect

        get_model_params = inspect.signature(bridge.get_model).parameters
        get_model_kwargs = {"wrap_with_ddp": False}
        if "bf16" in get_model_params:
            get_model_kwargs["bf16"] = True
        models = bridge.get_model(**get_model_kwargs)
        if not isinstance(models, list):
            models = [models]
        result["checks"]["model_created"] = True
        result["checks"]["num_model_chunks"] = len(models)

        # Count parameters (per-rank, sharded across TP/PP)
        total_params = sum(p.numel() for m in models for p in m.parameters())
        result["checks"]["total_params_per_rank"] = total_params

        # 6. Load weights
        bridge.load_weights(models, args.model_id)
        result["checks"]["weights_loaded"] = True

        # 6b. Optionally exercise the trainer→sampler weight-export path that
        # `bridge.export_weights` drives during hybrid-engine sync. Useful for
        # catching TP/EP/ETP merge bugs that only surface when converting
        # sharded megatron weights back to HF layout.
        if os.environ.get("MBRIDGE_TEST_EXPORT", "0") == "1":
            try:
                count = 0
                shapes = {}
                for name, tensor in bridge.export_weights(models):
                    count += 1
                    if count <= 3:
                        shapes[name] = list(tensor.shape)
                result["checks"]["export_weights_count"] = count
                result["checks"]["export_weights_sample_shapes"] = shapes
            except Exception as _exp_err:
                result["checks"]["export_weights_error"] = f"{type(_exp_err).__name__}: {_exp_err}"
                raise

        # 6c. Bit-exact round-trip check: load via bridge → export via bridge →
        # compare each yielded tensor to its HF safetensor source. Because the
        # whole path is pure reshape/gather/concat, exported values MUST be
        # bit-exact equal to HF. Any divergence localizes the offending tensor.
        if os.environ.get("MBRIDGE_EXPORT_COMPARE", "0") == "1":
            is_rank0 = int(os.environ.get("RANK", 0)) == 0
            compare = _export_compare_to_hf(bridge, models, is_rank0=is_rank0)
            if is_rank0 and compare is not None:
                result["checks"]["export_compare"] = compare

        # 7. Verify model structure
        model = models[0]
        from axon.utils.megatron.utils import unwrap_model

        raw_model = unwrap_model(model)

        has_decoder = hasattr(raw_model, "decoder")
        result["checks"]["has_decoder"] = has_decoder
        if has_decoder:
            n_layers = len(raw_model.decoder.layers)
            result["checks"]["decoder_layers_this_rank"] = n_layers

        # 8. Run forward pass
        from axon.models.mcore.forward.registry import get_mcore_forward_fn

        forward_fn = get_mcore_forward_fn(hf_config)
        result["checks"]["forward_fn_resolved"] = True

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prefer the model's chat template for instruction-tuned checkpoints —
        # otherwise a raw "The capital of France is" prompt drives a -it model
        # into weird continuations.
        if args.generate_tokens > 0 and getattr(tokenizer, "chat_template", None):
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": args.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = args.prompt
        else:
            # Default prompt for non-generation tests. If --prompt was supplied,
            # honour it (chat template is optional but nice to have for real models).
            if args.prompt and args.prompt != "What is the capital of France?":
                try:
                    text = (
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": args.prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        if getattr(tokenizer, "chat_template", None)
                        else args.prompt
                    )
                except Exception:
                    text = args.prompt
            else:
                text = "The capital of France is"
        encoded = tokenizer(text, return_tensors="pt", padding=False)
        seq_len = encoded["input_ids"].shape[1]
        input_ids = encoded["input_ids"].cuda()
        attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        result["checks"]["prompt_text"] = text
        result["checks"]["prompt_tokens"] = input_ids[0].tolist()

        for m in models:
            m.eval()
        with torch.no_grad():
            output = forward_fn(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                multi_modal_inputs={},
                logits_processor=None,
                value_model=False,
                data_format="thd",
            )

        result["checks"]["forward_pass_completed"] = True

        # 9. Check output (only meaningful on last PP stage)
        is_last_pp = mpu.is_pipeline_last_stage(ignore_virtual=True)
        result["checks"]["is_last_pp_stage"] = is_last_pp

        if is_last_pp and isinstance(output, torch.Tensor):
            result["checks"]["output_shape"] = list(output.shape)
            result["checks"]["output_has_nan"] = bool(torch.isnan(output).any())
            result["checks"]["output_has_inf"] = bool(torch.isinf(output).any())
            result["checks"]["output_dtype"] = str(output.dtype)
            if output.ndim == 3:
                result["checks"]["output_vocab_dim"] = output.shape[-1]
                # For multimodal configs (Gemma3-VL, Gemma4, etc.), vocab_size lives under text_config.
                _vocab = getattr(_tcfg, "vocab_size", None) or getattr(hf_config, "vocab_size", None)
                # Account for TP-sharded vocab.
                tp = mpu.get_tensor_model_parallel_world_size()
                result["checks"]["vocab_matches"] = _vocab is not None and (
                    output.shape[-1] == _vocab or output.shape[-1] * tp == _vocab
                )
                # Cheap sanity: top-1 next-token id for the last position,
                # gathered across TP so it's a real vocab id. Helps spot
                # obviously-broken logits (argmax degenerating to 0, BOS, etc.).
                try:
                    last = output[:, -1, :]
                    if tp > 1:
                        gathered = [torch.empty_like(last) for _ in range(tp)]
                        torch.distributed.all_gather(gathered, last, group=mpu.get_tensor_model_parallel_group())
                        full = torch.cat(gathered, dim=-1)
                    else:
                        full = last
                    full = full.float()[0]
                    vals, idxs = full.topk(5)
                    result["checks"]["top5"] = [
                        (int(i.item()), tokenizer.decode([int(i.item())]), float(v.item()))
                        for v, i in zip(vals, idxs, strict=False)
                    ]
                    result["checks"]["logit_max"] = float(full.max().item())
                    result["checks"]["logit_min"] = float(full.min().item())
                    result["checks"]["logit_mean"] = float(full.mean().item())
                    result["checks"]["logit_std"] = float(full.std().item())
                except Exception as _:
                    pass
        elif is_last_pp and isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    result["checks"][f"output_{k}_shape"] = list(v.shape)
                    result["checks"][f"output_{k}_has_nan"] = bool(torch.isnan(v).any())

        # 10. (Optional) greedy generation — the only real "does this produce
        # legible text" sanity check. Only runs with PP=1 since prefill across
        # PP stages would require scheduling the full pipeline each step.
        if args.generate_tokens > 0:
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            if pp_size > 1:
                result["checks"]["generation_skipped"] = "PP>1 not supported by this simple greedy loop"
            else:
                # Build a set of "stop" token ids: EOS plus common turn-end
                # tokens used by various chat-tuned models. Unknown names
                # resolve to unk_token_id and are filtered out below.
                stop_ids = set()
                if tokenizer.eos_token_id is not None:
                    stop_ids.add(int(tokenizer.eos_token_id))
                for name in ("<turn|>", "<|endoftext|>", "<end_of_turn>"):
                    try:
                        sid = tokenizer.convert_tokens_to_ids(name)
                        if isinstance(sid, int) and sid >= 0 and sid != tokenizer.unk_token_id:
                            stop_ids.add(int(sid))
                    except Exception:
                        pass
                # Also honour generation_config.eos_token_id if it's a list.
                cfg_eos = getattr(hf_config, "eos_token_id", None)
                if isinstance(cfg_eos, list):
                    stop_ids.update(int(x) for x in cfg_eos)
                elif isinstance(cfg_eos, int):
                    stop_ids.add(int(cfg_eos))
                gen_ids = input_ids.clone()
                generated = []
                for _ in range(args.generate_tokens):
                    cur_len = gen_ids.shape[1]
                    cur_attn = torch.ones(1, cur_len, dtype=torch.bool, device="cuda")
                    cur_pos = torch.arange(cur_len, device="cuda").unsqueeze(0)
                    with torch.no_grad():
                        step_out = forward_fn(
                            model=model,
                            input_ids=gen_ids,
                            attention_mask=cur_attn,
                            position_ids=cur_pos,
                            multi_modal_inputs={},
                            logits_processor=None,
                            value_model=False,
                            data_format="thd",
                        )
                    if not isinstance(step_out, torch.Tensor) or step_out.ndim != 3:
                        break
                    last = step_out[:, -1, :]
                    if tp_size > 1:
                        gathered = [torch.empty_like(last) for _ in range(tp_size)]
                        torch.distributed.all_gather(gathered, last, group=mpu.get_tensor_model_parallel_group())
                        full = torch.cat(gathered, dim=-1).float()[0]
                    else:
                        full = last.float()[0]
                    next_id = int(full.argmax().item())
                    generated.append(next_id)
                    if next_id in stop_ids:
                        break
                    gen_ids = torch.cat(
                        [gen_ids, torch.tensor([[next_id]], device="cuda", dtype=gen_ids.dtype)],
                        dim=1,
                    )
                result["checks"]["generated_ids"] = generated
                result["checks"]["generated_text"] = tokenizer.decode(generated, skip_special_tokens=False)
                result["checks"]["generated_text_clean"] = tokenizer.decode(generated, skip_special_tokens=True)

        result["passed"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    finally:
        # Only rank 0 writes results
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
