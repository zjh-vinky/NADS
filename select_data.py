import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel

    PEFT_AVAILABLE = True
except Exception:
    PeftConfig = None
    PeftModel = None
    PEFT_AVAILABLE = False

from nads.data import (
    TokenizingCausalLMCollator,
    build_raw_text_dataset,
    load_instruction_dataset,
    save_subset_as_jsonl,
)
from nads.selection import (
    SelectionResult,
    compute_utilities_and_embeddings_sparse,
    iterative_hard_thresholding_select,
    random_project_and_whiten,
    run_nads_selection,
)
from nads.training import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NADS data selection only (Stage2)")

    parser.add_argument("--output_dir", type=str, default="outputs/nads_select_run")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--navigator_ckpt", type=str, required=True, help="Path to already-trained navigator model")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--candidate_data_path", type=str, required=True)
    parser.add_argument("--candidate_dataset_split", type=str, default="train")
    parser.add_argument("--candidate_instruction_field", type=str, default="instruction")
    parser.add_argument("--candidate_response_field", type=str, default="response")
    parser.add_argument("--candidate_prompt_style", type=str, default="auto", choices=["auto", "default", "code"])
    parser.add_argument("--max_candidate_samples", type=int, default=None)

    parser.add_argument("--max_seq_length", type=int, default=1024)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="HF attention backend. Use auto to select based on runtime device.",
    )
    parser.add_argument(
        "--disable_attn_fallback",
        action="store_true",
        help="Disable fallback when requested attention backend is unavailable.",
    )
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument(
        "--disable_length_bucketing",
        action="store_true",
        help="Disable length-based ordering for candidate batches.",
    )
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")

    parser.add_argument("--stage2_batch_size", type=int, default=8)
    parser.add_argument("--nads_gamma", type=float, default=0.7)
    parser.add_argument("--nads_beta", type=float, default=1)
    parser.add_argument("--nads_nu", type=float, default=3e-3)
    parser.add_argument("--nads_iht_eta", type=float, default=0.02)
    parser.add_argument("--nads_rho", type=float, default=1e-4)
    parser.add_argument("--nads_proj_dim", type=int, default=256)
    parser.add_argument("--nads_select_k", type=int, default=20000)
    parser.add_argument("--nads_iht_steps", type=int, default=50)
    parser.add_argument(
        "--kl_vocab_chunk_size",
        type=int,
        default=0,
        help=">0 enables exact KL/entropy computation in vocab chunks to reduce peak memory.",
    )

    parser.add_argument("--save_selected_jsonl", action="store_true")

    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def _save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _dedup_keep_order(items: Tuple[Optional[str], ...]) -> Tuple[Optional[str], ...]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _resolve_requested_attn_impl(args: argparse.Namespace, device: torch.device) -> Optional[str]:
    if args.attn_implementation == "auto":
        return "flash_attention_2" if device.type == "cuda" else "eager"
    return args.attn_implementation


def _ensure_peft_available() -> None:
    if not PEFT_AVAILABLE:
        raise ImportError("LoRA adapter checkpoint requires `peft`. Please `pip install peft`.")


def _is_peft_adapter_checkpoint(path_or_name: str) -> bool:
    p = Path(path_or_name)
    return p.is_dir() and (p / "adapter_config.json").exists()


def _load_base_model(
    model_name_or_path: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[AutoModelForCausalLM, Optional[str]]:
    requested_impl = _resolve_requested_attn_impl(args, device)
    fallback_order = (
        (requested_impl, "sdpa", "eager", None)
        if device.type == "cuda"
        else (requested_impl, "eager", None)
    )
    candidates = (
        (requested_impl,)
        if args.disable_attn_fallback
        else _dedup_keep_order(tuple(x for x in fallback_order))
    )

    last_error: Optional[Exception] = None
    for attn_impl in candidates:
        load_kwargs = dict(
            torch_dtype=_torch_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **load_kwargs,
            )
            model.to(device)
            if attn_impl != requested_impl:
                print(
                    f"[Warn] Failed to use attn_implementation='{requested_impl}' for {model_name_or_path}. "
                    f"Fell back to '{attn_impl or 'default'}'."
                )
            return model, attn_impl
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load model: {model_name_or_path}")


def _load_model(
    model_name_or_path: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[str], bool]:
    if _is_peft_adapter_checkpoint(model_name_or_path):
        _ensure_peft_available()
        peft_cfg = PeftConfig.from_pretrained(model_name_or_path)
        base_model, attn_impl = _load_base_model(peft_cfg.base_model_name_or_path, args, device)
        model = PeftModel.from_pretrained(base_model, model_name_or_path, is_trainable=False)
        model.to(device)
        return model, attn_impl, True

    model, attn_impl = _load_base_model(model_name_or_path, args, device)
    return model, attn_impl, False


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _init_distributed_and_device(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available, but WORLD_SIZE > 1.")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    return is_distributed, rank, local_rank, world_size, device


def _finalize_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _compute_approx_lengths(
    raw_dataset,
    instruction_field: str,
    response_field: str,
) -> np.ndarray:
    """
    Approximate sequence lengths for bucketing using character counts.
    This is lightweight and avoids full upfront tokenization.
    """
    n = len(raw_dataset)
    lengths = np.empty(n, dtype=np.int32)

    try:
        instructions = raw_dataset[instruction_field]
        responses = raw_dataset[response_field]
        for i, (instruction, response) in enumerate(zip(instructions, responses)):
            lengths[i] = len(str(instruction)) + len(str(response))
    except Exception:
        for i, row in enumerate(raw_dataset):
            lengths[i] = len(str(row[instruction_field])) + len(str(row[response_field]))

    return lengths


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    is_distributed, rank, local_rank, world_size, device = _init_distributed_and_device(args)
    is_main = _is_main_process(rank)

    use_amp = (not args.disable_amp) and args.dtype in {"bfloat16", "float16"}
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed and dist.is_initialized():
        dist.barrier()

    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        model_max_length=args.max_seq_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_main:
        print("[Select] Loading candidate dataset...")
    raw_candidate = load_instruction_dataset(
        path_or_name=args.candidate_data_path,
        split=args.candidate_dataset_split,
        max_samples=args.max_candidate_samples,
    )

    cand_full = build_raw_text_dataset(
        raw_dataset=raw_candidate,
        instruction_field=args.candidate_instruction_field,
        response_field=args.candidate_response_field,
        prompt_style=args.candidate_prompt_style,
        eos_token=tokenizer.eos_token,
    )
    length_bucketing_enabled = not args.disable_length_bucketing
    bucketing_seconds = 0.0

    if length_bucketing_enabled and len(cand_full) > 1:
        t0 = time.time()
        if is_main:
            print("[Select] Building length-bucketed candidate order...")
        approx_lengths = _compute_approx_lengths(
            raw_dataset=raw_candidate,
            instruction_field=args.candidate_instruction_field,
            response_field=args.candidate_response_field,
        )
        order = np.argsort(approx_lengths, kind="stable")
        cand_for_loader = Subset(cand_full, indices=order.tolist())
        bucketing_seconds = float(time.time() - t0)
        if is_main:
            print(f"[Select] Length bucketing ready in {bucketing_seconds:.2f}s.")
    else:
        cand_for_loader = cand_full

    if is_distributed:
        local_indices = range(rank, len(cand_for_loader), world_size)
        cand_for_loader = Subset(cand_for_loader, indices=local_indices)

    pin_memory = (device.type == "cuda") and (not args.no_pin_memory)
    dataloader_kwargs = dict(
        batch_size=args.stage2_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=TokenizingCausalLMCollator(tokenizer=tokenizer, max_seq_length=args.max_seq_length),
        pin_memory=pin_memory,
    )
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)
        dataloader_kwargs["persistent_workers"] = True

    cand_loader = DataLoader(
        cand_for_loader,
        **dataloader_kwargs,
    )

    if is_main:
        dist_msg = f" | distributed world_size={world_size}" if is_distributed else ""
        print(f"[Select] Loading pretrained and navigator models...{dist_msg}")
    pretrain_model, pretrain_attn_impl, pretrain_is_adapter = _load_model(args.model_name_or_path, args, device)
    navigator_model, navigator_attn_impl, navigator_is_adapter = _load_model(args.navigator_ckpt, args, device)
    if args.torch_compile and hasattr(torch, "compile"):
        pretrain_model = torch.compile(pretrain_model, mode=args.compile_mode)
        navigator_model = torch.compile(navigator_model, mode=args.compile_mode)

    selection: Optional[SelectionResult] = None
    if not is_distributed:
        if is_main:
            print("[Select] Running NADS selection...")
        selection = run_nads_selection(
            pretrain_model=pretrain_model,
            navigator_model=navigator_model,
            dataloader=cand_loader,
            device=device,
            gamma=args.nads_gamma,
            beta=args.nads_beta,
            nu=args.nads_nu,
            rho=args.nads_rho,
            proj_dim=args.nads_proj_dim,
            k=min(args.nads_select_k, len(cand_full)),
            eta=args.nads_iht_eta,
            steps=args.nads_iht_steps,
            seed=args.seed,
            vocab_chunk_size=args.kl_vocab_chunk_size,
            use_amp=use_amp,
        )
    else:
        local_sample_ids, local_utilities, local_embeddings = compute_utilities_and_embeddings_sparse(
            pretrain_model=pretrain_model,
            navigator_model=navigator_model,
            dataloader=cand_loader,
            device=device,
            gamma=args.nads_gamma,
            vocab_chunk_size=args.kl_vocab_chunk_size,
            use_amp=use_amp,
        )
        part_file = out_dir / f"_stage2_rank_{rank}.npz"
        np.savez_compressed(
            part_file,
            sample_ids=local_sample_ids,
            utilities=local_utilities,
            embeddings=local_embeddings,
        )

        if dist.is_initialized():
            dist.barrier()

        if is_main:
            print("[Select] Merging distributed utility shards...")
            dataset_size = len(cand_full)
            # hidden_size = pretrain_model.config.hidden_size
            text_cfg = getattr(pretrain_model.config, "text_config", None)
            if isinstance(text_cfg, dict):
                hidden_size = text_cfg.get("hidden_size", None)
            else:
                hidden_size = getattr(text_cfg, "hidden_size", None)

            if hidden_size is None:
                hidden_size = getattr(pretrain_model.config, "hidden_size", None)

            if hidden_size is None:
                raise AttributeError(
                    f"Cannot infer hidden_size from config type={type(pretrain_model.config).__name__}"
                )
            utilities = np.zeros(dataset_size, dtype=np.float64)
            embeddings = np.zeros((dataset_size, hidden_size), dtype=np.float32)
            for r in range(world_size):
                shard = np.load(out_dir / f"_stage2_rank_{r}.npz")
                ids = shard["sample_ids"].astype(np.int64)
                if ids.size > 0:
                    utilities[ids] = shard["utilities"].astype(np.float64)
                    embeddings[ids] = shard["embeddings"].astype(np.float32)
                shard.close()

            projected, _ = random_project_and_whiten(
                embeddings=embeddings,
                proj_dim=args.nads_proj_dim,
                rho=args.nads_rho,
                seed=args.seed,
            )
            selected_idx, weights, history = iterative_hard_thresholding_select(
                utilities=utilities,
                projected_embeddings=projected,
                k=min(args.nads_select_k, len(cand_full)),
                beta=args.nads_beta,
                nu=args.nads_nu,
                eta=args.nads_iht_eta,
                steps=args.nads_iht_steps,
            )
            selection = SelectionResult(
                selected_indices=selected_idx,
                utilities=utilities,
                projected_embeddings=projected,
                weights=weights,
                objective_history=history,
                embeddings=embeddings,
            )

            for r in range(world_size):
                shard_path = out_dir / f"_stage2_rank_{r}.npz"
                if shard_path.exists():
                    shard_path.unlink()

        if dist.is_initialized():
            dist.barrier()

    if not is_main:
        del pretrain_model
        del navigator_model
        _cleanup_memory()
        _finalize_distributed(is_distributed)
        return

    if selection is None:
        raise RuntimeError("Selection result is missing on main process.")

    np.save(out_dir / "utilities.npy", selection.utilities)
    np.save(out_dir / "projected_embeddings.npy", selection.projected_embeddings)
    np.save(out_dir / "weights.npy", selection.weights)
    if hasattr(selection, 'raw_embeddings') and selection.raw_embeddings is not None:
        np.save(out_dir / "last_hidden_embeddings.npy", selection.raw_embeddings)
    elif hasattr(selection, 'embeddings') and selection.embeddings is not None:
        np.save(out_dir / "last_hidden_embeddings.npy", selection.embeddings)
    else:
        print("[Warn] Failed to save last_hidden_embeddings.npy: SelectionResult does not contain raw embeddings.")

    selected_indices = selection.selected_indices.tolist()
    _save_json(
        out_dir / "selected_indices.json",
        {
            "selected_indices": selected_indices,
            "objective_history": selection.objective_history,
        },
    )

    if args.save_selected_jsonl:
        save_subset_as_jsonl(
            raw_dataset=raw_candidate,
            indices=selected_indices,
            output_path=str(out_dir / "selected_constraint_set.jsonl"),
            instruction_field=args.candidate_instruction_field,
            response_field=args.candidate_response_field,
        )

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "navigator_ckpt": str(Path(args.navigator_ckpt).resolve()),
        "attn_implementation_requested": args.attn_implementation,
        "pretrain_attn_implementation_used": pretrain_attn_impl,
        "navigator_attn_implementation_used": navigator_attn_impl,
        "pretrain_is_adapter": pretrain_is_adapter,
        "navigator_is_adapter": navigator_is_adapter,
        "candidate_data_path": args.candidate_data_path,
        "candidate_size": len(raw_candidate),
        "selected_constraint_size": len(selected_indices),
        "selected_indices_file": str((out_dir / "selected_indices.json").resolve()),
        "stage2_batch_size": args.stage2_batch_size,
        "nads_gamma": args.nads_gamma,
        "nads_beta": args.nads_beta,
        "nads_nu": args.nads_nu,
        "nads_rho": args.nads_rho,
        "nads_proj_dim": args.nads_proj_dim,
        "nads_select_k": args.nads_select_k,
        "nads_iht_eta": args.nads_iht_eta,
        "nads_iht_steps": args.nads_iht_steps,
        "kl_vocab_chunk_size": args.kl_vocab_chunk_size,
        "length_bucketing": length_bucketing_enabled,
        "length_bucketing_seconds": bucketing_seconds,
        "distributed": is_distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "torch_compile": args.torch_compile,
    }
    _save_json(out_dir / "selection_summary.json", summary)

    del pretrain_model
    del navigator_model
    _cleanup_memory()
    _finalize_distributed(is_distributed)

    print("[Done] Data selection completed.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
