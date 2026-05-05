import argparse
import gc
import json
import math
import os
import shlex
import subprocess
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None
    PeftConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None
    PEFT_AVAILABLE = False

from nads.data import (
    CausalLMCollator,
    build_full_text_dataset,
    build_sft_dataset,
    load_instruction_dataset,
    split_train_val,
)
from nads.training import TrainConfig, set_global_seed, train_final_with_distillation, train_navigator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NADS training pipeline (Stage1 + Stage3 only)")

    # Core I/O
    parser.add_argument("--output_dir", type=str, default="outputs/nads_train_run")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")

    # D_new for training
    parser.add_argument("--new_data_path", type=str, required=True)
    parser.add_argument("--new_dataset_split", type=str, default="train")
    parser.add_argument("--new_instruction_field", type=str, default="instruction")
    parser.add_argument("--new_response_field", type=str, default="response")
    parser.add_argument("--new_prompt_style", type=str, default="auto", choices=["auto", "default", "code"])
    parser.add_argument("--max_new_samples", type=int, default=None)

    # Candidate dataset is only used in Stage3 with selected indices
    parser.add_argument("--candidate_data_path", type=str, default=None)
    parser.add_argument("--candidate_dataset_split", type=str, default="train")
    parser.add_argument("--candidate_instruction_field", type=str, default="instruction")
    parser.add_argument("--candidate_response_field", type=str, default="response")
    parser.add_argument("--candidate_prompt_style", type=str, default="auto", choices=["auto", "default", "code"])
    parser.add_argument("--max_candidate_samples", type=int, default=None)

    parser.add_argument("--selected_indices_path", type=str, default=None)

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--disable_length_bucketing", action="store_true")
    parser.add_argument("--length_bucket_window_mult", type=int, default=50)

    # Runtime
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
    parser.add_argument("--num_workers", type=int, default=0)

    # Stage switches
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage3", action="store_true")
    parser.add_argument("--navigator_ckpt", type=str, default=None)
    parser.add_argument("--stage1_backend", type=str, default="native", choices=["native", "llamafactory"])
    parser.add_argument("--stage3_backend", type=str, default="native", choices=["native", "llamafactory"])
    parser.add_argument("--stage1_finetune_type", type=str, default="full", choices=["full", "lora"])
    parser.add_argument("--stage3_finetune_type", type=str, default="full", choices=["full", "lora"])

    # LLaMA-Factory options (Stage1 only for strict algorithm parity)
    parser.add_argument("--llamafactory_cli", type=str, default="llamafactory-cli")
    parser.add_argument("--llamafactory_template", type=str, default="default")
    parser.add_argument("--llamafactory_cutoff_len", type=int, default=None)
    parser.add_argument(
        "--llamafactory_extra_args",
        type=str,
        default="",
        help="Extra args appended to `llamafactory-cli train` command.",
    )

    # LoRA options
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA.",
    )
    parser.add_argument(
        "--lora_modules_to_save",
        type=str,
        default=None,
        help="Comma-separated module names to keep trainable and save with LoRA adapters.",
    )
    parser.add_argument(
        "--lora_use_rslora",
        action="store_true",
        help="Enable rsLoRA scaling when finetune_type is lora.",
    )
    parser.add_argument(
        "--lora_merge_before_save",
        action="store_true",
        help="Merge LoRA adapter into base model before save (outputs full model weights).",
    )

    # Stage-1 (navigator)
    parser.add_argument("--stage1_batch_size", type=int, default=2)
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage1_lr", type=float, default=2e-5)
    parser.add_argument("--stage1_weight_decay", type=float, default=0.0)
    parser.add_argument("--stage1_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--stage1_grad_accum_steps", type=int, default=16)
    parser.add_argument("--stage1_early_stop_patience", type=int, default=3)
    parser.add_argument("--stage1_early_stop_delta", type=float, default=0.0)

    # Stage-3 (final distillation)
    parser.add_argument("--stage3_task_batch_size", type=int, default=2)
    parser.add_argument("--stage3_kd_batch_size", type=int, default=2)
    parser.add_argument("--stage3_epochs", type=int, default=2)
    parser.add_argument("--stage3_lr", type=float, default=2e-5)
    parser.add_argument("--stage3_weight_decay", type=float, default=0.0)
    parser.add_argument("--stage3_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--stage3_grad_accum_steps", type=int, default=16)
    parser.add_argument("--stage3_early_stop_patience", type=int, default=3)
    parser.add_argument("--stage3_early_stop_delta", type=float, default=0.0)
    parser.add_argument("--distill_lambda", type=float, default=1.0)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

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

def _parse_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items if items else None

def _quote_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)

def _ensure_peft_available() -> None:
    if not PEFT_AVAILABLE:
        raise ImportError("LoRA mode requires `peft`. Please `pip install peft`.")

def _is_peft_adapter_checkpoint(path_or_name: str) -> bool:
    p = Path(path_or_name)
    return p.is_dir() and (p / "adapter_config.json").exists()

def _resolve_requested_attn_impl(args: argparse.Namespace, device: torch.device) -> Optional[str]:
    if args.attn_implementation == "auto":
        return "flash_attention_2" if device.type == "cuda" else "eager"
    return args.attn_implementation

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
    allow_peft_adapter: bool = True,
) -> Tuple[torch.nn.Module, Optional[str], bool]:
    if allow_peft_adapter and _is_peft_adapter_checkpoint(model_name_or_path):
        _ensure_peft_available()
        peft_cfg = PeftConfig.from_pretrained(model_name_or_path)
        base_model, attn_impl = _load_base_model(peft_cfg.base_model_name_or_path, args, device)
        model = PeftModel.from_pretrained(base_model, model_name_or_path, is_trainable=False)
        model.to(device)
        return model, attn_impl, True

    model, attn_impl = _load_base_model(model_name_or_path, args, device)
    return model, attn_impl, False

def _maybe_apply_lora(
    model: torch.nn.Module,
    args: argparse.Namespace,
    finetune_type: str,
) -> torch.nn.Module:
    if finetune_type != "lora":
        return model
    _ensure_peft_available()
    if isinstance(model, PeftModel):
        return model

    target_modules = _parse_csv_list(args.lora_target_modules)
    if not target_modules:
        raise ValueError("--lora_target_modules must be non-empty when stage finetune type is lora.")

    modules_to_save = _parse_csv_list(args.lora_modules_to_save)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=args.lora_use_rslora,
    )
    return get_peft_model(model, lora_cfg)

def _count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return trainable, total

def _write_alpaca_json(
    raw_dataset,
    instruction_field: str,
    response_field: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, str]] = []
    for row in raw_dataset:
        records.append(
            {
                "instruction": str(row[instruction_field]),
                "input": "",
                "output": str(row[response_field]),
            }
        )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

def _write_llamafactory_dataset_info(
    dataset_dir: Path,
    train_name: str,
    train_file: str,
    eval_name: Optional[str] = None,
    eval_file: Optional[str] = None,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    info = {
        train_name: {
            "file_name": train_file,
        }
    }
    if eval_name is not None and eval_file is not None:
        info[eval_name] = {
            "file_name": eval_file,
        }
    with (dataset_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

def _run_llamafactory_stage1(
    args: argparse.Namespace,
    output_dir: Path,
    raw_new_train,
    raw_new_val,
) -> None:
    dataset_dir = (
        Path(args.output_dir) / "_llamafactory_data"
        if args.llamafactory_cutoff_len is None  # keep deterministic dir selection
        else Path(args.output_dir) / "_llamafactory_data"
    )
    train_name = "nads_stage1_train"
    eval_name = "nads_stage1_eval"
    train_file = f"{train_name}.json"
    eval_file = f"{eval_name}.json"

    _write_alpaca_json(
        raw_dataset=raw_new_train,
        instruction_field=args.new_instruction_field,
        response_field=args.new_response_field,
        out_path=dataset_dir / train_file,
    )
    if raw_new_val is not None:
        _write_alpaca_json(
            raw_dataset=raw_new_val,
            instruction_field=args.new_instruction_field,
            response_field=args.new_response_field,
            out_path=dataset_dir / eval_file,
        )

    _write_llamafactory_dataset_info(
        dataset_dir=dataset_dir,
        train_name=train_name,
        train_file=train_file,
        eval_name=eval_name if raw_new_val is not None else None,
        eval_file=eval_file if raw_new_val is not None else None,
    )

    finetuning_type = "lora" if args.stage1_finetune_type == "lora" else "full"
    cutoff_len = args.llamafactory_cutoff_len if args.llamafactory_cutoff_len is not None else args.max_seq_length
    cmd = [
        args.llamafactory_cli,
        "train",
        "--stage",
        "sft",
        "--do_train",
        "true",
        "--model_name_or_path",
        args.model_name_or_path,
        "--dataset_dir",
        str(dataset_dir.resolve()),
        "--dataset",
        train_name,
        "--template",
        args.llamafactory_template,
        "--finetuning_type",
        finetuning_type,
        "--output_dir",
        str(output_dir.resolve()),
        "--cutoff_len",
        str(cutoff_len),
        "--learning_rate",
        str(args.stage1_lr),
        "--num_train_epochs",
        str(args.stage1_epochs),
        "--per_device_train_batch_size",
        str(args.stage1_batch_size),
        "--gradient_accumulation_steps",
        str(args.stage1_grad_accum_steps),
        "--lr_scheduler_type",
        "linear",
        "--warmup_ratio",
        str(args.stage1_warmup_ratio),
        "--weight_decay",
        str(args.stage1_weight_decay),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--save_strategy",
        "epoch",
        "--logging_steps",
        "10",
        "--overwrite_output_dir",
        "true",
    ]

    if raw_new_val is not None:
        cmd.extend(
            [
                "--do_eval",
                "true",
                "--eval_dataset",
                eval_name,
                "--per_device_eval_batch_size",
                str(args.stage1_batch_size),
                "--evaluation_strategy",
                "epoch",
            ]
        )
    else:
        cmd.extend(["--do_eval", "false"])

    if args.dtype == "bfloat16":
        cmd.extend(["--bf16", "true"])
    elif args.dtype == "float16":
        cmd.extend(["--fp16", "true"])

    if args.stage1_finetune_type == "lora":
        target_modules = _parse_csv_list(args.lora_target_modules) or []
        cmd.extend(["--lora_rank", str(args.lora_r), "--lora_alpha", str(args.lora_alpha)])
        cmd.extend(["--lora_dropout", str(args.lora_dropout)])
        if target_modules:
            cmd.extend(["--lora_target", ",".join(target_modules)])

    if args.llamafactory_extra_args.strip():
        cmd.extend(shlex.split(args.llamafactory_extra_args.strip()))

    print("[Stage1] Running LLaMA-Factory command:")
    print(_quote_cmd(cmd))
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)

def _save_model_checkpoint(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    finetune_type: str,
    merge_lora_before_save: bool,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_format = "full_model"

    if finetune_type == "lora":
        if not isinstance(model, PeftModel):
            raise RuntimeError("Expected a PeftModel for LoRA save, but got full model.")
        if merge_lora_before_save:
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
            saved_format = "lora_merged_full_model"
        else:
            model.save_pretrained(output_dir)
            saved_format = "lora_adapter"
    else:
        model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)
    return saved_format

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

def _build_distributed_sampler(
    dataset,
    is_distributed: bool,
    rank: int,
    world_size: int,
    shuffle: bool,
    seed: int,
) -> Optional[DistributedSampler]:
    if not is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )

class LengthBucketedDistributedSampler(Sampler[int]):
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        bucket_window_mult: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive for length bucketing.")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive for length bucketing.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_window_mult = max(1, bucket_window_mult)
        self.drop_last = drop_last
        self.epoch = 0

        if self.drop_last and len(self.lengths) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.lengths) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if not self.lengths:
            return iter([])

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            shuffled_indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        else:
            shuffled_indices = list(range(len(self.lengths)))

        window_size = max(self.batch_size, self.batch_size * self.bucket_window_mult)
        ordered_indices: List[int] = []
        for start in range(0, len(shuffled_indices), window_size):
            window = shuffled_indices[start : start + window_size]
            window.sort(key=lambda idx: self.lengths[idx])

            if self.shuffle:
                batches = [window[i : i + self.batch_size] for i in range(0, len(window), self.batch_size)]
                if len(batches) > 1:
                    batch_perm = torch.randperm(len(batches), generator=generator).tolist()
                    batches = [batches[i] for i in batch_perm]
                ordered_indices.extend(chain.from_iterable(batches))
            else:
                ordered_indices.extend(window)

        if not self.drop_last:
            padding_size = self.total_size - len(ordered_indices)
            if padding_size > 0:
                if padding_size <= len(ordered_indices):
                    ordered_indices += ordered_indices[:padding_size]
                else:
                    ordered_indices += (ordered_indices * math.ceil(padding_size / len(ordered_indices)))[:padding_size]
        else:
            ordered_indices = ordered_indices[: self.total_size]

        per_rank_indices = ordered_indices[self.rank : self.total_size : self.num_replicas]
        return iter(per_rank_indices)

def _extract_input_lengths(dataset) -> List[int]:
    return [len(ids) for ids in dataset["input_ids"]]

def _build_train_sampler(
    dataset,
    batch_size: int,
    is_distributed: bool,
    rank: int,
    world_size: int,
    shuffle: bool,
    seed: int,
    enable_length_bucketing: bool,
    bucket_window_mult: int,
) -> Optional[Sampler[int]]:
    if dataset is None:
        return None
    if enable_length_bucketing:
        return LengthBucketedDistributedSampler(
            lengths=_extract_input_lengths(dataset),
            batch_size=batch_size,
            num_replicas=world_size if is_distributed else 1,
            rank=rank if is_distributed else 0,
            shuffle=shuffle,
            seed=seed,
            bucket_window_mult=bucket_window_mult,
            drop_last=False,
        )
    return _build_distributed_sampler(
        dataset=dataset,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        seed=seed,
    )

def _make_loader_kwargs(
    collate_fn,
    num_workers: int,
    device: torch.device,
) -> Dict:
    kwargs = {
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs

def _wrap_ddp_if_needed(
    model: torch.nn.Module,
    is_distributed: bool,
    device: torch.device,
    local_rank: int,
) -> torch.nn.Module:
    if not is_distributed:
        return model
    if device.type == "cuda":
        return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    return DDP(model, find_unused_parameters=False)

def _load_selected_indices(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "selected_indices" in data:
        data = data["selected_indices"]
    if not isinstance(data, list):
        raise ValueError(f"selected_indices file must contain a list or a dict with key 'selected_indices': {path}")
    return [int(x) for x in data]

def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size, device = _init_distributed_and_device(args)
    is_main = _is_main_process(rank)

    try:
        set_global_seed(args.seed + rank)

        use_amp = (not args.disable_amp) and args.dtype in {"bfloat16", "float16"}
        used_attn_impls: Dict[str, Optional[str]] = {}
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        if args.stage3_backend == "llamafactory":
            raise ValueError(
                "Stage3 currently uses custom joint task+KD objective (Eq.10-12). "
                "To keep algorithm unchanged, `--stage3_backend llamafactory` is disabled."
            )
        if is_distributed and (not args.skip_stage1) and args.stage1_backend == "llamafactory":
            raise ValueError(
                "Stage1 with --stage1_backend llamafactory does not support external DDP launch here. "
                "Use --stage1_backend native for torchrun multi-GPU."
            )

        out_dir = Path(args.output_dir)
        stage1_dir = out_dir / "stage1_navigator"
        stage3_dir = out_dir / "stage3_final"
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

        # ---------- Load and prepare D_new ----------
        if is_main:
            print("[Stage0] Loading new-task dataset...")
        raw_new = load_instruction_dataset(
            path_or_name=args.new_data_path,
            split=args.new_dataset_split,
            max_samples=args.max_new_samples,
        )
        raw_new_train, raw_new_val = split_train_val(raw_new, val_ratio=args.val_ratio, seed=args.seed)

        sft_new_train = build_sft_dataset(
            raw_dataset=raw_new_train,
            tokenizer=tokenizer,
            instruction_field=args.new_instruction_field,
            response_field=args.new_response_field,
            max_seq_length=args.max_seq_length,
            prompt_style=args.new_prompt_style,
            num_proc=args.num_proc,
        )
        sft_new_val = None
        if raw_new_val is not None:
            sft_new_val = build_sft_dataset(
                raw_dataset=raw_new_val,
                tokenizer=tokenizer,
                instruction_field=args.new_instruction_field,
                response_field=args.new_response_field,
                max_seq_length=args.max_seq_length,
                prompt_style=args.new_prompt_style,
                num_proc=args.num_proc,
            )

        sft_collator = CausalLMCollator(tokenizer=tokenizer, include_labels=True)
        stage1_train_sampler = _build_train_sampler(
            dataset=sft_new_train,
            batch_size=args.stage1_batch_size,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            seed=args.seed,
            enable_length_bucketing=not args.disable_length_bucketing,
            bucket_window_mult=args.length_bucket_window_mult,
        )
        stage1_val_sampler = _build_distributed_sampler(
            dataset=sft_new_val,
            is_distributed=is_distributed and sft_new_val is not None,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            seed=args.seed,
        )
        stage1_loader_kwargs = _make_loader_kwargs(
            collate_fn=sft_collator,
            num_workers=args.num_workers,
            device=device,
        )
        stage1_train_loader = DataLoader(
            sft_new_train,
            batch_size=args.stage1_batch_size,
            shuffle=stage1_train_sampler is None,
            sampler=stage1_train_sampler,
            **stage1_loader_kwargs,
        )
        stage1_val_loader = (
            DataLoader(
                sft_new_val,
                batch_size=args.stage1_batch_size,
                shuffle=False,
                sampler=stage1_val_sampler,
                **stage1_loader_kwargs,
            )
            if sft_new_val is not None
            else None
        )

        # ---------- Stage 1: train navigator ----------
        navigator_path = Path(args.navigator_ckpt) if args.navigator_ckpt else stage1_dir

        if args.skip_stage1:
            if is_main:
                if navigator_path.exists():
                    print(f"[Stage1] Skipped. Using existing navigator: {navigator_path}")
                else:
                    print("[Stage1] Skipped. No navigator checkpoint required for Stage3-only run.")
        else:
            if args.stage1_backend == "llamafactory":
                if is_main:
                    print("[Stage1] Training navigator via LLaMA-Factory...")
                    _run_llamafactory_stage1(
                        args=args,
                        output_dir=navigator_path,
                        raw_new_train=raw_new_train,
                        raw_new_val=raw_new_val,
                    )
                    _save_json(
                        navigator_path / "metrics.json",
                        {
                            "backend": "llamafactory",
                            "finetune_type": args.stage1_finetune_type,
                        },
                    )
                used_attn_impls["stage1_navigator"] = args.attn_implementation
                if is_distributed and dist.is_initialized():
                    dist.barrier()
            else:
                if is_main:
                    print("[Stage1] Training navigator model...")
                nav_model, nav_attn_impl, _ = _load_model(args.model_name_or_path, args, device, allow_peft_adapter=True)
                nav_model = _maybe_apply_lora(nav_model, args=args, finetune_type=args.stage1_finetune_type)
                nav_trainable, nav_total = _count_parameters(nav_model)
                if is_main:
                    print(f"[Stage1] Params trainable/total: {nav_trainable}/{nav_total}")
                used_attn_impls["stage1_navigator"] = nav_attn_impl

                stage1_cfg = TrainConfig(
                    epochs=args.stage1_epochs,
                    lr=args.stage1_lr,
                    weight_decay=args.stage1_weight_decay,
                    warmup_ratio=args.stage1_warmup_ratio,
                    grad_accum_steps=args.stage1_grad_accum_steps,
                    max_grad_norm=args.max_grad_norm,
                    use_amp=use_amp,
                    early_stopping_patience=args.stage1_early_stop_patience,
                    early_stopping_delta=args.stage1_early_stop_delta,
                    log_every_n_steps=args.log_every_n_steps,
                )

                nav_model_for_train = _wrap_ddp_if_needed(
                    model=nav_model,
                    is_distributed=is_distributed,
                    device=device,
                    local_rank=local_rank,
                )
                stage1_out = train_navigator(
                    model=nav_model_for_train,
                    train_loader=stage1_train_loader,
                    val_loader=stage1_val_loader,
                    config=stage1_cfg,
                    device=device,
                )

                if is_main:
                    stage1_saved_format = _save_model_checkpoint(
                        model=nav_model,
                        tokenizer=tokenizer,
                        output_dir=navigator_path,
                        finetune_type=args.stage1_finetune_type,
                        merge_lora_before_save=args.lora_merge_before_save,
                    )
                    _save_json(
                        navigator_path / "metrics.json",
                        {
                            "backend": "native",
                            "best_val_loss": stage1_out.best_val_loss,
                            "steps": stage1_out.steps,
                            "finetune_type": args.stage1_finetune_type,
                            "saved_format": stage1_saved_format,
                        },
                    )
                if is_distributed and dist.is_initialized():
                    dist.barrier()

                if is_distributed:
                    del nav_model_for_train
                del nav_model
                _cleanup_memory()

        # ---------- Stage 3: final model with distillation ----------
        selected_indices = None
        selected_indices_file = None

        if args.skip_stage3:
            if is_main:
                print("[Stage3] Skipped by user.")
        else:
            if args.selected_indices_path is None:
                raise ValueError("Stage3 requires --selected_indices_path (generated by select_data.py)")
            if args.candidate_data_path is None:
                raise ValueError("Stage3 requires --candidate_data_path")

            if is_main:
                print("[Stage3] Loading selected constraint set...")
            selected_indices = _load_selected_indices(args.selected_indices_path)
            selected_indices_file = str(Path(args.selected_indices_path).resolve())

            raw_candidate = load_instruction_dataset(
                path_or_name=args.candidate_data_path,
                split=args.candidate_dataset_split,
                max_samples=args.max_candidate_samples,
            )
            selected_constraint_raw = raw_candidate.select(selected_indices)

            kd_dataset = build_full_text_dataset(
                raw_dataset=selected_constraint_raw,
                tokenizer=tokenizer,
                instruction_field=args.candidate_instruction_field,
                response_field=args.candidate_response_field,
                max_seq_length=args.max_seq_length,
                prompt_style=args.candidate_prompt_style,
                num_proc=args.num_proc,
            )

            task_train_sampler = _build_train_sampler(
                dataset=sft_new_train,
                batch_size=args.stage3_task_batch_size,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                shuffle=True,
                seed=args.seed + 17,
                enable_length_bucketing=not args.disable_length_bucketing,
                bucket_window_mult=args.length_bucket_window_mult,
            )
            task_val_sampler = _build_distributed_sampler(
                dataset=sft_new_val,
                is_distributed=is_distributed and sft_new_val is not None,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                seed=args.seed + 23,
            )
            kd_sampler = _build_train_sampler(
                dataset=kd_dataset,
                batch_size=args.stage3_kd_batch_size,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                shuffle=True,
                seed=args.seed + 31,
                enable_length_bucketing=not args.disable_length_bucketing,
                bucket_window_mult=args.length_bucket_window_mult,
            )

            task_loader_kwargs = _make_loader_kwargs(
                collate_fn=CausalLMCollator(tokenizer=tokenizer, include_labels=True),
                num_workers=args.num_workers,
                device=device,
            )
            kd_loader_kwargs = _make_loader_kwargs(
                collate_fn=CausalLMCollator(tokenizer=tokenizer, include_labels=False),
                num_workers=args.num_workers,
                device=device,
            )
            task_train_loader = DataLoader(
                sft_new_train,
                batch_size=args.stage3_task_batch_size,
                shuffle=task_train_sampler is None,
                sampler=task_train_sampler,
                **task_loader_kwargs,
            )
            task_val_loader = (
                DataLoader(
                    sft_new_val,
                    batch_size=args.stage3_task_batch_size,
                    shuffle=False,
                    sampler=task_val_sampler,
                    **task_loader_kwargs,
                )
                if sft_new_val is not None
                else None
            )
            kd_loader = DataLoader(
                kd_dataset,
                batch_size=args.stage3_kd_batch_size,
                shuffle=kd_sampler is None,
                sampler=kd_sampler,
                **kd_loader_kwargs,
            )

            if is_main:
                print("[Stage3] Training final model with distillation...")
            student_model, student_attn_impl, _ = _load_model(
                args.model_name_or_path,
                args,
                device,
                allow_peft_adapter=True,
            )
            student_model = _maybe_apply_lora(student_model, args=args, finetune_type=args.stage3_finetune_type)

            teacher_source = args.model_name_or_path
            if _is_peft_adapter_checkpoint(teacher_source):
                _ensure_peft_available()
                teacher_source = PeftConfig.from_pretrained(teacher_source).base_model_name_or_path
            teacher_model, teacher_attn_impl, _ = _load_model(
                teacher_source,
                args,
                device,
                allow_peft_adapter=False,
            )
            used_attn_impls["stage3_student"] = student_attn_impl
            used_attn_impls["stage3_teacher"] = teacher_attn_impl
            for p in teacher_model.parameters():
                p.requires_grad = False
            teacher_model.eval()
            stage3_trainable, stage3_total = _count_parameters(student_model)
            if is_main:
                print(f"[Stage3] Student params trainable/total: {stage3_trainable}/{stage3_total}")

            stage3_cfg = TrainConfig(
                epochs=args.stage3_epochs,
                lr=args.stage3_lr,
                weight_decay=args.stage3_weight_decay,
                warmup_ratio=args.stage3_warmup_ratio,
                grad_accum_steps=args.stage3_grad_accum_steps,
                max_grad_norm=args.max_grad_norm,
                use_amp=use_amp,
                early_stopping_patience=args.stage3_early_stop_patience,
                early_stopping_delta=args.stage3_early_stop_delta,
                log_every_n_steps=args.log_every_n_steps,
            )

            student_model_for_train = _wrap_ddp_if_needed(
                model=student_model,
                is_distributed=is_distributed,
                device=device,
                local_rank=local_rank,
            )
            stage3_out = train_final_with_distillation(
                student_model=student_model_for_train,
                teacher_model=teacher_model,
                task_train_loader=task_train_loader,
                kd_train_loader=kd_loader,
                task_val_loader=task_val_loader,
                lambda_kd=args.distill_lambda,
                config=stage3_cfg,
                device=device,
            )

            if is_main:
                stage3_saved_format = _save_model_checkpoint(
                    model=student_model,
                    tokenizer=tokenizer,
                    output_dir=stage3_dir,
                    finetune_type=args.stage3_finetune_type,
                    merge_lora_before_save=args.lora_merge_before_save,
                )
                _save_json(
                    stage3_dir / "metrics.json",
                    {
                        "best_val_loss": stage3_out.best_val_loss,
                        "steps": stage3_out.steps,
                        "finetune_type": args.stage3_finetune_type,
                        "saved_format": stage3_saved_format,
                    },
                )
            if is_distributed and dist.is_initialized():
                dist.barrier()

            if is_distributed:
                del student_model_for_train
            del student_model
            del teacher_model
            _cleanup_memory()

        if not is_main:
            return

        # ---------- Final run summary ----------
        summary = {
            "model_name_or_path": args.model_name_or_path,
            "attn_implementation_requested": args.attn_implementation,
            "attn_implementation_used": used_attn_impls,
            "new_data_path": args.new_data_path,
            "new_train_size": len(raw_new_train),
            "new_val_size": 0 if raw_new_val is None else len(raw_new_val),
            "navigator_path": str(navigator_path.resolve()),
            "stage3_enabled": not args.skip_stage3,
            "stage1_backend": args.stage1_backend,
            "stage3_backend": args.stage3_backend,
            "stage1_finetune_type": args.stage1_finetune_type,
            "stage3_finetune_type": args.stage3_finetune_type,
            "lora_merge_before_save": args.lora_merge_before_save,
            "selected_constraint_size": 0 if selected_indices is None else len(selected_indices),
            "selected_indices_file": selected_indices_file,
            "length_bucketing_enabled": not args.disable_length_bucketing,
            "length_bucket_window_mult": args.length_bucket_window_mult,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
            "distributed": is_distributed,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
        }
        _save_json(out_dir / "run_summary.json", summary)

        print("[Done] Training pipeline completed.")
        print(f"Output directory: {out_dir}")
    finally:
        _finalize_distributed(is_distributed)

if __name__ == "__main__":
    main()