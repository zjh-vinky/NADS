import copy
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    grad_accum_steps: int
    max_grad_norm: float
    use_amp: bool
    early_stopping_patience: int
    early_stopping_delta: float
    log_every_n_steps: int


@dataclass
class TrainOutput:
    best_val_loss: Optional[float]
    steps: int


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _build_linear_warmup_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Linear decay to 0.
        remain_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - current_step) / float(remain_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    if not _is_dist_initialized():
        return True
    return dist.get_rank() == 0


def _set_dataloader_epoch(dataloader: Optional[DataLoader], epoch: int) -> None:
    if dataloader is None:
        return
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _should_log_step(step: int, total_steps: int, log_every_n_steps: int) -> bool:
    if log_every_n_steps <= 1:
        return True
    if step == total_steps:
        return True
    return step % log_every_n_steps == 0


def _maybe_no_sync(model: torch.nn.Module, enabled: bool):
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def _compute_task_loss_sample_average(
    model: torch.nn.Module,
    batch: dict,
    use_amp: bool,
) -> torch.Tensor:

    if "token_type_ids" not in batch or batch["token_type_ids"] is None:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

    amp_enabled = use_amp and batch["input_ids"].device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
    with amp_ctx:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            use_cache=False,
        ).logits

    shift_logits = logits.float()[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    valid_mask = shift_labels.ne(IGNORE_INDEX)

    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    sample_denom = valid_mask.sum(dim=-1).clamp_min(1).to(token_loss.dtype)
    sample_loss = token_loss.sum(dim=-1) / sample_denom
    return sample_loss.mean()


@torch.no_grad()
def evaluate_task_loss(
    model: torch.nn.Module,
    dataloader: Optional[DataLoader],
    device: torch.device,
    use_amp: bool,
) -> Optional[float]:
    if dataloader is None:
        return None

    model.eval()
    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        loss = _compute_task_loss_sample_average(
            model=model,
            batch=batch,
            use_amp=use_amp,
        )

        bs = batch["input_ids"].size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs

    if _is_dist_initialized():
        stats = torch.tensor([total_loss, float(total_count)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_count = int(stats[1].item())

    model.train()
    if total_count == 0:
        return None
    return total_loss / total_count


def train_navigator(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainConfig,
    device: torch.device,
) -> TrainOutput:
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_train_steps = math.ceil(len(train_loader) / max(1, config.grad_accum_steps)) * config.epochs
    scheduler = _build_linear_warmup_scheduler(optimizer, total_train_steps, config.warmup_ratio)

    best_val_loss = None
    best_state = None
    patience_count = 0
    global_step = 0

    for epoch in range(config.epochs):
        _set_dataloader_epoch(train_loader, epoch)
        if _is_main_process():
            print(f"[Stage1][Epoch {epoch + 1}/{config.epochs}] start")
        pbar = tqdm(
            train_loader,
            desc=f"Stage1 Epoch {epoch + 1}/{config.epochs}",
            leave=False,
            disable=not _is_main_process(),
        )

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            should_sync = step % max(1, config.grad_accum_steps) == 0
            with _maybe_no_sync(model, enabled=not should_sync):
                batch = _move_batch_to_device(batch, device)
                task_loss = _compute_task_loss_sample_average(
                    model=model,
                    batch=batch,
                    use_amp=config.use_amp,
                )
                (task_loss / max(1, config.grad_accum_steps)).backward()

            if should_sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if _is_main_process() and _should_log_step(step, len(train_loader), config.log_every_n_steps):
                tqdm.write(
                    f"[Stage1][Epoch {epoch + 1}/{config.epochs}][step {step}/{len(train_loader)}] "
                    f"loss={float(task_loss.detach().cpu()):.6f} gstep={global_step}"
                )

        val_loss = evaluate_task_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=config.use_amp,
        )

        if val_loss is None:
            continue
        if _is_main_process():
            print(f"[Stage1][Epoch {epoch + 1}/{config.epochs}] val_loss={val_loss:.6f}")

        if best_val_loss is None or val_loss < (best_val_loss - config.early_stopping_delta):
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainOutput(best_val_loss=best_val_loss, steps=global_step)


def _compute_kd_loss(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    use_amp: bool,
) -> torch.Tensor:

    if "token_type_ids" not in batch and "input_ids" in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

    amp_enabled = use_amp and batch["input_ids"].device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
    with amp_ctx:
        with torch.inference_mode():
            teacher_logits = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"], 
                use_cache=False,
            ).logits
        student_logits = student_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"], 
            use_cache=False,
        ).logits

    teacher_logits = teacher_logits.float()[:, :-1, :]
    student_logits = student_logits.float()[:, :-1, :]
    valid_mask = batch["attention_mask"][:, 1:].float()

    log_p = F.log_softmax(teacher_logits, dim=-1)
    p = log_p.exp()
    log_q = F.log_softmax(student_logits, dim=-1)

    kl = (p * (log_p - log_q)).sum(dim=-1)
    sample_denom = valid_mask.sum(dim=-1).clamp_min(1.0)
    sample_kl = (kl * valid_mask).sum(dim=-1) / sample_denom
    kd_loss = sample_kl.mean()
    return kd_loss


def train_final_with_distillation(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    task_train_loader: DataLoader,
    kd_train_loader: DataLoader,
    task_val_loader: Optional[DataLoader],
    lambda_kd: float,
    config: TrainConfig,
    device: torch.device,
) -> TrainOutput:
    student_model.train()
    teacher_model.eval()

    optimizer = AdamW(student_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_train_steps = math.ceil(len(task_train_loader) / max(1, config.grad_accum_steps)) * config.epochs
    scheduler = _build_linear_warmup_scheduler(optimizer, total_train_steps, config.warmup_ratio)

    best_val_loss = None
    best_state = None
    patience_count = 0
    global_step = 0

    for epoch in range(config.epochs):
        _set_dataloader_epoch(task_train_loader, epoch)
        _set_dataloader_epoch(kd_train_loader, epoch)
        if _is_main_process():
            print(f"[Stage3][Epoch {epoch + 1}/{config.epochs}] start")
        pbar = tqdm(
            task_train_loader,
            desc=f"Stage3 Epoch {epoch + 1}/{config.epochs}",
            leave=False,
            disable=not _is_main_process(),
        )
        kd_iter = iter(kd_train_loader)

        optimizer.zero_grad(set_to_none=True)
        for step, task_batch in enumerate(pbar, start=1):
            should_sync = step % max(1, config.grad_accum_steps) == 0
            with _maybe_no_sync(student_model, enabled=not should_sync):
                try:
                    kd_batch = next(kd_iter)
                except StopIteration:
                    kd_iter = iter(kd_train_loader)
                    kd_batch = next(kd_iter)

                # Ensure token_type_ids exists.
                for batch in (task_batch, kd_batch):
                    if "token_type_ids" not in batch and "input_ids" in batch:
                        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

                task_batch = _move_batch_to_device(task_batch, device)
                kd_batch = _move_batch_to_device(kd_batch, device)

                task_loss = _compute_task_loss_sample_average(
                    model=student_model,
                    batch=task_batch,
                    use_amp=config.use_amp,
                )

                kd_loss = _compute_kd_loss(
                    student_model,
                    teacher_model,
                    kd_batch,
                    use_amp=config.use_amp,
                )

                total_loss = task_loss + lambda_kd * kd_loss
                (total_loss / max(1, config.grad_accum_steps)).backward()

            if should_sync:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if _is_main_process() and _should_log_step(step, len(task_train_loader), config.log_every_n_steps):
                tqdm.write(
                    f"[Stage3][Epoch {epoch + 1}/{config.epochs}][step {step}/{len(task_train_loader)}] "
                    f"task_loss={float(task_loss.detach().cpu()):.6f} "
                    f"kd_loss={float(kd_loss.detach().cpu()):.6f} "
                    f"total={float(total_loss.detach().cpu()):.6f} "
                    f"gstep={global_step}"
                )

        val_loss = evaluate_task_loss(
            model=student_model,
            dataloader=task_val_loader,
            device=device,
            use_amp=config.use_amp,
        )

        if val_loss is None:
            continue
        if _is_main_process():
            print(f"[Stage3][Epoch {epoch + 1}/{config.epochs}] val_loss={val_loss:.6f}")

        if best_val_loss is None or val_loss < (best_val_loss - config.early_stopping_delta):
            best_val_loss = val_loss
            best_state = copy.deepcopy(student_model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config.early_stopping_patience:
                break

    if best_state is not None:
        student_model.load_state_dict(best_state)

    return TrainOutput(best_val_loss=best_val_loss, steps=global_step)
