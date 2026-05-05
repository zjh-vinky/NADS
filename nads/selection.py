import os
import numpy as np

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class SelectionResult:
    selected_indices: np.ndarray
    utilities: np.ndarray
    projected_embeddings: np.ndarray
    weights: np.ndarray
    objective_history: List[float]
    embeddings: Optional[np.ndarray] = None


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _forward_logits_and_last_hidden(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get logits and last hidden state while avoiding full hidden-state stacks when possible.
    """
    # Fast path for common CausalLM models (Llama/Qwen/Gemma etc.)
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        base_out = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = base_out.last_hidden_state
        logits = model.lm_head(last_hidden)
        return logits, last_hidden

    # Fallback path for generic/unknown wrappers.
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    last_hidden = out.hidden_states[-1]
    logits = out.logits
    return logits, last_hidden


def _forward_logits_only(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        base_out = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        return model.lm_head(base_out.last_hidden_state)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    return out.logits


def _compute_token_utility(
    pre_logits: torch.Tensor,
    nav_logits: torch.Tensor,
    gamma: float,
    vocab_chunk_size: int = 0,
) -> torch.Tensor:
    """
    Exact Eq.(2) utility computation from logits:
      KL(P_pre || P_nav) * exp(-gamma * H(P_pre))

    When vocab_chunk_size > 0, compute expectations in vocab chunks to lower peak memory.
    """
    log_z_pre = torch.logsumexp(pre_logits, dim=-1)
    log_z_nav = torch.logsumexp(nav_logits, dim=-1)

    vocab_size = pre_logits.size(-1)
    if vocab_chunk_size <= 0 or vocab_chunk_size >= vocab_size:
        p = torch.softmax(pre_logits, dim=-1)
        expected_pre = (p * pre_logits).sum(dim=-1)
        expected_nav = (p * nav_logits).sum(dim=-1)
    else:
        expected_pre = torch.zeros_like(log_z_pre)
        expected_nav = torch.zeros_like(log_z_pre)
        log_z_pre_exp = log_z_pre.unsqueeze(-1)
        for start in range(0, vocab_size, vocab_chunk_size):
            end = min(start + vocab_chunk_size, vocab_size)
            pre_chunk = pre_logits[..., start:end]
            nav_chunk = nav_logits[..., start:end]
            p_chunk = torch.exp(pre_chunk - log_z_pre_exp)
            expected_pre = expected_pre + (p_chunk * pre_chunk).sum(dim=-1)
            expected_nav = expected_nav + (p_chunk * nav_chunk).sum(dim=-1)

    entropy = log_z_pre - expected_pre
    kl = (expected_pre - expected_nav) - log_z_pre + log_z_nav
    return kl * torch.exp(-gamma * entropy)


def _mean_pool_last_hidden(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    bos_token_id: Optional[int],
) -> torch.Tensor:
    mask = attention_mask.float()
    # Exclude leading BOS from sequence representation when BOS is prepended for AR alignment.
    if bos_token_id is not None and mask.size(1) > 0:
        first_is_bos = input_ids[:, 0].eq(int(bos_token_id))
        mask[:, 0] = mask[:, 0] * (~first_is_bos).to(mask.dtype)

    mask = mask.unsqueeze(-1)
    return (last_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def compute_utilities_and_embeddings(
    pretrain_model: torch.nn.Module,
    navigator_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    gamma: float,
    vocab_chunk_size: int = 0,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-pass version for Stage2:
    - utility u_i from Eq.(2)
    - embedding z_i (mean pooled last hidden state)
    """
    pretrain_model.eval()
    navigator_model.eval()

    dataset_size = len(dataloader.dataset)
    hidden_size = pretrain_model.config.hidden_size
    bos_token_id = getattr(pretrain_model.config, "bos_token_id", None)
    utilities = np.zeros(dataset_size, dtype=np.float64)
    embeddings = np.zeros((dataset_size, hidden_size), dtype=np.float32)

    pbar = tqdm(dataloader, desc="Computing utilities + embeddings (single pass)", leave=False)
    for batch in pbar:
        sample_id = batch["sample_id"].cpu().numpy()
        batch = _move_batch_to_device(batch, device)

        amp_enabled = use_amp and device.type == "cuda"
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
        with amp_ctx:
            pre_logits, last_hidden = _forward_logits_and_last_hidden(
                model=pretrain_model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            nav_logits = _forward_logits_only(
                model=navigator_model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        pre_logits = pre_logits.float()[:, :-1, :]
        nav_logits = nav_logits.float()[:, :-1, :]
        valid_mask = batch["attention_mask"][:, 1:].float()

        # Eq.(2): D_KL(P_pre || P_nav) * exp(-gamma * H(P_pre))
        token_utility = _compute_token_utility(
            pre_logits=pre_logits,
            nav_logits=nav_logits,
            gamma=gamma,
            vocab_chunk_size=vocab_chunk_size,
        )
        denom = valid_mask.sum(dim=-1).clamp_min(1.0)
        sample_utility = (token_utility * valid_mask).sum(dim=-1) / denom

        pooled = _mean_pool_last_hidden(
            last_hidden=last_hidden,
            attention_mask=batch["attention_mask"],
            input_ids=batch["input_ids"],
            bos_token_id=bos_token_id,
        )

        utilities[sample_id] = sample_utility.detach().cpu().numpy().astype(np.float64)
        embeddings[sample_id] = pooled.detach().cpu().numpy().astype(np.float32)

    return utilities, embeddings


@torch.inference_mode()
def compute_utilities_and_embeddings_sparse(
    pretrain_model: torch.nn.Module,
    navigator_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    gamma: float,
    vocab_chunk_size: int = 0,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sparse variant for distributed inference:
    returns (sample_ids, utilities, embeddings) for only local shard samples.
    """
    pretrain_model.eval()
    navigator_model.eval()

    # hidden_size = pretrain_model.config.hidden_size
    text_cfg = getattr(pretrain_model.config, "text_config", None)
    if isinstance(text_cfg, dict):
        hidden_size = text_cfg.get("hidden_size", None)
    else:
        hidden_size = getattr(text_cfg, "hidden_size", None)

    if hidden_size is None:
        # fallback for non-gemma configs
        hidden_size = getattr(pretrain_model.config, "hidden_size", None)

    if hidden_size is None:
        raise AttributeError(
            f"Cannot infer hidden size from config type={type(pretrain_model.config).__name__}"
        )
    bos_token_id = getattr(pretrain_model.config, "bos_token_id", None)
    all_sample_ids: List[np.ndarray] = []
    all_utilities: List[np.ndarray] = []
    all_embeddings: List[np.ndarray] = []

    pbar = tqdm(dataloader, desc="Computing local utilities + embeddings", leave=False)
    for batch in pbar:
        sample_id = batch["sample_id"].cpu().numpy().astype(np.int64)
        batch = _move_batch_to_device(batch, device)

        amp_enabled = use_amp and device.type == "cuda"
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
        with amp_ctx:
            pre_logits, last_hidden = _forward_logits_and_last_hidden(
                model=pretrain_model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            nav_logits = _forward_logits_only(
                model=navigator_model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        pre_logits = pre_logits.float()[:, :-1, :]
        nav_logits = nav_logits.float()[:, :-1, :]
        valid_mask = batch["attention_mask"][:, 1:].float()

        token_utility = _compute_token_utility(
            pre_logits=pre_logits,
            nav_logits=nav_logits,
            gamma=gamma,
            vocab_chunk_size=vocab_chunk_size,
        )
        denom = valid_mask.sum(dim=-1).clamp_min(1.0)
        sample_utility = (token_utility * valid_mask).sum(dim=-1) / denom

        pooled = _mean_pool_last_hidden(
            last_hidden=last_hidden,
            attention_mask=batch["attention_mask"],
            input_ids=batch["input_ids"],
            bos_token_id=bos_token_id,
        )

        all_sample_ids.append(sample_id)
        all_utilities.append(sample_utility.detach().cpu().numpy().astype(np.float64))
        all_embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))

    if all_sample_ids:
        sample_ids = np.concatenate(all_sample_ids, axis=0)
        utilities = np.concatenate(all_utilities, axis=0)
        embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        sample_ids = np.zeros((0,), dtype=np.int64)
        utilities = np.zeros((0,), dtype=np.float64)
        embeddings = np.zeros((0, hidden_size), dtype=np.float32)

    return sample_ids, utilities, embeddings


@torch.inference_mode()
def compute_forgetting_utilities(
    pretrain_model: torch.nn.Module,
    navigator_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    gamma: float,
    vocab_chunk_size: int = 0,
    use_amp: bool = True,
) -> np.ndarray:
    pretrain_model.eval()
    navigator_model.eval()

    dataset_size = len(dataloader.dataset)
    utilities = np.zeros(dataset_size, dtype=np.float64)

    pbar = tqdm(dataloader, desc="Computing Eq.(2) utilities", leave=False)
    for batch in pbar:
        sample_id = batch["sample_id"].cpu().numpy()
        batch = _move_batch_to_device(batch, device)

        amp_enabled = use_amp and device.type == "cuda"
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
        with amp_ctx:
            pre_logits = _forward_logits_only(
                model=pretrain_model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            nav_logits = _forward_logits_only(
                model=navigator_model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        pre_logits = pre_logits.float()[:, :-1, :]
        nav_logits = nav_logits.float()[:, :-1, :]
        valid_mask = batch["attention_mask"][:, 1:].float()

        # Eq.(2): D_KL(P_pre || P_nav) * exp(-gamma * H(P_pre))
        token_utility = _compute_token_utility(
            pre_logits=pre_logits,
            nav_logits=nav_logits,
            gamma=gamma,
            vocab_chunk_size=vocab_chunk_size,
        )

        denom = valid_mask.sum(dim=-1).clamp_min(1.0)
        sample_utility = (token_utility * valid_mask).sum(dim=-1) / denom

        utilities[sample_id] = sample_utility.detach().cpu().numpy().astype(np.float64)

    return utilities


@torch.inference_mode()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> np.ndarray:
    model.eval()

    dataset_size = len(dataloader.dataset)
    hidden_size = model.config.hidden_size
    bos_token_id = getattr(model.config, "bos_token_id", None)
    embeddings = np.zeros((dataset_size, hidden_size), dtype=np.float32)

    pbar = tqdm(dataloader, desc="Extracting z_i embeddings", leave=False)
    for batch in pbar:
        sample_id = batch["sample_id"].cpu().numpy()
        batch = _move_batch_to_device(batch, device)

        amp_enabled = use_amp and device.type == "cuda"
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
        with amp_ctx:
            _, last_hidden = _forward_logits_and_last_hidden(
                model=model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        pooled = _mean_pool_last_hidden(
            last_hidden=last_hidden,
            attention_mask=batch["attention_mask"],
            input_ids=batch["input_ids"],
            bos_token_id=bos_token_id,
        )

        embeddings[sample_id] = pooled.detach().cpu().numpy().astype(np.float32)

    return embeddings


def random_project_and_whiten(
    embeddings: np.ndarray,
    proj_dim: int,
    rho: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

    n, d = embeddings.shape
    if proj_dim > d:
        raise ValueError(f"proj_dim ({proj_dim}) must be <= original dim ({d})")

    rng = np.random.default_rng(seed)

    mu = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mu

    # Eq.(7): random projection.
    random_matrix = rng.standard_normal((d, proj_dim), dtype=np.float32) / np.sqrt(proj_dim)
    projected = centered @ random_matrix  # [N, E]

    # Whitening: z <- (Sigma + rho I)^(-1/2) z
    cov = (projected.T @ projected) / float(n)
    cov_reg = cov + rho * np.eye(proj_dim, dtype=np.float32)

    eigvals, eigvecs = np.linalg.eigh(cov_reg)
    eigvals = np.clip(eigvals, 1e-8, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    whitened = projected @ inv_sqrt

    aux = {
        "mu": mu,
        "random_matrix": random_matrix,
        "cov": cov,
        "cov_reg": cov_reg,
        "inv_sqrt": inv_sqrt,
    }
    return whitened.astype(np.float32), aux


def _objective_tilde(
    w: np.ndarray,
    utilities: np.ndarray,
    z2: np.ndarray,
    beta: float,
    nu: float,
) -> float:
    # Objective: sum_i w_i u_i + beta * sum_e log(1 + sum_i w_i z_{i,e}^2) - (nu/2) * ||w||_2^2
    energy = (w[:, None] * z2).sum(axis=0)
    weighted_util = (w * utilities).sum()
    logdet_part = beta * np.log1p(energy).sum()
    l2_part = -0.5 * nu * np.dot(w, w)
    total = float(weighted_util + logdet_part + l2_part)
    return total, weighted_util, logdet_part, l2_part


def iterative_hard_thresholding_select(
    utilities: np.ndarray,
    projected_embeddings: np.ndarray,
    k: int,
    beta: float,
    nu: float,
    eta: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    n = utilities.shape[0]
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if nu < 0:
        raise ValueError(f"nu must be non-negative, got {nu}")

    z2 = projected_embeddings.astype(np.float64) ** 2
    u = utilities.astype(np.float64)

    w = np.zeros(n, dtype=np.float64)
    history: List[float] = []
    history_detail: List[tuple] = []

    for step in tqdm(range(steps), desc="IHT optimization Eq.(9)", leave=False):
        denom = 1.0 + (w[:, None] * z2).sum(axis=0)  # [E]

        # Closed-form gradient with L2 regularization term.
        grad = u + beta * (z2 / denom[None, :]).sum(axis=1) - nu * w

        # Eq.(9): w <- P_k((w + eta * grad)_+)
        v = np.maximum(w + eta * grad, 0.0)
        topk_idx = np.argpartition(v, -k)[-k:]
        new_w = np.zeros_like(w)
        new_w[topk_idx] = v[topk_idx]
        w = new_w

        total, weighted_util, logdet_part, l2_part = _objective_tilde(w, u, z2, beta, nu)
        history.append(total)
        history_detail.append((weighted_util, logdet_part, l2_part))
        print(f"[IHT step {step+1}] F(w)={total:.6f}, weighted_util={weighted_util:.6f}, logdet={logdet_part:.6f}, l2reg={l2_part:.6f}")

    final_idx = np.argpartition(w, -k)[-k:]
    final_idx = final_idx[np.argsort(-w[final_idx])]
    return final_idx.astype(np.int64), w.astype(np.float64), history


def run_nads_selection(
    pretrain_model: torch.nn.Module,
    navigator_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    gamma: float,
    beta: float,
    nu: float,
    rho: float,
    proj_dim: int,
    k: int,
    eta: float,
    steps: int,
    seed: int,
    vocab_chunk_size: int = 0,
    use_amp: bool = True,
) -> SelectionResult:
    utilities_path = "nads_results/llama3_code_weight/utilities.npy"
    embeddings_path = "nads_results/llama3_code_weight/projected_embeddings.npy"
    if os.path.exists(utilities_path) and os.path.exists(embeddings_path):
        print("[NADS] Loading cached utilities and embeddings ...")
        utilities = np.load(utilities_path)
        embeddings = np.load(embeddings_path)
    else:
        utilities, embeddings = compute_utilities_and_embeddings(
            pretrain_model=pretrain_model,
            navigator_model=navigator_model,
            dataloader=dataloader,
            device=device,
            gamma=gamma,
            vocab_chunk_size=vocab_chunk_size,
            use_amp=use_amp,
        )
        np.save(utilities_path, utilities)
        np.save(embeddings_path, embeddings)

    projected, _ = random_project_and_whiten(
        embeddings=embeddings,
        proj_dim=proj_dim,
        rho=rho,
        seed=seed,
    )

    selected_idx, weights, history = iterative_hard_thresholding_select(
        utilities=utilities,
        projected_embeddings=projected,
        k=k,
        beta=beta,
        nu=nu,
        eta=eta,
        steps=steps,
    )

    return SelectionResult(
        selected_indices=selected_idx,
        utilities=utilities,
        projected_embeddings=projected,
        embeddings=embeddings,
        weights=weights,
        objective_history=history,
    )
