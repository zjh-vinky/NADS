import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

IGNORE_INDEX = -100

PROMPT_DEFAULT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

PROMPT_CODE = (
    "You are a proficient coding assistant. "
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def _parse_slice_spec(path_or_name: str) -> Tuple[str, Optional[Tuple[Optional[int], Optional[int]]]]:
    if "[" not in path_or_name or not path_or_name.endswith("]"):
        return path_or_name, None
    base = path_or_name[: path_or_name.index("[")]
    body = path_or_name[path_or_name.index("[") + 1 : -1]
    if ":" in body:
        start_s, end_s = body.split(":", 1)
        start = int(start_s) if start_s else None
        end = int(end_s) if end_s else None
        return base, (start, end)
    idx = int(body)
    return base, (idx, idx + 1)


def _apply_slice(dataset: Dataset, sl: Optional[Tuple[Optional[int], Optional[int]]]) -> Dataset:
    if sl is None:
        return dataset
    start, end = sl
    if start is None:
        start = 0
    if end is None:
        end = len(dataset)
    start = max(0, start)
    end = min(len(dataset), end)
    if end <= start:
        raise ValueError(f"Invalid slice [{start}:{end}] for dataset length {len(dataset)}")
    return dataset.select(range(start, end))


def _load_dataset_from_dir(path: str, split: str) -> Dataset:
    # Priority 1: HuggingFace saved dataset directory.
    try:
        loaded = load_from_disk(path)
        if isinstance(loaded, DatasetDict):
            if split not in loaded:
                raise ValueError(f"Split '{split}' not found in {path}. Available: {list(loaded.keys())}")
            return loaded[split]
        return loaded
    except Exception:
        pass

    def _glob_recursive(exts: Sequence[str]) -> List[str]:
        files: List[str] = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True))
        files = [f for f in files if os.path.isfile(f)]
        # Exclude Hugging Face dataset metadata files from raw data loading.
        skip_json_names = {"dataset_info.json", "dataset_infos.json"}
        filtered: List[str] = []
        for f in files:
            name = os.path.basename(f).lower()
            if name in skip_json_names:
                continue
            filtered.append(f)
        files = filtered
        return sorted(set(files))

    # Priority 2: directory containing parquet/json/jsonl/csv files.
    parquet_files = _glob_recursive([".parquet"])
    if parquet_files:
        return load_dataset("parquet", data_files=parquet_files, split=split)

    json_files = _glob_recursive([".json", ".jsonl"])
    if json_files:
        return load_dataset("json", data_files=json_files, split=split)

    csv_files = _glob_recursive([".csv"])
    if csv_files:
        return load_dataset("csv", data_files=csv_files, split=split)

    raise ValueError(f"Unable to load dataset from directory: {path}")


def load_instruction_dataset(
    path_or_name: str,
    split: str,
    max_samples: Optional[int] = None,
) -> Dataset:
    path_or_name, slice_spec = _parse_slice_spec(path_or_name)

    if os.path.isdir(path_or_name):
        dataset = _load_dataset_from_dir(path_or_name, split)
    elif path_or_name.endswith(".json") or path_or_name.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=path_or_name, split=split)
    elif path_or_name.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=path_or_name, split=split)
    else:
        dataset = load_dataset(path_or_name, split=split)

    dataset = _apply_slice(dataset, slice_spec)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    return dataset


def split_train_val(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Optional[Dataset]]:
    if val_ratio <= 0.0:
        return dataset, None
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def build_source_prompt(instruction: str, prompt_style: str) -> str:
    if prompt_style == "code":
        return PROMPT_CODE.format(instruction=instruction)
    if prompt_style == "default":
        return PROMPT_DEFAULT.format(instruction=instruction)
    if prompt_style == "auto":
        # For code datasets like Magicoder, this usually works better.
        use_code = any(k in instruction.lower() for k in ["python", "code", "program", "debug"])
        template = PROMPT_CODE if use_code else PROMPT_DEFAULT
        return template.format(instruction=instruction)
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def _get_bos_token_id(tokenizer: transformers.PreTrainedTokenizer) -> Optional[int]:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        return None
    return int(bos_token_id)


def _build_sft_example(
    instruction: str,
    response: str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    prompt_style: str,
) -> Dict[str, List[int]]:
    bos_token_id = _get_bos_token_id(tokenizer)
    source = build_source_prompt(instruction, prompt_style)
    target = f"{response}{tokenizer.eos_token}"

    source_ids = tokenizer(source, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids

    input_ids = source_ids + target_ids
    labels = [IGNORE_INDEX] * len(source_ids) + target_ids
    if bos_token_id is not None:
        # Prepend BOS so autoregressive shifting predicts the first text token.
        input_ids = [bos_token_id] + input_ids
        labels = [IGNORE_INDEX] + labels

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def _build_full_text_example(
    instruction: str,
    response: str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    prompt_style: str,
) -> Dict[str, List[int]]:
    bos_token_id = _get_bos_token_id(tokenizer)
    text = build_source_prompt(instruction, prompt_style) + f"{response}{tokenizer.eos_token}"
    text_max_length = max_seq_length
    if bos_token_id is not None:
        text_max_length = max(1, max_seq_length - 1)

    input_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=text_max_length,
    ).input_ids
    if bos_token_id is not None:
        input_ids = [bos_token_id] + input_ids
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
    return {"input_ids": input_ids}


def build_sft_dataset(
    raw_dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    instruction_field: str,
    response_field: str,
    max_seq_length: int,
    prompt_style: str,
    num_proc: int = 1,
) -> Dataset:
    def _map_fn(examples: Dict[str, Sequence[str]]) -> Dict[str, List[List[int]]]:
        all_input_ids = []
        all_labels = []
        instructions = examples[instruction_field]
        responses = examples[response_field]
        for instruction, response in zip(instructions, responses):
            ex = _build_sft_example(
                instruction=str(instruction),
                response=str(response),
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                prompt_style=prompt_style,
            )
            all_input_ids.append(ex["input_ids"])
            all_labels.append(ex["labels"])
        return {"input_ids": all_input_ids, "labels": all_labels}

    dataset = raw_dataset.map(
        _map_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing SFT dataset",
    )

    # Remove degenerate samples with no supervised target token.
    def _has_valid_labels(ex: Dict[str, List[int]]) -> bool:
        return any(v != IGNORE_INDEX for v in ex["labels"])

    dataset = dataset.filter(_has_valid_labels, num_proc=num_proc, desc="Filtering empty-label samples")
    return dataset


def build_full_text_dataset(
    raw_dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    instruction_field: str,
    response_field: str,
    max_seq_length: int,
    prompt_style: str,
    num_proc: int = 1,
) -> Dataset:
    def _map_fn(examples: Dict[str, Sequence[str]]) -> Dict[str, List[List[int]]]:
        all_input_ids = []
        instructions = examples[instruction_field]
        responses = examples[response_field]
        for instruction, response in zip(instructions, responses):
            ex = _build_full_text_example(
                instruction=str(instruction),
                response=str(response),
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                prompt_style=prompt_style,
            )
            all_input_ids.append(ex["input_ids"])
        return {"input_ids": all_input_ids}

    dataset = raw_dataset.map(
        _map_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing full-text dataset",
    )
    return dataset


def attach_sample_id(dataset: Dataset) -> Dataset:
    def _add_idx(_: Dict[str, List[int]], idx: int) -> Dict[str, int]:
        return {"sample_id": idx}

    return dataset.map(_add_idx, with_indices=True, desc="Attaching sample_id")


def save_subset_as_jsonl(
    raw_dataset: Dataset,
    indices: Sequence[int],
    output_path: str,
    instruction_field: str,
    response_field: str,
) -> None:
    subset = raw_dataset.select(list(indices))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in subset:
            obj = {
                instruction_field: row[instruction_field],
                response_field: row[response_field],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class RawTextDataset(TorchDataset):
    """
    Lazy text dataset for selection stage.
    Builds prompt+response text on-the-fly, avoiding expensive full-dataset tokenization upfront.
    """

    def __init__(
        self,
        raw_dataset: Dataset,
        instruction_field: str,
        response_field: str,
        prompt_style: str,
        eos_token: str,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.instruction_field = instruction_field
        self.response_field = response_field
        self.prompt_style = prompt_style
        self.eos_token = eos_token

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.raw_dataset[int(idx)]
        instruction = str(row[self.instruction_field])
        response = str(row[self.response_field])
        text = build_source_prompt(instruction, self.prompt_style) + f"{response}{self.eos_token}"
        return {
            "text": text,
            "sample_id": int(idx),
        }


def build_raw_text_dataset(
    raw_dataset: Dataset,
    instruction_field: str,
    response_field: str,
    prompt_style: str,
    eos_token: str,
) -> RawTextDataset:
    return RawTextDataset(
        raw_dataset=raw_dataset,
        instruction_field=instruction_field,
        response_field=response_field,
        prompt_style=prompt_style,
        eos_token=eos_token,
    )


@dataclass
class CausalLMCollator:
    tokenizer: transformers.PreTrainedTokenizer
    include_labels: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        if self.include_labels:
            labels = [torch.tensor(x["labels"], dtype=torch.long) for x in instances]
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )
            batch["labels"] = labels

        if "sample_id" in instances[0]:
            batch["sample_id"] = torch.tensor([int(x["sample_id"]) for x in instances], dtype=torch.long)

        return batch


@dataclass
class TokenizingCausalLMCollator:
    """
    Batch tokenization collator for selection stage.
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_seq_length: int
    prepend_bos: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        texts = [str(x["text"]) for x in instances]
        bos_token_id = _get_bos_token_id(self.tokenizer) if self.prepend_bos else None
        text_max_length = self.max_seq_length
        if bos_token_id is not None:
            text_max_length = max(1, self.max_seq_length - 1)

        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=text_max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if bos_token_id is not None:
            bos_ids = torch.full((input_ids.size(0), 1), bos_token_id, dtype=input_ids.dtype)
            bos_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype)
            input_ids = torch.cat([bos_ids, input_ids], dim=1)
            attention_mask = torch.cat([bos_mask, attention_mask], dim=1)
            if input_ids.size(1) > self.max_seq_length:
                input_ids = input_ids[:, : self.max_seq_length]
                attention_mask = attention_mask[:, : self.max_seq_length]

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "sample_id" in instances[0]:
            batch["sample_id"] = torch.tensor([int(x["sample_id"]) for x in instances], dtype=torch.long)
        return batch
