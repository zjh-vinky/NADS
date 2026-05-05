#!/bin/sh
set -eu

# Specify visible GPUs. Use a single GPU ID or comma-separated GPU IDs.
export CUDA_VISIBLE_DEVICES='0,1'
num_gpus=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c | tr -d ' ') + 1 ))

# ===== Basic configuration ===== #
model_name="llama-3_2-3b"
base_model="../models/$model_name"
finetune_type="full" # full/lora

# ===== Data configuration ===== #
dataset_name="MetaMathQA"
new_dataset="datasets/$dataset_name"
candidate_dataset="datasets/Magpie-Qwen2.5-Pro-300K-Filtered"
selected_indices_path="./nads_results/llama_math_wight/selected_indices.json"
split="train"
new_field_input="query"
new_field_output="response"
candidate_field_input="instruction"
candidate_field_output="response"

# Whether to split a validation set from the new task data.
use_validation=false # true/false

# ===== Training configuration ===== #
epochs=3
lr=1e-4
batch_size=1
max_seq_len=1024
gradient_accumulation_steps=1
num_proc=32
num_workers=8
log_every_n_steps=40

# ===== LoRA configuration ===== #
lora_r=16
lora_alpha=32
lora_dropout=0.05

# ===== Experiment configuration ===== #
base_dir="./nads_results"
global_bs=$((batch_size * gradient_accumulation_steps * num_gpus))
run_name="stage3-${finetune_type}-${model_name}-${dataset_name}-${lr}-bs${global_bs}-val-${use_validation}"
output_dir="${base_dir}/${run_name}"

if [ "$use_validation" = "true" ]; then
  val_ratio="0.01"
else
  val_ratio="0.0"
fi

torchrun --master-port 29505 --nproc_per_node="$num_gpus" NADS-main/train_arithmetic.py \
  --skip_stage1 \
  --stage1_backend native \
  --stage3_backend native \
  --model_name_or_path "$base_model" \
  --new_data_path "$new_dataset" \
  --new_dataset_split "$split" \
  --new_instruction_field "$new_field_input" \
  --new_response_field "$new_field_output" \
  --candidate_data_path "$candidate_dataset" \
  --candidate_dataset_split "$split" \
  --candidate_instruction_field "$candidate_field_input" \
  --candidate_response_field "$candidate_field_output" \
  --selected_indices_path "$selected_indices_path" \
  --output_dir "$output_dir" \
  --stage3_finetune_type "$finetune_type" \
  --stage3_task_batch_size "$batch_size" \
  --stage3_kd_batch_size "$batch_size" \
  --stage3_epochs "$epochs" \
  --stage3_lr "$lr" \
  --stage3_grad_accum_steps "$gradient_accumulation_steps" \
  --num_proc "$num_proc" \
  --num_workers "$num_workers" \
  --log_every_n_steps "$log_every_n_steps" \
  --max_seq_length "$max_seq_len" \
  --val_ratio "$val_ratio" \
  --device "cuda" \
  --lora_r "$lora_r" \
  --lora_alpha "$lora_alpha" \
  --lora_dropout "$lora_dropout" \
  --distill_lambda 5

echo "=================================================="
echo "$finetune_type training finished!"
echo "check checkpoints in: $output_dir/..."
echo "=================================================="
