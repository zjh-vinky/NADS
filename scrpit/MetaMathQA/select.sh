export CUDA_VISIBLE_DEVICES=1,2



num_gpus=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c) + 1 ))



torchrun --master_port 29504 --nproc_per_node=$num_gpus NADS-main/select_data.py \
  --model_name_or_path ./models \
  --navigator_ckpt ./navigator_ckpt \
  --candidate_data_path dataset \
  --candidate_instruction_field instruction \
  --candidate_response_field response \
  --output_dir nads_results \
  --stage2_batch_size 2 \
  --num_workers 10 \
  --prefetch_factor 4 \
  --max_seq_length 1024 \
  --nads_select_k 15000 \
  --attn_implementation sdpa \
  --disable_attn_fallback \
  --dtype bfloat16 \
  --torch_compile