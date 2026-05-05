#!/bin/bash


# ================= Global configuration =================
OUTPUT_ROOT="results/metamathqa"
export HF_DATASETS_OFFLINE=0


# GPU and environment
export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL="1"
BATCH_SIZE=32

# ================= Model configuration array =================
MODELS_CONFIG=(
    # Format: "model_name|original_base_model_path|fine_tuned_model_path"
)

# ================= Script logic =================


# Process each model
for config in "${MODELS_CONFIG[@]}"; do
    IFS='|' read -r MODEL_NAME ORIGINAL_BASE_MODEL EVAL_MODEL_DIR <<< "$config"
    CURRENT_OUTPUT_PATH="$OUTPUT_ROOT/$MODEL_NAME/$EVAL_MODEL_DIR"
    
    echo "#########################################################"
    echo "Processing model: $MODEL_NAME"
    echo "---------------------------------------------------------"
    echo "Evaluation weights: $EVAL_MODEL_DIR"
    echo "Results directory: $CURRENT_OUTPUT_PATH"
    echo "#########################################################"

    # --- Step 0: Fix tokenizer files ---
    echo "[Step 0] Fixing tokenizer files..."
    if [ -d "$ORIGINAL_BASE_MODEL" ] && [ -d "$EVAL_MODEL_DIR" ]; then
        cp -f "$ORIGINAL_BASE_MODEL"/tokenizer* "$EVAL_MODEL_DIR/" 2>/dev/null
        cp -f "$ORIGINAL_BASE_MODEL"/special_tokens_map.json "$EVAL_MODEL_DIR/" 2>/dev/null
        cp -f "$ORIGINAL_BASE_MODEL"/*.model "$EVAL_MODEL_DIR/" 2>/dev/null 
        echo "Tokenizer files synced."
    else
        echo "Error: one or more paths do not exist. Skipping this model."
        continue
    fi

    # --- Step 0.5: Inject hack parameters ---
    # Passing these parameters through the command line can fail, so they are
    # written directly into config.json.
    # The "dtype" field is also injected to use transformers' argument cleanup
    # behavior and avoid runtime errors.
    echo "[Step 0.5] Applying config.json patch (hack fix)..."
    python3 -c "
import json
import os
config_path = '$EVAL_MODEL_DIR/config.json'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    modified = False
    
    # 1. Ensure torch_dtype exists
    if data.get('torch_dtype') != 'bfloat16':
        data['torch_dtype'] = 'bfloat16'
        modified = True
        print('   -> Set torch_dtype: bfloat16')

    # 2. Inject dtype. This is the key hack for avoiding the error.
    if data.get('dtype') != 'bfloat16':
        data['dtype'] = 'bfloat16'
        modified = True
        print('   -> Injected dtype: bfloat16 (hack)')

    if modified:
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        print('   config.json updated.')
    else:
        print('   No changes needed.')
"

    mkdir -p "$CURRENT_OUTPUT_PATH"

    # --- Step 1: Define common arguments ---
    # Important: do not add torch_dtype=... on the command line, or it will fail.
    # Precision has already been written into config.json and will be loaded by
    # the model automatically.
    # MODEL_ARGS="pretrained=$EVAL_MODEL_DIR,trust_remote_code=True"
    MODEL_ARGS="pretrained=$ORIGINAL_BASE_MODEL,peft=$EVAL_MODEL_DIR,trust_remote_code=True"

    # --- Step 2: Run grouped evaluations ---

    # 5. HumanEval
    echo ">>> [$MODEL_NAME] Running HumanEval..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks humaneval \
        --num_fewshot 0 \
        --batch_size 32 \
        --output_path "$CURRENT_OUTPUT_PATH/0shot_humaneval" \
        --device cuda \
        --confirm_run_unsafe_code

    # 1. GSM8K
    echo ">>> [$MODEL_NAME] Running GSM8K..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 32 \
        --output_path "$CURRENT_OUTPUT_PATH/5shot_gsm8k" \
        --device cuda
    

    # 2. MMLU
    echo ">>> [$MODEL_NAME] Running MMLU..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 2 \
        --output_path "$CURRENT_OUTPUT_PATH/5shot_mmlu" \
        --device cuda


    # 4. Commonsense (0-shot)
    echo ">>> [$MODEL_NAME] Running Commonsense Tasks..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks hellaswag,openbookqa \
        --num_fewshot 0 \
        --batch_size 32 \
        --output_path "$CURRENT_OUTPUT_PATH/0shot_commonsense" \
        --device cuda \
        --trust_remote_code


    echo "Model $MODEL_NAME evaluation complete."
    echo ""
done

echo "========================================================="
echo "All model evaluations are complete. Please check $OUTPUT_ROOT"
