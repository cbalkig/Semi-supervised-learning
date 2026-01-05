#!/usr/bin/env bash
set -e

# Force Python to print logs immediately (crucial for nohup/background jobs)
export PYTHONUNBUFFERED=1

# Usage: ./exec.sh neurodomain_vegfru.yaml
CFG_FILE="$1"

if [[ -z "$CFG_FILE" ]]; then
    echo "Usage: ./exec.sh <path_to_config_yaml>"
    exit 1
fi

# --- 1. Parse Config for Paths (Python One-Liner) ---
# We extract 'save_dir' and 'save_name' to construct the path to 'model_best.pth'
read -r SAVE_DIR SAVE_NAME <<< $(python3 -c "
import yaml, sys
with open('$CFG_FILE', 'r') as f:
    cfg = yaml.safe_load(f)
    print(f\"{cfg.get('save_dir', './saved_models')} {cfg.get('save_name', 'default_run')}\")
")

MODEL_PATH="${SAVE_DIR}/${SAVE_NAME}/model_best.pth"

# --- 2. Run Training ---
echo "================================================"
echo " STEP 1: Starting Training with FreeMatch"
echo " Config: $CFG_FILE"
echo " Time: $(date)"
echo "================================================"

nohup python train.py --c "$CFG_FILE"

# --- 3. Run Evaluation ---
echo ""
echo "================================================"
echo " STEP 2: Starting Evaluation on Target Test Set"
echo " Model: $MODEL_PATH"
echo " Time: $(date)"
echo "================================================"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Best model not found at $MODEL_PATH"
    echo "Training might have failed or not produced a checkpoint."
    exit 1
fi

nohup python eval.py --c "$CFG_FILE" --load_path "$MODEL_PATH"

echo "================================================"
echo " DONE: Pipeline Finished Successfully"
echo " Time: $(date)"
echo "================================================"
