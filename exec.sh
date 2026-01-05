#!/usr/bin/env bash
set -e

# --- 0. Activate Virtual Environment ---
if [[ -f ".venv/bin/activate" ]]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "Warning: .venv/bin/activate not found. Using system Python."
fi

export PYTHONUNBUFFERED=1
CFG_FILE="$1"

if [[ -z "$CFG_FILE" ]]; then
    echo "Usage: ./exec.sh <path_to_config_yaml>"
    exit 1
fi

# --- 1. Parse Config for Paths ---
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

# 1. Run nohup in background (&) so the script doesn't hang immediately
nohup python train.py --c "$CFG_FILE" >> nohup.out 2>&1 &
TRAIN_PID=$!  # Save the Process ID of training

echo "Training running with PID: $TRAIN_PID"
echo "Streaming logs from nohup.out (Ctrl+C will stop tailing, but NOT the training)..."
echo "------------------------------------------------"

# 2. Start tail in background
tail -f nohup.out &
TAIL_PID=$!   # Save the Process ID of tail

# 3. Wait for training to finish
# If training crashes or finishes, wait will release.
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# 4. Kill the tail process so we stop watching logs
kill $TAIL_PID

# 5. Check if training was successful
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Error: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

# --- 3. Run Evaluation ---
echo ""
echo "================================================"
echo " STEP 2: Starting Evaluation on Target Test Set"
echo " Model: $MODEL_PATH"
echo " Time: $(date)"
echo "================================================"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Best model not found at $MODEL_PATH"
    exit 1
fi

# Same logic for Evaluation
nohup python eval.py --c "$CFG_FILE" --load_path "$MODEL_PATH" >> nohup.out 2>&1 &
EVAL_PID=$!

echo "Evaluation running with PID: $EVAL_PID"
echo "Streaming logs..."

tail -f nohup.out &
TAIL_PID=$!

wait $EVAL_PID
EVAL_EXIT_CODE=$?

kill $TAIL_PID

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "Error: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo "================================================"
echo " DONE: Pipeline Finished Successfully"
echo " Time: $(date)"
echo "================================================"
