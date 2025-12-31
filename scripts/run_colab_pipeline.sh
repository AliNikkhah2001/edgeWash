#!/usr/bin/env bash

# Colab-friendly end-to-end pipeline for the handwashing project.
# - Sets up (optional) virtualenv and installs dependencies
# - Downloads datasets, preprocesses them, trains models, and evaluates
# - Starts TensorBoard for live monitoring

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/pipeline"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/colab_pipeline_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

log "Project root: $PROJECT_ROOT"
log "Pipeline log: $LOG_FILE"

PYTHON=${PYTHON:-python3}
USE_VENV=${USE_VENV:-false}
VENV_DIR=${VENV_DIR:-$PROJECT_ROOT/.venv-handwash}
DATASETS=${DATASETS:-kaggle}               # comma or space separated: kaggle,pskus,metc
TRAIN_MODELS=${TRAIN_MODELS:-"mobilenetv2 lstm gru"}
EPOCHS=${EPOCHS:-5}                        # bump to 20-50 for full runs
LEARNING_RATE=${LEARNING_RATE:-1e-4}
BATCH_MOBILENET=${BATCH_MOBILENET:-64}
BATCH_SEQUENCE=${BATCH_SEQUENCE:-32}
AUTO_TUNE_BATCH=${AUTO_TUNE_BATCH:-true}
TB_PORT=${TB_PORT:-6008}
SANITY_CHECK=${SANITY_CHECK:-true}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-false}
SKIP_PREPROCESS=${SKIP_PREPROCESS:-false}
SKIP_TRAIN=${SKIP_TRAIN:-false}
SKIP_EVAL=${SKIP_EVAL:-false}
RESUME_FROM=${RESUME_FROM:-}
TRAIN_CSV_OVERRIDE=${TRAIN_CSV_OVERRIDE:-}
VAL_CSV_OVERRIDE=${VAL_CSV_OVERRIDE:-}
TEST_CSV_OVERRIDE=${TEST_CSV_OVERRIDE:-}
USE_EXISTING_PROCESSED=${USE_EXISTING_PROCESSED:-true}
SKIP_DOWNLOAD_IF_PRESENT=${SKIP_DOWNLOAD_IF_PRESENT:-true}
TB_PID=""

if [[ ! -d "$PROJECT_ROOT/training" ]]; then
  log "Expected training/ directory under $PROJECT_ROOT"
  exit 1
fi

IN_COLAB=$($PYTHON - <<'PY'
import importlib.util
print("true" if importlib.util.find_spec("google.colab") else "false")
PY
)
log "Running in Colab: $IN_COLAB"

if [[ "$USE_VENV" == "true" ]]; then
  log "Creating virtualenv at $VENV_DIR"
  $PYTHON -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  PYTHON="$VENV_DIR/bin/python"
fi

log "Python binary: $PYTHON"
log "Config: datasets=$DATASETS models=$TRAIN_MODELS epochs=$EPOCHS lr=$LEARNING_RATE"

# Dependency install (TensorFlow is expected to be preinstalled on Colab)
$PYTHON -m pip install --no-cache-dir -U pip setuptools wheel
REQS=(scikit-learn pandas numpy opencv-python-headless matplotlib seaborn tqdm requests nbformat)
$PYTHON -m pip install --no-cache-dir -U "${REQS[@]}"

GPU_MEM_MB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU status:"
  nvidia-smi || true
  GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
else
  log "nvidia-smi not available; GPU may be missing."
fi

$PYTHON - <<'PY'
import tensorflow as tf, cv2, sklearn, numpy, pandas
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs visible to TF: {tf.config.list_physical_devices('GPU')}")
print(f"OpenCV: {cv2.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
PY

if [[ "$AUTO_TUNE_BATCH" == "true" && "$GPU_MEM_MB" -gt 0 ]]; then
  BATCH_MOBILENET=$(( GPU_MEM_MB / 90 ))
  BATCH_SEQUENCE=$(( GPU_MEM_MB / 180 ))
  (( BATCH_MOBILENET < 64 )) && BATCH_MOBILENET=64
  (( BATCH_MOBILENET > 256 )) && BATCH_MOBILENET=256
  (( BATCH_SEQUENCE < 32 )) && BATCH_SEQUENCE=32
  (( BATCH_SEQUENCE > 128 )) && BATCH_SEQUENCE=128
  log "Auto-tuned batch sizes from GPU mem ${GPU_MEM_MB}MB -> frame: $BATCH_MOBILENET, sequence: $BATCH_SEQUENCE"
fi

mkdir -p "$PROJECT_ROOT"/{datasets/raw,datasets/processed,models,checkpoints,logs,results}
export PYTHONPATH="$PROJECT_ROOT/training:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=1

if command -v tensorboard >/dev/null 2>&1; then
  log "Starting TensorBoard on port $TB_PORT (logs -> $LOG_DIR/tensorboard.out)"
  if tensorboard --logdir "$PROJECT_ROOT/logs" --host 0.0.0.0 --port "$TB_PORT" --load_fast=false >"$LOG_DIR/tensorboard.out" 2>&1 & then
    TB_PID=$!
    log "TensorBoard PID: $TB_PID"
    if [[ "$IN_COLAB" == "true" ]]; then
      cat <<EOF
To view TensorBoard inside Colab, run this in a Python cell:
%load_ext tensorboard
%tensorboard --logdir /content/edgeWash/logs --host 0.0.0.0 --port $TB_PORT
or:
from google.colab import output
output.serve_kernel_port_as_window($TB_PORT)
EOF
    fi
  else
    log "TensorBoard failed to start (port in use?). Check $LOG_DIR/tensorboard.out"
  fi
else
  log "tensorboard command not found; skipping auto-start."
fi

cleanup() {
  if [[ -n "$TB_PID" ]]; then
    log "Stopping TensorBoard (PID $TB_PID)"
    kill "$TB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Processed CSV paths (override if you already have them)
TRAIN_CSV_PATH=${TRAIN_CSV_OVERRIDE:-$PROJECT_ROOT/datasets/processed/train.csv}
VAL_CSV_PATH=${VAL_CSV_OVERRIDE:-$PROJECT_ROOT/datasets/processed/val.csv}
TEST_CSV_PATH=${TEST_CSV_OVERRIDE:-$PROJECT_ROOT/datasets/processed/test.csv}

PROCESSED_READY=false
if [[ -f "$TRAIN_CSV_PATH" && -f "$VAL_CSV_PATH" && -f "$TEST_CSV_PATH" ]]; then
  PROCESSED_READY=true
fi

# Parse dataset list
DATASET_LIST=()
for ds in $(echo "$DATASETS" | tr ',' ' '); do
  ds_lower=$(echo "$ds" | tr '[:upper:]' '[:lower:]')
  if [[ -n "$ds_lower" ]]; then
    DATASET_LIST+=("$ds_lower")
  fi
done
if [[ ${#DATASET_LIST[@]} -eq 0 ]]; then
  DATASET_LIST=("kaggle")
fi

cleanup_dataset() {
  local name="$1"
  log "Cleaning up dataset artifacts for $name to save space..."
  rm -rf "$PROJECT_ROOT/datasets/raw/$name" || true
  # Remove extracted frames to free space
  find "$PROJECT_ROOT/datasets/processed" -type f -name '*.jpg' -delete || true
  # Remove empty dirs
  find "$PROJECT_ROOT/datasets/processed" -type d -empty -delete || true
}

run_train_eval() {
  local name_label="$1"
  if [[ ! -f "$TRAIN_CSV_PATH" || ! -f "$VAL_CSV_PATH" || ! -f "$TEST_CSV_PATH" ]]; then
    log "Processed CSVs missing; cannot train/eval. train=$TRAIN_CSV_PATH val=$VAL_CSV_PATH test=$TEST_CSV_PATH"
    return 1
  fi

  if [[ "$SKIP_TRAIN" == "true" ]]; then
    log "SKIP_TRAIN=true; skipping training for $name_label"
    return 0
  fi

  for MODEL in $TRAIN_MODELS; do
    MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
    case "$MODEL_LOWER" in
      mobilenetv2|resnet50|efficientnetb0) BATCH="$BATCH_MOBILENET" ;;
      3d_cnn) BATCH=12 ;;
      lstm|gru) BATCH="$BATCH_SEQUENCE" ;;
      *) log "Unknown model: $MODEL_LOWER"; return 1 ;;
    esac
    log "Training $MODEL_LOWER on $name_label (epochs=$EPOCHS batch=$BATCH lr=$LEARNING_RATE)..."
    MODEL_TYPE="$MODEL_LOWER" BATCH_SIZE="$BATCH" EPOCHS="$EPOCHS" LR="$LEARNING_RATE" RESUME_FROM_ENV="$RESUME_FROM" \
    TRAIN_CSV_ENV="$TRAIN_CSV_PATH" VAL_CSV_ENV="$VAL_CSV_PATH" $PYTHON - <<'PY'
import os, json
from pathlib import Path
import train
model_type = os.environ["MODEL_TYPE"]
train_csv = Path(os.environ["TRAIN_CSV_ENV"])
val_csv = Path(os.environ["VAL_CSV_ENV"])
resume = os.environ.get("RESUME_FROM_ENV") or None
if not train_csv.exists() or not val_csv.exists():
    raise SystemExit("Missing processed CSVs; run preprocessing first.")
result = train.train_model(
    model_type=model_type,
    train_csv=train_csv,
    val_csv=val_csv,
    batch_size=int(os.environ["BATCH_SIZE"]),
    epochs=int(os.environ["EPOCHS"]),
    learning_rate=float(os.environ["LR"]),
    resume_from=resume
)
print(json.dumps({
    "dataset": os.environ.get("DATASET_NAME", "existing"),
    "model_type": model_type,
    "final_model": result["final_model_path"],
    "best_epoch": int(result["best_epoch"]) + 1,
    "best_val_acc": float(result["history"]["val_accuracy"][result["best_epoch"]]),
    "best_val_loss": float(result["history"]["val_loss"][result["best_epoch"]])
}, indent=2))
PY

    FINAL_MODEL_PATH=$(ls "$PROJECT_ROOT/models/${MODEL_LOWER}_final.keras" 2>/dev/null | head -n1 || true)
    if [[ -z "$FINAL_MODEL_PATH" ]]; then
      log "Final model for $MODEL_LOWER not found; skipping evaluation."
      continue
    fi
    if [[ "$SKIP_EVAL" == "true" ]]; then
      log "SKIP_EVAL=true; skipping evaluation for $MODEL_LOWER"
      continue
    fi
    log "Evaluating $MODEL_LOWER on $name_label ..."
    MODEL_TYPE="$MODEL_LOWER" MODEL_PATH="$FINAL_MODEL_PATH" BATCH_SIZE="$BATCH" TEST_CSV_ENV="$TEST_CSV_PATH" $PYTHON - <<'PY'
import os, json
from pathlib import Path
import evaluate
model_type = os.environ["MODEL_TYPE"]
model_path = os.environ["MODEL_PATH"]
test_csv = Path(os.environ["TEST_CSV_ENV"])
if not test_csv.exists():
    raise SystemExit("Missing test.csv; run preprocessing.")
results = evaluate.evaluate_model(
    model_path=model_path,
    test_csv=test_csv,
    model_type=model_type,
    batch_size=int(os.environ["BATCH_SIZE"]),
    save_results=True
)
summary = {k: v for k, v in results.items() if isinstance(v, (float, int, str))}
summary["dataset"] = os.environ.get("DATASET_NAME", "existing")
print(json.dumps(summary, indent=2))
PY
  done
}

TRAIN_EVAL_RAN=false

process_dataset() {
  local name="$1"
  log "==== Processing dataset: $name ===="

  # If we already have processed CSVs and reuse is allowed, skip download/preprocess
  if [[ "$USE_EXISTING_PROCESSED" == "true" && "$PROCESSED_READY" == "true" ]]; then
    log "Processed CSVs already present (train/val/test). Skipping download/preprocess for $name."
    run_train_eval "$name" && TRAIN_EVAL_RAN=true
    return
  fi

  if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    if [[ "$SKIP_DOWNLOAD_IF_PRESENT" == "true" && -d "$PROJECT_ROOT/datasets/raw/$name" ]]; then
      log "Raw data for $name already present; skipping download."
    else
      log "Downloading $name ..."
    DATASET_NAME="$name" $PYTHON - <<'PY'
import os, sys
import download_datasets as dl
name = os.environ["DATASET_NAME"]
if name == "kaggle":
    ok = dl.download_kaggle_dataset()
elif name == "pskus":
    ok = dl.download_pskus_dataset()
elif name == "metc":
    ok = dl.download_metc_dataset()
elif name == "synthetic_blender_rozakar":
    ok = dl.download_synthetic_blender_rozakar()
else:
    raise SystemExit(f"Unknown dataset {name}")
if not ok:
    sys.exit(f"Download failed for {name}")
status = dl.verify_datasets()
info = status.get(name, {})
print(f"{name}: exists={info.get('exists')} files={info.get('num_files')} path={info.get('path')}")
PY
    fi
  else
    log "Skipping download (SKIP_DOWNLOAD=true)"
  fi

  if [[ "$SKIP_PREPROCESS" == "false" ]]; then
    log "Preprocessing $name ..."
    DATASET_NAME="$name" $PYTHON - <<'PY'
import os, sys
from pathlib import Path
import preprocess_data
name = os.environ["DATASET_NAME"]
use_kaggle = name == "kaggle"
use_pskus = name == "pskus"
use_metc = name == "metc"
result = preprocess_data.preprocess_all_datasets(use_kaggle=use_kaggle, use_pskus=use_pskus, use_metc=use_metc)
if not result:
    sys.exit(f"Preprocessing failed for {name}")
print("Preprocessed files:")
for k, v in result.items():
    print(f"  {k}: {v}")
PY
  else
    log "Skipping preprocessing (SKIP_PREPROCESS=true)"
  fi

  if [[ "$SANITY_CHECK" == "true" ]]; then
    log "Sanity check: class distribution"
    TRAIN_CSV_PATH="$TRAIN_CSV_PATH" $PYTHON - <<'PY'
import os
import pandas as pd
from pathlib import Path
train_csv = Path(os.environ["TRAIN_CSV_PATH"])
if not train_csv.exists():
    print("Missing train.csv; skipping sanity check")
    raise SystemExit(0)
df = pd.read_csv(train_csv)
col = "class_name" if "class_name" in df.columns else "class_id"
counts = df[col].value_counts()
print(counts.to_string())
if len(counts) == 1:
    print("WARNING: single-class dataset; check labeling.")
if "Other" in counts.index:
    if counts["Other"] / len(df) > 0.95:
        print("WARNING: 'Other' exceeds 95% of samples.")
PY
  fi

  run_train_eval "$name" && TRAIN_EVAL_RAN=true
  cleanup_dataset "$name"
  log "==== Finished dataset: $name ===="
}

for ds in "${DATASET_LIST[@]}"; do
  process_dataset "$ds"
done

if [[ "$TRAIN_EVAL_RAN" == "false" && "$USE_EXISTING_PROCESSED" == "true" && "$PROCESSED_READY" == "true" ]]; then
  log "Existing processed CSVs detected; running train/eval without download/preprocess."
  run_train_eval "existing_processed"
fi

log "Pipeline completed. Review $LOG_FILE for full details."
