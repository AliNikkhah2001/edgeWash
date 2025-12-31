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
BATCH_MOBILENET=${BATCH_MOBILENET:-32}
BATCH_SEQUENCE=${BATCH_SEQUENCE:-16}
TB_PORT=${TB_PORT:-6006}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-false}
SKIP_PREPROCESS=${SKIP_PREPROCESS:-false}
SKIP_TRAIN=${SKIP_TRAIN:-false}
SKIP_EVAL=${SKIP_EVAL:-false}
RESUME_FROM=${RESUME_FROM:-}
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

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU status:"
  nvidia-smi || true
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

if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
  log "Dataset download step..."
  DATASETS_ENV="$DATASETS" $PYTHON - <<'PY'
import os, sys
import download_datasets as dl
datasets = {d.strip().lower() for chunk in os.environ.get("DATASETS_ENV","kaggle").replace(",", " ").split() for d in [chunk] if d.strip()}
if not datasets:
    datasets = {"kaggle"}
print(f"Datasets requested: {sorted(datasets)}")
ok = True
if "kaggle" in datasets:
    ok = dl.download_kaggle_dataset() and ok
if "pskus" in datasets:
    ok = dl.download_pskus_dataset() and ok
if "metc" in datasets:
    ok = dl.download_metc_dataset() and ok
status = dl.verify_datasets()
print("Dataset status:")
for name, info in status.items():
    icon = "✓" if info["exists"] else "✗"
    print(f"  {icon} {info['name']}: {info['num_files']} files at {info['path']}")
if not ok:
    sys.exit("Dataset download failed; see logs above.")
PY
else
  log "Skipping dataset download (SKIP_DOWNLOAD=true)"
fi

if [[ "$SKIP_PREPROCESS" == "false" ]]; then
  log "Preprocessing datasets..."
  DATASETS_ENV="$DATASETS" $PYTHON - <<'PY'
import os, sys
from pathlib import Path
import preprocess_data
datasets = {d.strip().lower() for chunk in os.environ.get("DATASETS_ENV","kaggle").replace(",", " ").split() for d in [chunk] if d.strip()}
use_kaggle = "kaggle" in datasets
use_pskus = "pskus" in datasets
use_metc = "metc" in datasets
result = preprocess_data.preprocess_all_datasets(use_kaggle=use_kaggle, use_pskus=use_pskus, use_metc=use_metc)
if not result:
    sys.exit("Preprocessing failed")
print("Preprocessed files:")
for k, v in result.items():
    print(f"  {k}: {v}")
PY
else
  log "Skipping preprocessing (SKIP_PREPROCESS=true)"
fi

if [[ "$SKIP_TRAIN" == "false" ]]; then
  for MODEL in $TRAIN_MODELS; do
    MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
    case "$MODEL_LOWER" in
      mobilenetv2) BATCH="$BATCH_MOBILENET" ;;
      lstm|gru) BATCH="$BATCH_SEQUENCE" ;;
      *) log "Unknown model: $MODEL_LOWER"; exit 1 ;;
    esac
    log "Training $MODEL_LOWER (epochs=$EPOCHS batch=$BATCH lr=$LEARNING_RATE)..."
    MODEL_TYPE="$MODEL_LOWER" BATCH_SIZE="$BATCH" EPOCHS="$EPOCHS" LR="$LEARNING_RATE" RESUME_FROM_ENV="$RESUME_FROM" $PYTHON - <<'PY'
import os, json
from pathlib import Path
import train
model_type = os.environ["MODEL_TYPE"]
train_csv = Path("datasets/processed/train.csv")
val_csv = Path("datasets/processed/val.csv")
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
    "model_type": model_type,
    "final_model": result["final_model_path"],
    "best_epoch": int(result["best_epoch"]) + 1,
    "best_val_acc": float(result["history"]["val_accuracy"][result["best_epoch"]]),
    "best_val_loss": float(result["history"]["val_loss"][result["best_epoch"]])
}, indent=2))
PY
    LAST_CKPT=$(ls -dt "$PROJECT_ROOT/checkpoints/${MODEL_LOWER}_"* 2>/dev/null | head -n1 || true)
    if [[ -n "$LAST_CKPT" ]]; then
      log "Latest checkpoint dir for $MODEL_LOWER: $LAST_CKPT"
    else
      log "No checkpoint folder found for $MODEL_LOWER (check training logs)."
    fi
  done
else
  log "Skipping training (SKIP_TRAIN=true)"
fi

if [[ "$SKIP_EVAL" == "false" ]]; then
  for MODEL in $TRAIN_MODELS; do
    MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
    case "$MODEL_LOWER" in
      mobilenetv2) BATCH="$BATCH_MOBILENET" ;;
      lstm|gru) BATCH="$BATCH_SEQUENCE" ;;
      *) log "Unknown model: $MODEL_LOWER"; exit 1 ;;
    esac
    FINAL_MODEL_PATH=$(ls "$PROJECT_ROOT/models/${MODEL_LOWER}_final.keras" 2>/dev/null | head -n1 || true)
    if [[ -z "$FINAL_MODEL_PATH" ]]; then
      log "Final model for $MODEL_LOWER not found; skipping evaluation."
      continue
    fi
    log "Evaluating $MODEL_LOWER using $FINAL_MODEL_PATH ..."
    MODEL_TYPE="$MODEL_LOWER" MODEL_PATH="$FINAL_MODEL_PATH" BATCH_SIZE="$BATCH" $PYTHON - <<'PY'
import os, json
from pathlib import Path
import evaluate
model_type = os.environ["MODEL_TYPE"]
model_path = os.environ["MODEL_PATH"]
test_csv = Path("datasets/processed/test.csv")
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
print(json.dumps(summary, indent=2))
PY
  done
else
  log "Skipping evaluation (SKIP_EVAL=true)"
fi

log "Pipeline completed. Review $LOG_FILE for full details."
