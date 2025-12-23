#!/usr/bin/env bash
# End-to-end helper for EdgeWash: install deps, fetch data, train, launch TensorBoard and Streamlit demo.
set -euo pipefail

DATASET="kaggle"           # kaggle|pskus|metc
ARCH="frames"             # frames|videos|merged
WITH_FLOW=0                # compute optical flow (required for merged)
LOG_DIR="runs/handwash"   # TensorBoard logdir
TB_PORT=6006
STREAMLIT_PORT=8501
VENV_DIR=".venv"

usage() {
  cat <<EOF
Usage: $0 [--dataset kaggle|pskus|metc] [--arch frames|videos|merged] [--with-flow] [--logdir PATH] [--tb-port N] [--streamlit-port N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2;;
    --arch)
      ARCH="$2"; shift 2;;
    --with-flow)
      WITH_FLOW=1; shift 1;;
    --logdir)
      LOG_DIR="$2"; shift 2;;
    --tb-port)
      TB_PORT="$2"; shift 2;;
    --streamlit-port)
      STREAMLIT_PORT="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option $1"; usage; exit 1;;
  esac
done

case "$DATASET" in
  kaggle) FETCH_SCRIPT="dataset-kaggle/get-and-preprocess-dataset.sh"; TRAIN_PREFIX="kaggle";;
  pskus)  FETCH_SCRIPT="dataset-pskus/get-and-preprocess-dataset.sh"; TRAIN_PREFIX="pskus";;
  metc)   FETCH_SCRIPT="dataset-metc/get-and-preprocess-dataset.sh"; TRAIN_PREFIX="rsu-metc";;
  *) echo "Unsupported dataset $DATASET"; exit 1;;
endcase

case "$ARCH" in
  frames) TRAIN_SCRIPT="${TRAIN_PREFIX}-classify-frames.py"; RUN_NAME="${TRAIN_PREFIX}-single-frame";;
  videos) TRAIN_SCRIPT="${TRAIN_PREFIX}-classify-videos.py"; RUN_NAME="${TRAIN_PREFIX}-videos";;
  merged) TRAIN_SCRIPT="${TRAIN_PREFIX}-classify-merged-network.py"; RUN_NAME="${TRAIN_PREFIX}-merged";;
  *) echo "Unsupported architecture $ARCH"; exit 1;;
endcase

if [[ "$ARCH" == "merged" ]]; then
  WITH_FLOW=1
fi

# 1) Environment and dependencies
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install streamlit matplotlib

# 2) Data acquisition and preprocessing
if [[ ! -d "dataset-${DATASET}" ]]; then
  echo "Missing dataset-${DATASET} folder; please ensure datasets directory is present."
fi
if [[ -x "$FETCH_SCRIPT" ]]; then
  bash "$FETCH_SCRIPT"
else
  echo "Fetch script $FETCH_SCRIPT not found or not executable" && exit 1
fi

# 3) Optional optical flow
if [[ $WITH_FLOW -eq 1 ]]; then
  FLOW_TARGET=$(find "dataset-${DATASET}" -maxdepth 2 -type d -name '*preprocessed*' | head -n1)
  if [[ -z "$FLOW_TARGET" ]]; then
    echo "Could not locate a preprocessed dataset folder inside dataset-${DATASET}."
    exit 1
  fi
  python calculate-optical-flow.py "$FLOW_TARGET"
fi

# 4) Training + evaluation with TensorBoard logging
mkdir -p "$LOG_DIR"
export HANDWASH_TENSORBOARD_LOGDIR="$LOG_DIR"
python "$TRAIN_SCRIPT"
TRAINED_MODEL_PATH="${RUN_NAME}final-model"

# 5) Launch TensorBoard
if command -v tensorboard >/dev/null 2>&1; then
  tensorboard --logdir "$LOG_DIR" --port "$TB_PORT" --host 0.0.0.0 &
  echo "TensorBoard running on http://localhost:${TB_PORT}"
else
  echo "tensorboard executable not found; install TensorFlow with TensorBoard support."
fi

# 6) Launch Streamlit demo
export HANDWASH_INFERENCE_MODEL="$TRAINED_MODEL_PATH"
export HANDWASH_USE_MERGED=$WITH_FLOW
streamlit run tools/edgewash_streamlit_app.py --server.address 0.0.0.0 --server.port "$STREAMLIT_PORT"
