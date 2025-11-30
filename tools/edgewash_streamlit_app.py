"""Minimal Streamlit interface for EdgeWash models."""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from classify_dataset import IMG_SHAPE, MobileNetPreprocessingLayer

# Environment-controlled defaults
DEFAULT_MODEL_PATH = os.getenv("HANDWASH_INFERENCE_MODEL", "kaggle-single-framefinal-model")
MODEL_NAME = os.getenv("HANDWASH_NN", "MobileNetV2")
USE_MERGED = os.getenv("HANDWASH_USE_MERGED", "0") == "1"
FRAME_STEP = int(os.getenv("HANDWASH_FRAME_STEP", "2"))
CLASS_NAMES = ["Other", "Palm to palm", "Right over left", "Left over right",
               "Interlaced", "Backs of fingers", "Thumbs"]


def resize_frame(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (IMG_SHAPE[1], IMG_SHAPE[0]))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def load_video_frames(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return frames


def compute_optical_flow(frames: List[np.ndarray], step: int) -> List[np.ndarray]:
    if len(frames) < 2:
        return []
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    flow_frames: List[np.ndarray] = []
    for i in range(0, len(gray_frames) - step, step):
        flow = cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i + step],
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = np.zeros((gray_frames[i].shape[0], gray_frames[i].shape[1], 3), dtype=np.float32)
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 1] = 255
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
        flow_frames.append(colored)
    return flow_frames


def prepare_batches(frames: List[np.ndarray], flow_frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rgb_resized = np.array([resize_frame(f) for f in frames])
    if not len(flow_frames):
        return rgb_resized, np.empty((0,))
    clipped = flow_frames[: len(rgb_resized)]
    of_resized = np.array([cv2.resize(f, (IMG_SHAPE[1], IMG_SHAPE[0])) for f in clipped])
    return rgb_resized, of_resized


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> tf.keras.Model:
    custom_objects = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    model.compile(metrics=["accuracy"])
    return model


def predict_for_video(model: tf.keras.Model, rgb_batch: np.ndarray, flow_batch: np.ndarray):
    if USE_MERGED and flow_batch.size:
        preds = model.predict([rgb_batch, flow_batch], verbose=0)
    else:
        preds = model.predict(rgb_batch, verbose=0)
    mean_pred = np.mean(preds, axis=0)
    return preds, mean_pred


def render_topk(probabilities: np.ndarray, k: int = 3):
    best_indices = probabilities.argsort()[-k:][::-1]
    for idx in best_indices:
        st.write(f"{CLASS_NAMES[idx]}: {probabilities[idx]*100:.2f}%")


def main():
    st.set_page_config(page_title="EdgeWash demo", layout="wide")
    st.title("EdgeWash hand-washing classifier")
    st.caption("Upload a short video to see model predictions. Environment variables control model selection.")

    st.sidebar.header("Runtime configuration")
    st.sidebar.write(f"Model path: `{DEFAULT_MODEL_PATH}`")
    st.sidebar.write(f"Backbone: `{MODEL_NAME}`")
    st.sidebar.write("Merged RGB+Optical flow" if USE_MERGED else "Single RGB stream")

    uploaded = st.file_uploader("Upload an .mp4 hand-washing clip", type=["mp4"])
    if not uploaded:
        st.info("Waiting for upload...")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = Path(tmp.name)

    frames = load_video_frames(video_path)
    if not frames:
        st.error("Could not read any frames from the uploaded video.")
        return

    flow_frames = compute_optical_flow(frames, FRAME_STEP) if USE_MERGED else []
    rgb_batch, flow_batch = prepare_batches(frames, flow_frames)

    model = load_model(DEFAULT_MODEL_PATH)
    per_frame, averaged = predict_for_video(model, rgb_batch, flow_batch)

    st.subheader("Aggregate prediction")
    render_topk(averaged)

    st.subheader("Frame-by-frame confidence for predicted class")
    predicted_class = int(np.argmax(averaged))
    st.line_chart({"confidence": per_frame[:, predicted_class]})

    st.success("Inference complete.")


if __name__ == "__main__":
    main()
