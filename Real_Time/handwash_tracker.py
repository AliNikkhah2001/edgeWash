from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


ArrayLike = np.ndarray
ThresholdValue = Union[int, float, Mapping[str, Union[int, float]]]


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    class_names: List[str]
    image_size: Tuple[int, int] = (224, 224)
    no_action_class: str = "Other"
    confidence_threshold: float = 0.0
    batch_size: int = 32


@dataclass
class FrameResult:
    user_id: str
    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: Dict[str, float]
    class_frames_done: Dict[str, int]
    class_seconds_done: Dict[str, float]
    class_completed_by_frames: Dict[str, bool]
    class_completed_by_seconds: Dict[str, bool]


@dataclass
class ClassProgress:
    frames_done: int = 0
    seconds_done: float = 0.0


@dataclass
class UserState:
    user_id: str
    seen_frames: int = 0
    per_class: Dict[str, ClassProgress] = field(default_factory=dict)


class ModelInferenceEngine:
    """Minimal model wrapper: load model and predict probabilities for frames."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.class_names = list(config.class_names)
        if config.no_action_class not in self.class_names:
            raise ValueError("no_action_class must be inside class_names")
        self.no_action_class_id = self.class_names.index(config.no_action_class)
        self.model = self._load_model(config.model_path)

    def _load_model(self, model_path: str):
        import tensorflow as tf

        try:
            return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load model. This usually means the model file was saved with a different "
                "TensorFlow/Keras version than the one installed now. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def _preprocess_frame(self, frame_bgr: ArrayLike) -> ArrayLike:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Each frame must have shape [H, W, 3] in BGR format")
        resized = cv2.resize(frame_bgr, self.config.image_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def predict_probabilities(self, frames_bgr: Sequence[ArrayLike]) -> np.ndarray:
        if len(frames_bgr) == 0:
            return np.zeros((0, len(self.class_names)), dtype=np.float32)

        batch = np.stack([self._preprocess_frame(frame) for frame in frames_bgr], axis=0)
        probs = np.asarray(
            self.model.predict(batch, batch_size=self.config.batch_size, verbose=0),
            dtype=np.float32,
        )
        if probs.ndim != 2 or probs.shape[1] != len(self.class_names):
            raise ValueError(
                f"Expected output shape [N, {len(self.class_names)}], got {tuple(probs.shape)}"
            )
        return probs

    def build_prediction(self, probs: np.ndarray) -> Dict[str, Any]:
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        if confidence < self.config.confidence_threshold:
            class_id = self.no_action_class_id
            confidence = float(probs[class_id])
        return {
            "predicted_class_id": class_id,
            "predicted_class": self.class_names[class_id],
            "confidence": confidence,
            "probabilities": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
        }

    def predict_one(self, frame_bgr: ArrayLike) -> Dict[str, Any]:
        probs = self.predict_probabilities([frame_bgr])[0]
        return self.build_prediction(probs)


class MultiUserFrameInference:
    """Call with a list of user ids and a list of frames.

    The class keeps history per user and returns frame-level outputs plus cumulative
    per-class progress in frames and seconds.
    """

    def __init__(
        self,
        inference_engine: ModelInferenceEngine,
        fps: float,
        min_frames_for_completion: ThresholdValue,
        min_seconds_for_completion: ThresholdValue,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.engine = inference_engine
        self.fps = float(fps)
        self.seconds_per_frame = 1.0 / self.fps
        self.tracked_classes = [
            name for name in self.engine.class_names if name != self.engine.config.no_action_class
        ]
        self.min_frames_for_completion = self._normalize_thresholds(min_frames_for_completion, int)
        self.min_seconds_for_completion = self._normalize_thresholds(min_seconds_for_completion, float)
        self.user_states: Dict[str, UserState] = {}

    def _normalize_thresholds(self, value: ThresholdValue, caster):
        if isinstance(value, Mapping):
            return {name: caster(value.get(name, 0)) for name in self.tracked_classes}
        return {name: caster(value) for name in self.tracked_classes}

    def _get_user_state(self, user_id: str) -> UserState:
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState(
                user_id=user_id,
                per_class={name: ClassProgress() for name in self.engine.class_names},
            )
        return self.user_states[user_id]

    def infer(self, user_ids: Sequence[Union[str, int]], frames_bgr: Sequence[ArrayLike]) -> List[FrameResult]:
        if len(user_ids) != len(frames_bgr):
            raise ValueError("user_ids and frames_bgr must have the same length")

        probs_batch = self.engine.predict_probabilities(frames_bgr)
        results: List[FrameResult] = []

        for idx, raw_user_id in enumerate(user_ids):
            user_id = str(raw_user_id)
            state = self._get_user_state(user_id)
            prediction = self.engine.build_prediction(probs_batch[idx])
            predicted_class = prediction["predicted_class"]
            state.seen_frames += 1

            if predicted_class != self.engine.config.no_action_class:
                progress = state.per_class[predicted_class]
                progress.frames_done += 1
                progress.seconds_done += self.seconds_per_frame

            frames_done = {
                name: state.per_class[name].frames_done for name in self.tracked_classes
            }
            seconds_done = {
                name: state.per_class[name].seconds_done for name in self.tracked_classes
            }
            completed_by_frames = {
                name: frames_done[name] >= self.min_frames_for_completion[name]
                for name in self.tracked_classes
            }
            completed_by_seconds = {
                name: seconds_done[name] >= self.min_seconds_for_completion[name]
                for name in self.tracked_classes
            }

            results.append(
                FrameResult(
                    user_id=user_id,
                    predicted_class=predicted_class,
                    predicted_class_id=prediction["predicted_class_id"],
                    confidence=prediction["confidence"],
                    probabilities=prediction["probabilities"],
                    class_frames_done=frames_done,
                    class_seconds_done=seconds_done,
                    class_completed_by_frames=completed_by_frames,
                    class_completed_by_seconds=completed_by_seconds,
                )
            )

        return results

    def get_user_summary(self, user_id: Union[str, int]) -> Dict[str, Any]:
        state = self._get_user_state(str(user_id))
        return {
            "user_id": state.user_id,
            "seen_frames": state.seen_frames,
            "per_class": {
                name: {
                    "frames_done": state.per_class[name].frames_done,
                    "seconds_done": state.per_class[name].seconds_done,
                    "completed_by_frames": state.per_class[name].frames_done >= self.min_frames_for_completion[name],
                    "completed_by_seconds": state.per_class[name].seconds_done >= self.min_seconds_for_completion[name],
                    "required_frames": self.min_frames_for_completion[name],
                    "required_seconds": self.min_seconds_for_completion[name],
                }
                for name in self.tracked_classes
            },
        }

    def get_all_user_summaries(self) -> Dict[str, Dict[str, Any]]:
        return {user_id: self.get_user_summary(user_id) for user_id in self.user_states}
