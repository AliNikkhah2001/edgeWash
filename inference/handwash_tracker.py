from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from collections import defaultdict

import cv2
import numpy as np


ArrayLike = np.ndarray
ThresholdValue = Union[int, float, Mapping[str, Union[int, float]]]


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    class_names: List[str]
    image_size: Tuple[int, int]
    no_action_class: str = "no_action"
    confidence_threshold: float = 0.0
    batch_size: int = 32
    sequence_length_fallback: int = 16


@dataclass
class FramePrediction:
    user_id: str
    timestamp_seconds: float
    frame_index: int
    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: Dict[str, float]
    per_class_completion_by_frames: Dict[str, bool]
    per_class_completion_by_seconds: Dict[str, bool]


@dataclass
class UserClassStats:
    frames: int = 0
    seconds: float = 0.0
    completed_by_frames: bool = False
    completed_by_seconds: bool = False


@dataclass
class UserHistory:
    user_id: str
    total_frames_seen: int = 0
    last_timestamp_seconds: Optional[float] = None
    last_frame_index: Optional[int] = None
    class_stats: Dict[str, UserClassStats] = field(default_factory=dict)
    predictions: List[FramePrediction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total_frames_seen": self.total_frames_seen,
            "last_timestamp_seconds": self.last_timestamp_seconds,
            "last_frame_index": self.last_frame_index,
            "class_stats": {name: asdict(stats) for name, stats in self.class_stats.items()},
            "predictions": [asdict(pred) for pred in self.predictions],
        }


class ModelInferenceEngine:
    """Loads a Keras model and returns per-frame class probabilities.

    Supports both frame models of shape [B, H, W, C] and sequence models of
    shape [B, T, H, W, C]. For sequence models, a single input frame is repeated
    across the temporal axis so that one-frame calls still work.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.class_names = list(config.class_names)
        if config.no_action_class not in self.class_names:
            raise ValueError(
                f"no_action_class='{config.no_action_class}' must exist in class_names."
            )
        self.no_action_class_id = self.class_names.index(config.no_action_class)
        self.model = self._load_model(config.model_path)
        self.input_shape = self._get_model_input_shape(self.model)
        self.is_sequence_model = len(self.input_shape) == 5
        self.sequence_length = self._infer_sequence_length()

    def _load_model(self, model_path: str):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required to load the model. Install it with `pip install tensorflow`."
            ) from exc

        custom_objects: Dict[str, Any] = {}
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess

            custom_objects["preprocess_input"] = mobilenet_v2_preprocess
        except Exception:
            pass

        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )

    def _get_model_input_shape(self, model) -> Tuple[Optional[int], ...]:
        shape = model.input_shape
        if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return tuple(shape)

    def _infer_sequence_length(self) -> int:
        if self.is_sequence_model and len(self.input_shape) > 1 and self.input_shape[1] is not None:
            return int(self.input_shape[1])
        return int(self.config.sequence_length_fallback)

    def preprocess_frame(self, frame_bgr: ArrayLike) -> ArrayLike:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Each frame must have shape [H, W, 3] in BGR format.")
        resized = cv2.resize(frame_bgr, self.config.image_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def _prepare_batch(self, frames_bgr: Sequence[ArrayLike]) -> ArrayLike:
        processed = [self.preprocess_frame(frame) for frame in frames_bgr]
        batch = np.stack(processed, axis=0)
        if self.is_sequence_model:
            batch = np.repeat(batch[:, None, ...], self.sequence_length, axis=1)
        return batch

    def predict_probabilities(self, frames_bgr: Sequence[ArrayLike]) -> ArrayLike:
        if not frames_bgr:
            return np.zeros((0, len(self.class_names)), dtype=np.float32)
        batch = self._prepare_batch(frames_bgr)
        probs = self.model.predict(batch, verbose=0, batch_size=self.config.batch_size)
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[1] != len(self.class_names):
            raise ValueError(
                f"Expected model output shape [N, {len(self.class_names)}], got {probs.shape}."
            )
        return probs

    def predict_one_frame(self, frame_bgr: ArrayLike) -> Dict[str, Any]:
        probs = self.predict_probabilities([frame_bgr])[0]
        return self._build_prediction_dict(probs)

    def _build_prediction_dict(self, probs: ArrayLike) -> Dict[str, Any]:
        pred_class_id = int(np.argmax(probs))
        confidence = float(probs[pred_class_id])
        if confidence < self.config.confidence_threshold:
            pred_class_id = self.no_action_class_id
            confidence = float(probs[pred_class_id])
        probabilities = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        return {
            "predicted_class_id": pred_class_id,
            "predicted_class": self.class_names[pred_class_id],
            "confidence": confidence,
            "probabilities": probabilities,
        }


class CompletionThresholds:
    def __init__(
        self,
        class_names: Sequence[str],
        no_action_class: str,
        min_frames: ThresholdValue,
        min_seconds: ThresholdValue,
    ) -> None:
        self.class_names = list(class_names)
        self.no_action_class = no_action_class
        self.min_frames = self._normalize_thresholds(min_frames, cast_type=int)
        self.min_seconds = self._normalize_thresholds(min_seconds, cast_type=float)

    def _normalize_thresholds(
        self,
        value: ThresholdValue,
        cast_type: Any,
    ) -> Dict[str, Union[int, float]]:
        tracked_classes = [name for name in self.class_names if name != self.no_action_class]
        if isinstance(value, Mapping):
            return {name: cast_type(value.get(name, 0)) for name in tracked_classes}
        return {name: cast_type(value) for name in tracked_classes}


class MultiUserStepTracker:
    """Maintains per-user step history and completion state over time."""

    def __init__(
        self,
        inference_engine: ModelInferenceEngine,
        fps: float,
        min_frames_for_completion: ThresholdValue = 1,
        min_seconds_for_completion: ThresholdValue = 0.0,
        store_prediction_history: bool = True,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.engine = inference_engine
        self.fps = float(fps)
        self.seconds_per_frame = 1.0 / self.fps
        self.store_prediction_history = store_prediction_history
        self.thresholds = CompletionThresholds(
            class_names=self.engine.class_names,
            no_action_class=self.engine.config.no_action_class,
            min_frames=min_frames_for_completion,
            min_seconds=min_seconds_for_completion,
        )
        self.histories: MutableMapping[str, UserHistory] = {}

    def _empty_class_stats(self) -> Dict[str, UserClassStats]:
        return {name: UserClassStats() for name in self.engine.class_names}

    def _get_or_create_user(self, user_id: str) -> UserHistory:
        if user_id not in self.histories:
            self.histories[user_id] = UserHistory(user_id=user_id, class_stats=self._empty_class_stats())
        return self.histories[user_id]

    def update(
        self,
        user_ids: Sequence[Union[str, int]],
        frames_bgr: Sequence[ArrayLike],
        timestamps_seconds: Optional[Sequence[float]] = None,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> List[FramePrediction]:
        if not (len(user_ids) == len(frames_bgr)):
            raise ValueError("user_ids and frames_bgr must have the same length")
        if timestamps_seconds is not None and len(timestamps_seconds) != len(frames_bgr):
            raise ValueError("timestamps_seconds must have the same length as frames_bgr")
        if frame_indices is not None and len(frame_indices) != len(frames_bgr):
            raise ValueError("frame_indices must have the same length as frames_bgr")

        probabilities = self.engine.predict_probabilities(frames_bgr)
        outputs: List[FramePrediction] = []

        for i, raw_user_id in enumerate(user_ids):
            user_id = str(raw_user_id)
            user_history = self._get_or_create_user(user_id)
            frame_index = int(frame_indices[i]) if frame_indices is not None else user_history.total_frames_seen
            timestamp_seconds = (
                float(timestamps_seconds[i])
                if timestamps_seconds is not None
                else frame_index * self.seconds_per_frame
            )

            inference = self.engine._build_prediction_dict(probabilities[i])
            predicted_class = str(inference["predicted_class"])
            predicted_class_id = int(inference["predicted_class_id"])

            user_history.total_frames_seen += 1
            user_history.last_frame_index = frame_index
            user_history.last_timestamp_seconds = timestamp_seconds

            if predicted_class != self.engine.config.no_action_class:
                class_stats = user_history.class_stats[predicted_class]
                class_stats.frames += 1
                class_stats.seconds += self.seconds_per_frame
                class_stats.completed_by_frames = (
                    class_stats.frames >= self.thresholds.min_frames[predicted_class]
                )
                class_stats.completed_by_seconds = (
                    class_stats.seconds >= self.thresholds.min_seconds[predicted_class]
                )

            frame_prediction = FramePrediction(
                user_id=user_id,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
                predicted_class=predicted_class,
                predicted_class_id=predicted_class_id,
                confidence=float(inference["confidence"]),
                probabilities=dict(inference["probabilities"]),
                per_class_completion_by_frames=self.get_completion_flags(user_id, by="frames"),
                per_class_completion_by_seconds=self.get_completion_flags(user_id, by="seconds"),
            )
            if self.store_prediction_history:
                user_history.predictions.append(frame_prediction)
            outputs.append(frame_prediction)

        return outputs

    def get_completion_flags(self, user_id: Union[str, int], by: str = "frames") -> Dict[str, bool]:
        history = self._get_or_create_user(str(user_id))
        tracked_classes = [name for name in self.engine.class_names if name != self.engine.config.no_action_class]
        if by == "frames":
            return {name: history.class_stats[name].completed_by_frames for name in tracked_classes}
        if by == "seconds":
            return {name: history.class_stats[name].completed_by_seconds for name in tracked_classes}
        raise ValueError("by must be 'frames' or 'seconds'")

    def get_user_summary(self, user_id: Union[str, int]) -> Dict[str, Any]:
        history = self._get_or_create_user(str(user_id))
        tracked_classes = [name for name in self.engine.class_names if name != self.engine.config.no_action_class]
        return {
            "user_id": history.user_id,
            "total_frames_seen": history.total_frames_seen,
            "last_frame_index": history.last_frame_index,
            "last_timestamp_seconds": history.last_timestamp_seconds,
            "per_class": {
                name: {
                    "frames_done": history.class_stats[name].frames,
                    "seconds_done": history.class_stats[name].seconds,
                    "completed_by_frames": history.class_stats[name].completed_by_frames,
                    "completed_by_seconds": history.class_stats[name].completed_by_seconds,
                    "required_frames": self.thresholds.min_frames[name],
                    "required_seconds": self.thresholds.min_seconds[name],
                }
                for name in tracked_classes
            },
        }

    def get_all_user_summaries(self) -> Dict[str, Dict[str, Any]]:
        return {user_id: self.get_user_summary(user_id) for user_id in self.histories.keys()}


def print_user_summaries(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    for user_id, summary in summaries.items():
        print(f"\nUser: {user_id}")
        print(f"  total_frames_seen={summary['total_frames_seen']}")
        print(f"  last_frame_index={summary['last_frame_index']}")
        print(f"  last_timestamp_seconds={summary['last_timestamp_seconds']:.3f}" if summary['last_timestamp_seconds'] is not None else "  last_timestamp_seconds=None")
        for class_name, class_summary in summary["per_class"].items():
            print(
                "  "
                f"{class_name}: frames_done={class_summary['frames_done']}, "
                f"seconds_done={class_summary['seconds_done']:.3f}, "
                f"completed_by_frames={class_summary['completed_by_frames']}, "
                f"completed_by_seconds={class_summary['completed_by_seconds']}"
            )
