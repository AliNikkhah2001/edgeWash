from __future__ import annotations

import numpy as np

from handwash_tracker import ModelConfig, MultiUserStepTracker, print_user_summaries


CLASS_NAMES = ["step_1", "step_2", "step_3", "no_action"]
IMAGE_SIZE = (64, 64)


class MockInferenceEngine:
    def __init__(self) -> None:
        self.config = ModelConfig(
            model_path="mock.keras",
            class_names=CLASS_NAMES,
            image_size=IMAGE_SIZE,
            no_action_class="no_action",
            confidence_threshold=0.55,
            batch_size=8,
        )
        self.class_names = list(CLASS_NAMES)
        self.no_action_class_id = self.class_names.index("no_action")

    def predict_probabilities(self, frames_bgr):
        probs = []
        for frame in frames_bgr:
            b, g, r = frame.mean(axis=(0, 1))
            channel_scores = np.array([b, g, r], dtype=np.float32)
            if channel_scores.max() < 20:
                probs.append(np.array([0.1, 0.1, 0.1, 0.7], dtype=np.float32))
                continue
            normalized = channel_scores / max(channel_scores.sum(), 1.0)
            probs.append(np.array([normalized[0], normalized[1], normalized[2], 0.05], dtype=np.float32))
        probs = np.stack(probs, axis=0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def _build_prediction_dict(self, probs):
        pred_class_id = int(np.argmax(probs))
        confidence = float(probs[pred_class_id])
        if confidence < self.config.confidence_threshold:
            pred_class_id = self.no_action_class_id
            confidence = float(probs[pred_class_id])
        return {
            "predicted_class_id": pred_class_id,
            "predicted_class": self.class_names[pred_class_id],
            "confidence": confidence,
            "probabilities": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
        }

    def predict_one_frame(self, frame_bgr):
        return self._build_prediction_dict(self.predict_probabilities([frame_bgr])[0])


def solid_color_frame(bgr: tuple[int, int, int], size: tuple[int, int] = (80, 80)) -> np.ndarray:
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:] = np.array(bgr, dtype=np.uint8)
    return frame


def run_demo() -> None:
    engine = MockInferenceEngine()
    tracker = MultiUserStepTracker(
        inference_engine=engine,
        fps=2.0,
        min_frames_for_completion={"step_1": 2, "step_2": 2, "step_3": 2},
        min_seconds_for_completion={"step_1": 1.0, "step_2": 1.0, "step_3": 1.0},
    )

    one_frame_result = engine.predict_one_frame(solid_color_frame((255, 0, 0)))
    print("Single-frame inference result:")
    print(one_frame_result)

    user_ids = ["user_1", "user_1", "user_1", "user_2", "user_2", "user_3"]
    frames = [
        solid_color_frame((255, 0, 0)),
        solid_color_frame((255, 0, 0)),
        solid_color_frame((0, 255, 0)),
        solid_color_frame((0, 255, 0)),
        solid_color_frame((0, 0, 255)),
        solid_color_frame((0, 0, 0)),
    ]
    timestamps_seconds = [0.0, 0.5, 1.0, 0.0, 0.5, 0.0]
    frame_indices = [0, 1, 2, 0, 1, 0]

    predictions = tracker.update(
        user_ids=user_ids,
        frames_bgr=frames,
        timestamps_seconds=timestamps_seconds,
        frame_indices=frame_indices,
    )

    print("\nFrame-by-frame outputs:")
    for prediction in predictions:
        print(
            {
                "user_id": prediction.user_id,
                "frame_index": prediction.frame_index,
                "timestamp_seconds": prediction.timestamp_seconds,
                "predicted_class": prediction.predicted_class,
                "confidence": round(prediction.confidence, 4),
                "completion_by_frames": prediction.per_class_completion_by_frames,
                "completion_by_seconds": prediction.per_class_completion_by_seconds,
            }
        )

    summaries = tracker.get_all_user_summaries()
    print("\nAggregated user summaries:")
    print_user_summaries(summaries)


if __name__ == "__main__":
    run_demo()
