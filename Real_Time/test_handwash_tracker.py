from __future__ import annotations

import numpy as np

from handwash_tracker import ModelConfig, ModelInferenceEngine, MultiUserFrameInference


CLASS_NAMES = [
    "Other",
    "Step1_PalmToPalm",
    "Step2_PalmOverDorsum",
    "Step3_InterlacedFingers",
    "Step4_BackOfFingers",
    "Step5_ThumbRub",
    "Step6_Fingertips",
]


class MockModelInferenceEngine:
    """Small fake model so the tracking code can be tested without TensorFlow."""

    def __init__(self, class_names, no_action_class="Other", confidence_threshold=0.0):
        self.config = ModelConfig(
            model_path="mock.keras",
            class_names=list(class_names),
            image_size=(224, 224),
            no_action_class=no_action_class,
            confidence_threshold=confidence_threshold,
            batch_size=32,
        )
        self.class_names = list(class_names)
        self.no_action_class_id = self.class_names.index(no_action_class)

    def predict_probabilities(self, frames_bgr):
        probs = []
        for frame in frames_bgr:
            # Very simple fake logic from average BGR values.
            mean_b, mean_g, mean_r = frame.mean(axis=(0, 1))
            p = np.zeros(len(self.class_names), dtype=np.float32)

            if max(mean_b, mean_g, mean_r) < 10:
                p[self.no_action_class_id] = 1.0
            elif mean_r >= mean_g and mean_r >= mean_b:
                p[self.class_names.index("Step1_PalmToPalm")] = 0.85
                p[self.no_action_class_id] = 0.15
            elif mean_g >= mean_r and mean_g >= mean_b:
                p[self.class_names.index("Step2_PalmOverDorsum")] = 0.85
                p[self.no_action_class_id] = 0.15
            else:
                p[self.class_names.index("Step3_InterlacedFingers")] = 0.85
                p[self.no_action_class_id] = 0.15

            probs.append(p)
        return np.stack(probs, axis=0)

    def build_prediction(self, probs: np.ndarray):
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

    def predict_one(self, frame_bgr):
        return self.build_prediction(self.predict_probabilities([frame_bgr])[0])


def make_dummy_frame(bgr_color, h=224, w=224):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = np.array(bgr_color, dtype=np.uint8)
    return frame


def run_mock_test():
    engine = MockModelInferenceEngine(CLASS_NAMES, no_action_class="Other", confidence_threshold=0.5)
    tracker = MultiUserFrameInference(
        inference_engine=engine,
        fps=30.0,
        min_frames_for_completion=2,
        min_seconds_for_completion=0.05,
    )

    user_ids = ["user_1", "user_1", "user_2", "user_2", "user_1", "user_3"]
    frames = [
        make_dummy_frame((0, 0, 255)),   # red -> Step1
        make_dummy_frame((0, 255, 0)),   # green -> Step2
        make_dummy_frame((255, 0, 0)),   # blue -> Step3
        make_dummy_frame((255, 0, 0)),   # blue -> Step3
        make_dummy_frame((0, 0, 255)),   # red -> Step1
        make_dummy_frame((0, 0, 0)),     # black -> Other
    ]

    results = tracker.infer(user_ids=user_ids, frames_bgr=frames)

    print("FRAME RESULTS")
    for i, result in enumerate(results):
        print(f"\nitem {i}")
        print("user_id:", result.user_id)
        print("predicted_class:", result.predicted_class)
        print("confidence:", round(result.confidence, 4))
        print("class_frames_done:", result.class_frames_done)
        print("class_seconds_done:", {k: round(v, 3) for k, v in result.class_seconds_done.items()})
        print("class_completed_by_frames:", result.class_completed_by_frames)
        print("class_completed_by_seconds:", result.class_completed_by_seconds)

    print("\nUSER SUMMARIES")
    for user_id, summary in tracker.get_all_user_summaries().items():
        print(f"\n{user_id}")
        for class_name, class_info in summary["per_class"].items():
            print(class_name, class_info)


def run_real_model_example():
    # Replace the model path with your own file if your environment can load it.
    config = ModelConfig(
        model_path="mobilenetv2_final.keras",
        class_names=CLASS_NAMES,
        image_size=(224, 224),
        no_action_class="Other",
        confidence_threshold=0.7,
    )
    engine = ModelInferenceEngine(config)
    tracker = MultiUserFrameInference(
        inference_engine=engine,
        fps=30.0,
        min_frames_for_completion=10,
        min_seconds_for_completion=1.0,
    )

    user_ids = ["user_1", "user_2"]
    frames = [
        make_dummy_frame((0, 0, 255)),
        make_dummy_frame((0, 255, 0)),
    ]
    results = tracker.infer(user_ids=user_ids, frames_bgr=frames)
    print(results)


if __name__ == "__main__":
    run_mock_test()
    # Uncomment only when your TensorFlow / Keras environment can load the real model.
    # run_real_model_example()
