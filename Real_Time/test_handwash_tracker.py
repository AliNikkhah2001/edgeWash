from __future__ import annotations

import numpy as np

from handwash_tracker import ModelConfig, ModelInferenceEngine, MultiUserFrameInference


NUM_CLASSES = 8
CLASS_NAMES = [
    'No_Action',
    'Soaping',
    'Step_1_PalmToPalm',
    'Step_2_PalmOverDorsum',
    'Step_3_InterlacedFingers',
    'Step_4_BackOfFingers',
    'Step_5_ThumbRub',
    'Step_6_Fingertips',
]


def make_dummy_frame(bgr_color, h=224, w=224):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = np.array(bgr_color, dtype=np.uint8)
    return frame


def run_real_model_test():
    config = ModelConfig(
        model_path='mobilenetv2_final.keras',
        class_names=CLASS_NAMES,
        image_size=(224, 224),
        no_action_class='No_Action',
        confidence_threshold=0.7,
        batch_size=32,
    )
    model = ModelInferenceEngine(config)

    inference = MultiUserFrameInference(
        inference_engine=model,
        fps=30.0,
        min_frames_for_completion=10,
        min_seconds_for_completion=1.0,
    )

    user_ids = ['user_1', 'user_1', 'user_2', 'user_2']
    frames = [
        make_dummy_frame((0, 0, 255)),
        make_dummy_frame((0, 255, 0)),
        make_dummy_frame((255, 0, 0)),
        make_dummy_frame((64, 64, 64)),
    ]

    results = inference.infer(user_ids=user_ids, frames_bgr=frames)

    print('FRAME RESULTS')
    for i, result in enumerate(results):
        print(f'\nitem {i}')
        print('user_id:', result.user_id)
        print('predicted_class:', result.predicted_class)
        print('predicted_class_id:', result.predicted_class_id)
        print('confidence:', round(result.confidence, 4))
        print('probabilities:', {k: round(v, 4) for k, v in result.probabilities.items()})
        print('class_frames_done:', result.class_frames_done)
        print('class_seconds_done:', {k: round(v, 3) for k, v in result.class_seconds_done.items()})
        print('class_completed_by_frames:', result.class_completed_by_frames)
        print('class_completed_by_seconds:', result.class_completed_by_seconds)

    print('\nUSER SUMMARIES')
    summaries = inference.get_all_user_summaries()
    for user_id, summary in summaries.items():
        print(f'\n{user_id}')
        for class_name, info in summary['per_class'].items():
            print(class_name, info)


if __name__ == '__main__':
    run_real_model_test()
