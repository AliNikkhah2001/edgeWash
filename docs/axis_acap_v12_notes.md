# Axis ACAP v12 notes for EdgeWash deployment

This note captures constraints and recommendations for running handwash models on Axis cameras with ACAP v12.

## Deep-learning framework support and limitations
- On ACAP v12 devices, the DLPU accelerates inference only for TensorFlow Lite INT8 models. CV25 uses a proprietary format converted from TFLite or ONNX.
- No custom ops, PyTorch, or raw ONNX on the DLPU. Unsupported ops fall back to CPU and run much slower.
- Heavy architectures (for example ViTs or deformable conv) are discouraged. Use light backbones such as MobileNet, ResNet-18, or EfficientNet-Lite.

References:
- https://developer.axis.com/computer-vision/computer-vision-on-device/supported-frameworks/
- https://developer.axis.com/computer-vision/computer-vision-on-device/general-suggestions/

## SoC-specific tips
- ARTPEC-8: best with per-tensor quantization, 3x3 conv (stride 2), filter counts divisible by 6, ReLU activation.
- ARTPEC-9: tensors <= 4D, batch = 1, prefer ReLU6, filters divisible by 16.
- CV25: uses a proprietary format converted from TFLite or ONNX.

References:
- https://developer.axis.com/computer-vision/computer-vision-on-device/optimization-tips/

## Implication for EdgeWash
- Convert TF/Keras models to quantized TFLite INT8 to leverage the DLPU.
- Prefer built-in TFLite ops only; avoid SELECT_TF_OPS.

## Recommended architectures
Axis recommends SoC-specific yet overlapping families:
- Backbone (default): MobileNet v2 across all SoCs.
- Detection heads:
  - SSD MobileNet v2: light, real-time baseline.
  - SSDLite MobileDet: higher accuracy, extra compute.
  - YOLOv5 (n/s/m): supported on ARTPEC-8/9; higher accuracy at higher latency.
- ARTPEC-8: ResNet-18 is noted as a stronger backbone (DIY training).
- CV25: MobileNet v2 with SSD or SSDLite.

Reference:
- https://developer.axis.com/computer-vision/computer-vision-on-device/recommended-model-architecture/

## Model Zoo benchmarks (human detection)
Axis Model Zoo provides benchmark tables for models and SoCs. Key observations:
- SSD MobileNet v2: best real-time performance (about 10 to 30 ms) with moderate accuracy (about 25 to 26 mAP).
- SSDLite MobileDet: about 33 mAP but about 1.5x to 2x slower than SSD MobileNet on the same SoC.
- YOLOv5 (n/s/m) on ARTPEC-8/9: accuracy up to about 38 mAP with 40 to 95 ms latency.

Reference:
- https://github.com/AxisCommunications/axis-model-zoo

## Pose estimation for tracking
- Axis provides an ARTPEC-8 pose estimator example using MoveNet SinglePose Lightning (192x192 input, 17 keypoints).
- Pose-based tracking can be robust to occlusions when boxes overlap.
- Example is implemented as a multi-container ACAP app (capture + TFLite inference, plus a Flask API).

Reference:
- https://raw.githubusercontent.com/AxisCommunications/acap-computer-vision-sdk-examples/main/pose-estimator-with-flask/README.md

## Tracking algorithms on ACAP
The DLPU performs inference only; tracking runs on CPU within the ACAP app. Options include:
- SORT (Kalman filter + Hungarian assignment)
- DeepSORT (adds appearance embeddings)
- ByteTrack (uses high/low-confidence detections)

Reference:
- https://developer.axis.com/analytics/axis-scene-metadata/reference/concepts/object-tracking/

## Axis-provided analytics
- AXIS Object Analytics (AOA): detects, classifies, tracks, emits events. No custom training.
- Digital Autotracking (PTZ): motion-based tracking and zoom, not customizable.

## Answers to the research questions
- Can we deploy any model on this device? No. DLPU supports INT8 TFLite only, with strict operator limits. Unsupported ops fall back to CPU.
- Is the choice of vision backbone limited? Yes. Prefer MobileNet v2; ResNet-18 is feasible on ARTPEC-8. Transformers and large nets are discouraged.
- What algorithms/models for tracking human movement?
  - Detector or pose per frame: SSD MobileNet v2, SSDLite MobileDet, YOLOv5 (n/s/m), or MoveNet SinglePose Lightning.
  - Tracker on CPU: SORT, DeepSORT, or ByteTrack.
  - Alternative: use AXIS Object Analytics or Digital Autotracking.
