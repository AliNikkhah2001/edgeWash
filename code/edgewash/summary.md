# EdgeWash WHO Movement Classifier

Lightweight edge-focused pipeline for WHO hand-washing movement recognition with exportable on-device models.

- Media: RGB video frames with optional optical flow; supports PSKUS/METC/Kaggle splits.
- Architecture: single-frame MobileNetV2 CNN, time-distributed CNN + GRU for short clips, and two-stream RGB+flow variants.
- Contribution: preprocessing + training scripts for WHO step classification tuned for edge deployment.
- Availability: code open; datasets linked (mix of public and non-public) not bundled; trained weights not included; no external API.
