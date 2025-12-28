# EDI-Riga Handwashing Movement Classifiers

Hospital-focused WHO step classifiers with training scripts and pretrained checkpoints maintained by EDI-Riga.

- Media: RGB video frames with optional optical flow; supports PSKUS/METC/Kaggle splits.
- Architecture: single-frame CNN baselines, time-distributed CNN + GRU video models, and two-stream RGB+flow networks.
- Contribution: end-to-end preprocessing and training recipes for WHO step recognition on hospital data.
- Availability: code open; datasets referenced (public/non-public) not bundled; weights not included; no external API.
