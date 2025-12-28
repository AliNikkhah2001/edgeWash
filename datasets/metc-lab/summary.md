# METC Lab Handwashing Dataset

Lab-collected WHO handwash recordings from the Medical Education Technology Center (Riga Stradins University). Multiple camera interfaces, annotated with WHO step labels; aligns with PSKUS label scheme (6 steps + "Other").

- Data type: RGB videos from multiple camera interfaces.
- Labels: per-frame WHO step codes (6 steps + Other).
- Availability: data public on Zenodo; preprocessing scripts in `code/edgewash`; weights not included.

## Download
- Zenodo: `https://zenodo.org/record/5808789/files/METC_handwashing_dataset.zip?download=1`
- Script: `./fetch.sh` downloads to `datasets/metc-lab/raw/`.

## Notes
- Use with `code/edgewash/dataset-metc/` preprocessing scripts to extract frames and split train/test.
