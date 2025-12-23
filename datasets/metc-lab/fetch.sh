#!/usr/bin/env bash
set -euo pipefail
mkdir -p raw
cd raw
echo "Downloading METC lab dataset (large download)..."
curl -L "https://zenodo.org/record/5808789/files/METC_handwashing_dataset.zip?download=1" -o METC_handwashing_dataset.zip
echo "Download complete. Unzip with: unzip METC_handwashing_dataset.zip"
