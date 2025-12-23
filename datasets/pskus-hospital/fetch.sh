#!/usr/bin/env bash
set -euo pipefail
mkdir -p raw
cd raw
echo "Downloading PSKUS hospital dataset (large download)..."
curl -L "https://zenodo.org/record/4537209/files/Handwashing_dataset.zip?download=1" -o Handwashing_dataset.zip
echo "Download complete. Unzip with: unzip Handwashing_dataset.zip"
