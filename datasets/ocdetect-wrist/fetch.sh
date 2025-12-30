#!/usr/bin/env bash
set -euo pipefail
mkdir -p raw
cd raw
echo "Downloading OCDetect dataset (large ~31.6 GB)..."
curl -L "https://zenodo.org/records/13924901/files/OCDetect_dataset.zip?download=1" -o OCDetect_dataset.zip
echo "Download complete. Unzip with: unzip OCDetect_dataset.zip"
