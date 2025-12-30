#!/usr/bin/env bash
set -euo pipefail
if ! command -v gdown >/dev/null 2>&1; then
  echo "Missing gdown. Install with: pip install gdown"
  exit 1
fi
mkdir -p raw
cd raw
echo "Downloading UWash raw dataset from Google Drive..."
gdown "https://drive.google.com/uc?id=1ZRdRiwXp4xbFUWIIjIQ0OEK6gK0cwODN" -O Dataset_raw.zip
echo "Download complete. Unzip with: unzip Dataset_raw.zip"
