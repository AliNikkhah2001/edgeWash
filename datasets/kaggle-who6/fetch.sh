#!/usr/bin/env bash
set -euo pipefail
mkdir -p raw
cd raw
echo "Downloading Kaggle WHO6 handwash tarball..."
curl -L "https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar" -o kaggle-dataset-6classes.tar
echo "Extracting..."
tar xf kaggle-dataset-6classes.tar
echo "Done."
