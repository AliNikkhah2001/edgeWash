#!/usr/bin/env bash
set -euo pipefail
if ! command -v gdown >/dev/null 2>&1; then
  echo "Missing gdown. Install with: pip install gdown"
  exit 1
fi
mkdir -p raw
cd raw
echo "Downloading synthetic hand-washing dataset archives from Google Drive..."
for id in \\
  "1EW3JQvElcuXzawxEMRkA8YXwK_Ipiv-p" \\
  "163TsrDe4q5KTQGCv90JRYFkCs7AGxFip" \\
  "1GxyTYfSodumH78NbjWdmbjm8JP8AOkAY" \\
  "1IoRsgBBr8qoC3HO-vEr6E7K4UZ6ku6-1" \\
  "1svCYnwDazy5FN1DYSgqbGscvDKL_YnID"
do
  gdown "https://drive.google.com/uc?id=${id}"
done
echo "Download complete. Extract the archives in this folder."
