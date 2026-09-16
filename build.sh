#!/usr/bin/env bash
# Node Sentinel Render / Production Build Script
set -e

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Checking Tesseract OCR binary availability ==="
if command -v tesseract >/dev/null 2>&1; then
    echo "Tesseract executable found at: $(command -v tesseract)"
    tesseract --version | head -n 1
else
    echo "Notice: Tesseract not found in default PATH. Direct digital PDF parsing will be prioritized."
    echo "For full scanned/image OCR on Render, deploy using the Docker service configuration in render.yaml."
fi

echo "=== Build completed successfully ==="
