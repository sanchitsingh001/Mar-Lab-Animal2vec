#!/usr/bin/env bash
set -e

echo "=== Loading conda ==="
source /opt/conda/etc/profile.d/conda.sh

echo "=== Creating conda environment: env_transformer ==="
conda create -y --name env_transformer python=3.9

echo "=== Activating env_transformer ==="
conda activate env_transformer

echo "=== Upgrading pip to 24.0 ==="
pip install --upgrade pip==24.0

echo "=== Installing Python requirements ==="
pip install -r requirements.txt

echo "=== Installing system compiler (gxx_linux-64) ==="
conda install -y -c conda-forge gxx_linux-64

echo "=== Reinstalling fairseq from pinned commit ==="
pip uninstall -y fairseq || true
pip install git+https://github.com/facebookresearch/fairseq.git@d871f616

echo "=== Pinning numpy to 1.23.5 ==="
pip uninstall -y numpy
pip install numpy==1.23.5

echo "=== Environment setup complete ==="
echo "Activate later with: conda activate env_transformer"

