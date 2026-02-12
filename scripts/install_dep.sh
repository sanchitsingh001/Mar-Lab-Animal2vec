#!/usr/bin/env bash
set -e

echo "=== Installing system dependencies (wget, bzip2, ca-certificates) ==="
apt-get update
apt-get install -y wget bzip2 ca-certificates

echo "=== Downloading Miniconda ==="
cd /tmp
MINICONDA=Miniconda3-latest-Linux-x86_64.sh
wget -q https://repo.anaconda.com/miniconda/${MINICONDA}

echo "=== Installing Miniconda to /opt/conda ==="
bash ${MINICONDA} -b -p /opt/conda
rm -f ${MINICONDA}

echo "=== Making conda available immediately ==="
export PATH=/opt/conda/bin:$PATH
hash -r

echo "=== Verifying conda ==="
conda --version

echo "=== Persisting PATH for future shells ==="
if ! grep -q "/opt/conda/bin" /root/.bashrc; then
  echo 'export PATH=/opt/conda/bin:$PATH' >> /root/.bashrc
fi

echo "=== Initializing conda ==="
/opt/conda/bin/conda init bash

echo "=== Disabling auto-activation of base env ==="
conda config --set auto_activate_base false

echo "=== Installation complete ==="
echo "Open a new shell or run: source /root/.bashrc"

