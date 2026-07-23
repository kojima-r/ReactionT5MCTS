#!/usr/bin/env bash
# Create a dedicated conda env for AiZynthFinder + PaRoutes analysis.
# AiZynthFinder pulls in route-distances, which also provides the tree-edit
# distance used by PaRoutes' official route_quality.py analysis (that package
# requires Python < 3.9 to install via pip, but the aizynthfinder dependency
# resolves it for a compatible interpreter here).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh

ENV=aizynth
if ! conda env list | grep -q "/envs/${ENV}\b"; then
    conda create -y -n "${ENV}" python=3.11
fi
conda activate "${ENV}"
pip install --upgrade pip
# aizynthfinder (>=4) uses ONNX models and installs route-distances + rdkit + tqdm
pip install "aizynthfinder[all]"
python -c "import aizynthfinder, route_distances; print('aizynthfinder', aizynthfinder.__version__)"
echo "ENV READY"
