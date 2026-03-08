#!/usr/bin/env bash
#SBATCH -J merge
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err

set -euo pipefail

mkdir -p logs

source /projects/m000226/miniconda3/etc/profile.d/conda.sh
conda activate plm

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

SCRIPT_DIR=/projects/m000226/seqfm-geometry/src/graph
DATA_DIR=/projects/m000226/seqfm-geometry/graph/
OUT_DIR=/projects/m000226/seqfm-geometry/graph
mkdir -p "$OUT_DIR"

python $SCRIPT_DIR/merge.py \
  --glob "$DATA_DIR/fmppi_unsup_perturb_M40_m20.part*.tsv" \
  --out  "$OUT_DIR/fmppi_unsup_perturb_M40_m20.tsv"

python $SCRIPT_DIR/merge.py \
  --glob "$DATA_DIR/fmppi_unsup_pll_M10_m10.part*.tsv" \
  --out  "$OUT_DIR/fmppi_unsup_pll_M10_m10.tsv"

python $SCRIPT_DIR/merge.py \
  --glob "$DATA_DIR/fmppi_unsup_pll_M20_m10.part*.tsv" \
  --out  "$OUT_DIR/fmppi_unsup_pll_M20_m10.tsv"