#!/usr/bin/env bash
#SBATCH -J knn_sm
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail

mkdir -p logs

source /projects/m000226/miniconda3/etc/profile.d/conda.sh
conda activate plm

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

SCRIPT_DIR=/projects/m000226/seqfm-geometry/src/graph
DATA_DIR=/projects/m000226/seqfm-geometry/parent_graph/
OUT_DIR=/projects/m000226/seqfm-geometry/graph
mkdir -p "$OUT_DIR"

python $SCRIPT_DIR/knn.py \
  --emb_npz $DATA_DIR/emb/esm2_t33_650m.npz \
  --nodes $DATA_DIR/node_ids.txt \
  --cand_M 500 \
  --topm 20 \
  --weight row_softmax \
  --tau 0.05 \
  --outdir $OUT_DIR