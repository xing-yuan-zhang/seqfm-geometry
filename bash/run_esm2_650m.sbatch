#!/usr/bin/env bash
#SBATCH -J esm2_650m
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

export HF_HOME=/projects/m000226/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR=/projects/m000226/seqfm-geometry/src/emb
DATA_DIR=/projects/m000226/seqfm-geometry/parent_graph/
OUT_DIR=/projects/m000226/seqfm-geometry/emb
mkdir -p "$OUT_DIR"

python $SCRIPT_DIR/emb.py \
  --backend esm2 \
  --model facebook/esm2_t33_650M_UR50D \
  --fasta $DATA_DIR/seqs.fasta \
  --node_ids $DATA_DIR/node_ids.txt \
  --out $OUT_DIR/esm2_t33_650m.npz \
  --max_len 1022 \
  --stride 768 \
  --bucket 128 \
  --win_bs 8 \
  --cpu_threads 4 \
  --l2norm