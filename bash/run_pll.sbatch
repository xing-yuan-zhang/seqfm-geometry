#!/usr/bin/env bash
#SBATCH -J pll
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

SCRIPT_DIR=/projects/m000226/seqfm-geometry/src/graph
DATA_DIR=/projects/m000226/seqfm-geometry/parent_graph/
OUT_DIR=/projects/m000226/seqfm-geometry/graph
mkdir -p "$OUT_DIR"

python $SCRIPT_DIR/attention_proxy.py \
  --emb_npz $DATA_DIR/emb/esm2_t33_650m.npz \
  --fasta $DATA_DIR/seqs.fasta \
  --nodes $DATA_DIR/node_ids.txt \
  --model_name facebook/esm2_t33_650M_UR50D \
  --method pll \
  --pll_sample_k 64 \
  --cand_M 200 \
  --topm 20 \
  --outdir $OUT_DIR