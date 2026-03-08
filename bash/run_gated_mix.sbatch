#!/usr/bin/env bash
#SBATCH -J gated_mix
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --array=0-3
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err

set -euo pipefail

mkdir -p logs

source /projects/m000226/miniconda3/etc/profile.d/conda.sh
conda activate plm

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

SCRIPT_DIR=/projects/m000226/seqfm-geometry/src/graph
EDGE_DIR=/projects/m000226/seqfm-geometry/parent_graph
DATA_DIR=/projects/m000226/seqfm-geometry/graph
OUT_DIR=/projects/m000226/seqfm-geometry/graph/gated_mix
mkdir -p "$OUT_DIR"

FILES=(
  "fmppi_unsup_perturb_M40_m20.tsv"
  "fmppi_unsup_pll_M20_m10.tsv"
  "fm_sim_knn_M500_m20_cos_pos.tsv"
  "fm_sim_knn_M500_m20_row_softmax.tsv"
)

FM_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
FM_PATH="$DATA_DIR/$FM_FILE"

BASENAME="$(basename "$FM_FILE" .tsv)"
RUN_OUTDIR="$OUT_DIR/$BASENAME"
mkdir -p "$RUN_OUTDIR"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Running with fm edge file: $FM_PATH"
echo "Output dir: $RUN_OUTDIR"

python "$SCRIPT_DIR/gated_mix.py" \
  --string_edges "$EDGE_DIR/string_edges.tsv" \
  --fmppi_edges "$FM_PATH" \
  --lambda_ 0.7 \
  --beta 1.0 \
  --g exp \
  --outdir "$RUN_OUTDIR"