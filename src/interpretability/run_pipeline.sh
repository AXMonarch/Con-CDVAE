#!/bin/bash
export PYTHONPATH=/afs/inf.ed.ac.uk/user/s23/s2305255/Desktop/MLP/Con-CDVAE:/afs/inf.ed.ac.uk/user/s23/s2305255/Desktop/MLP/Con-CDVAE/src:$PYTHONPATH


# run_pipeline.sh
# ---------------
# Full SAE explainability pipeline for Con-CDVAE.
# Edit the variables in the CONFIG section then run:
#   bash run_pipeline.sh
#
# Prerequisites:
#   - Con-CDVAE conda environment activated (concdvae310)
#   - Con-CDVAE repo on PYTHONPATH
#   - A trained Con-CDVAE checkpoint
#   - MP20 dataset downloaded to $DATA_PATH

set -euo pipefail

# ===========================================================================
# CONFIG — edit these
# ===========================================================================
CHECKPOINT="/afs/inf.ed.ac.uk/user/s23/s2305255/Desktop/MLP/Con-CDVAE/src/model/mp20_format/epoch=330-step=17543.ckpt"
DATA_PATH="/afs/inf.ed.ac.uk/user/s23/s2305255/Desktop/MLP/Con-CDVAE/data/mp_20"
LAYER_IDX=-1          # -1 = last DimeNet++ interaction block (recommended)
BATCH_SIZE=32         # crystals per batch for extraction
DEVICE="cuda"         # or "cpu"

# SAE hyperparameters
INPUT_DIM=256         # GNN hidden dim — check your model config
EXPANSION=16          # SAE dictionary = INPUT_DIM * EXPANSION  (4096 features)
K=32                  # Top-K: features firing per atom-token
EPOCHS=20
SAE_BATCH=4096
LR=2e-4

# Output directories
ACTS_DIR="./sae_activations"
SAE_DIR="./sae_checkpoints"
ANALYSIS_DIR="./sae_analysis"
# ===========================================================================

echo "========================================================"
echo " Con-CDVAE SAE Explainability Pipeline"
echo "========================================================"
echo "Checkpoint : $CHECKPOINT"
echo "Data       : $DATA_PATH"
echo "Device     : $DEVICE"
echo ""

# Step 1 — Extract activations
echo ">>> STEP 1: Extracting GNN activations from frozen Con-CDVAE ..."
python extract_activations.py \
    --checkpoint  "$CHECKPOINT" \
    --data_path   "$DATA_PATH"  \
    --output_dir  "$ACTS_DIR"   \
    --layer_idx   $LAYER_IDX    \
    --batch_size  $BATCH_SIZE   \
    --splits      train val     \
    --device      $DEVICE

echo ""
echo ">>> STEP 2: Training Top-K SAE ..."
python train_sae.py \
    --activations_dir "$ACTS_DIR" \
    --output_dir      "$SAE_DIR"  \
    --input_dim       $INPUT_DIM  \
    --expansion       $EXPANSION  \
    --k               $K          \
    --epochs          $EPOCHS     \
    --batch_size      $SAE_BATCH  \
    --lr              $LR         \
    --device          $DEVICE

echo ""
echo ">>> STEP 3: Interpreting SAE features ..."
python interpretable_features.py \
    --sae_checkpoint  "$SAE_DIR/sae_best.pt" \
    --activations_dir "$ACTS_DIR"            \
    --output_dir      "$ANALYSIS_DIR"        \
    --top_n_features  100                    \
    --device          $DEVICE

echo ""
echo "========================================================"
echo " Pipeline complete!"
echo " Results in: $ANALYSIS_DIR"
echo "========================================================"
