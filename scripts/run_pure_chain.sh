#!/bin/bash
# Force PyTorch to only see GPU 1 as its default 'cuda:0'
export CUDA_VISIBLE_DEVICES=1

OUT_DIR="runs/ulaanbaatar_pure_chain"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/pipeline.log"
INPUT="data/Ulaanbaatar.wav"

echo "Starting Pure Chain" > "$LOG"

# Step 1: A
echo "--- Step 1: A ---" >> "$LOG"
bash experiments/mvp_a_rave_latent/run.sh --config experiments/mvp_a_rave_latent/config_chain_boost.yaml --mode render --input "$INPUT" --output "$OUT_DIR/step_1_a.wav" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED at Step 1" >> "$LOG"; exit 1; fi

# Step 2: D
echo "--- Step 2: D ---" >> "$LOG"
bash experiments/mvp_d_ckpt_morph/run.sh --config experiments/mvp_d_ckpt_morph/config_chain_boost.yaml --mode render --input "$OUT_DIR/step_1_a.wav" --output "$OUT_DIR/step_2_d.wav" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED at Step 2" >> "$LOG"; exit 1; fi

# Step 3: D
echo "--- Step 3: D ---" >> "$LOG"
bash experiments/mvp_d_ckpt_morph/run.sh --config experiments/mvp_d_ckpt_morph/config_chain_boost.yaml --mode render --input "$OUT_DIR/step_2_d.wav" --output "$OUT_DIR/step_3_d.wav" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED at Step 3" >> "$LOG"; exit 1; fi

# Step 4: A
echo "--- Step 4: A ---" >> "$LOG"
bash experiments/mvp_a_rave_latent/run.sh --config experiments/mvp_a_rave_latent/config_chain_boost.yaml --mode render --input "$OUT_DIR/step_3_d.wav" --output "$OUT_DIR/step_4_a.wav" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED at Step 4" >> "$LOG"; exit 1; fi

# Step 5: A
echo "--- Step 5: A ---" >> "$LOG"
bash experiments/mvp_a_rave_latent/run.sh --config experiments/mvp_a_rave_latent/config_chain_boost.yaml --mode render --input "$OUT_DIR/step_4_a.wav" --output "$OUT_DIR/step_5_a.wav" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED at Step 5" >> "$LOG"; exit 1; fi

# Step 6: D
echo "--- Step 6: D ---" >> "$LOG"
bash experiments/mvp_d_ckpt_morph/run.sh --config experiments/mvp_d_ckpt_morph/config_chain_boost.yaml --mode render --input "$OUT_DIR/step_5_a.wav" --output "$OUT_DIR/step_6_d.wav" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED at Step 6" >> "$LOG"; exit 1; fi

# Step 7: A
echo "--- Step 7: A ---" >> "$LOG"
bash experiments/mvp_a_rave_latent/run.sh --config experiments/mvp_a_rave_latent/config_chain_boost.yaml --mode render --input "$OUT_DIR/step_6_d.wav" --output "$OUT_DIR/step_7_a.wav" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED at Step 7" >> "$LOG"; exit 1; fi

echo "DONE" >> "$LOG"
