#!/bin/bash

LOG_DIR="experiments/throughtput/logs"
mkdir -p "$LOG_DIR"

for D_MEM in  512; do
    for BS in  12000 16000 20000 24000 28000  32000  36000; do
        echo "=== Running d_mem=$D_MEM batch_size=$BS ==="
        torchrun --nproc_per_node=3 run.py \
            model=tgn dataset=LASTFM \
            compile=false compute.tf32=false \
            batch_size=$BS \
            mem_updater.d_mem=$D_MEM \
            epoch=5 \
            2>&1 | tee "$LOG_DIR/dmem_${D_MEM}_bs_${BS}.log"
    done
done
