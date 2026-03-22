#!/bin/bash
set -euo pipefail

BASE_URL="https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl"
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/DATA"

download() {
    local dataset=$1
    shift
    mkdir -p "$DATA_DIR/$dataset"
    for file in "$@"; do
        wget -nc -P "$DATA_DIR/$dataset" "$BASE_URL/$dataset/$file"
    done
}

download GDELT   int_train.npz int_full.npz node_features.pt labels.csv ext_full.npz edges.csv edge_features.pt
download LASTFM  edges.csv ext_full.npz int_full.npz int_train.npz
download MAG     int_train.npz labels.csv int_full.npz ext_full.npz edges.csv node_features.pt
download MOOC    edges.csv ext_full.npz int_full.npz int_train.npz
download REDDIT  edge_features.pt edges.csv ext_full.npz int_full.npz int_train.npz labels.csv
download WIKI    edge_features.pt edges.csv ext_full.npz int_full.npz int_train.npz labels.csv
