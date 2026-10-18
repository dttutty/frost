#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_IDS="${GPU_IDS:-0,1}"
SERIAL_GPU="${SERIAL_GPU:-0}"
SEEDS_RAW="${SEEDS:-0}"
MEMORY_BASED_MODELS_RAW="${MEMORY_BASED_MODELS:-tgn jodie apan}"
MEMORYLESS_MODELS_RAW="${MEMORYLESS_MODELS:-tgat cawn tcl graph_mixer dygformer}"
DATASETS_RAW="${DATASETS:-LASTFM MOOC REDDIT WIKIPEDIA}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
MEASURE_BATCHES="${MEASURE_BATCHES:-100}"
SERIAL_MAX_TRAIN_STEPS="${SERIAL_MAX_TRAIN_STEPS:-$(( WARMUP_BATCHES + MEASURE_BATCHES + 1 ))}"
DIST_MAX_TRAIN_STEPS="${DIST_MAX_TRAIN_STEPS:-$(( WARMUP_BATCHES + MEASURE_BATCHES + 1 ))}"
DYGLIB_MAX_TRAIN_BATCHES="${DYGLIB_MAX_TRAIN_BATCHES:-$(( WARMUP_BATCHES + MEASURE_BATCHES ))}"
FROST_MEMORYLESS_QUEUE_SIZE="${FROST_MEMORYLESS_QUEUE_SIZE:-16}"
RUN_MEMORY_BASED="${RUN_MEMORY_BASED:-1}"
RUN_MEMORYLESS="${RUN_MEMORYLESS:-1}"

if (( WARMUP_BATCHES < 0 )); then
  echo "WARMUP_BATCHES must be >= 0, got ${WARMUP_BATCHES}" >&2
  exit 1
fi
if (( MEASURE_BATCHES < 1 )); then
  echo "MEASURE_BATCHES must be >= 1, got ${MEASURE_BATCHES}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} != 2 )); then
  echo "This benchmark expects exactly 2 GPUs for the distributed runs. Got GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi

SEEDS_NORM="${SEEDS_RAW//,/ }"
MEMORY_BASED_MODELS_NORM="${MEMORY_BASED_MODELS_RAW//,/ }"
MEMORYLESS_MODELS_NORM="${MEMORYLESS_MODELS_RAW//,/ }"
DATASETS_NORM="${DATASETS_RAW//,/ }"
IFS=' ' read -r -a SEEDS <<< "${SEEDS_NORM}"
IFS=' ' read -r -a MEMORY_BASED_MODELS <<< "${MEMORY_BASED_MODELS_NORM}"
IFS=' ' read -r -a MEMORYLESS_MODELS <<< "${MEMORYLESS_MODELS_NORM}"
IFS=' ' read -r -a DATASETS <<< "${DATASETS_NORM}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/experiments/memory/com_memory_footprint/memory_footprint_${STAMP}}"
CFG_DIR="${RUN_DIR}/configs"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${CFG_DIR}" "${LOG_DIR}"

PROGRESS_MD="${RUN_DIR}/progress.md"
MB_RESULTS_TSV="${RUN_DIR}/memory_based_results.tsv"
MB_SUMMARY_TSV="${RUN_DIR}/memory_based_summary.tsv"
ML_RESULTS_TSV="${RUN_DIR}/memoryless_results.tsv"
ML_SUMMARY_TSV="${RUN_DIR}/memoryless_summary.tsv"
REPORT_MD="${RUN_DIR}/report.md"

cat > "${PROGRESS_MD}" <<EOF
# GPU Memory Footprint Benchmark

- Run dir: ${RUN_DIR}
- GPU_IDS (distributed): ${GPU_IDS}
- SERIAL_GPU: ${SERIAL_GPU}
- Warmup train batches: ${WARMUP_BATCHES}
- Measured train batches: ${MEASURE_BATCHES}
- Metric: stable-state GPU footprint from measured-window train batch-end snapshots
- Snapshot fields: \`torch.cuda.memory_reserved()\` and \`torch.cuda.memory_allocated()\`
- Summary focus: reserved MiB
- CPU-resident feature policy:
  - FROST: default host-memory feature loading
  - TGL train.py / train_dist.py: CPU/shared-memory features with per-batch transfer
  - DyGLib: \`features_in_host_memory=True\`

## Progress

EOF

printf "group\tmodel\tdataset\tseed\tsystem\tn_gpus\tmacro_batch_size\tper_gpu_batch_size\treserved_mib_mean_per_gpu_max\treserved_mib_p90_per_gpu_max\treserved_mib_max_per_gpu_max\treserved_mib_mean_total\tallocated_mib_mean_per_gpu_max\tallocated_mib_mean_total\tmemory_measured_batches\tlog\n" > "${MB_RESULTS_TSV}"
printf "group\tmodel\tdataset\tseed\tsystem\tn_gpus\tmacro_batch_size\tper_gpu_batch_size\treserved_mib_mean_per_gpu_max\treserved_mib_p90_per_gpu_max\treserved_mib_max_per_gpu_max\treserved_mib_mean_total\tallocated_mib_mean_per_gpu_max\tallocated_mib_mean_total\tmemory_measured_batches\tlog\n" > "${ML_RESULTS_TSV}"

memory_based_macro_batch_size() {
  local dataset="${1^^}"
  case "${dataset}" in
    WIKI|WIKIPEDIA) echo 600 ;;
    MOOC) echo 1200 ;;
    LASTFM) echo 2000 ;;
    REDDIT) echo 2000 ;;
    *) echo 600 ;;
  esac
}

memoryless_macro_batch_size() {
  local dataset="${1^^}"
  case "${dataset}" in
    WIKI|WIKIPEDIA|MOOC|LASTFM|REDDIT) echo 600 ;;
    *) echo 600 ;;
  esac
}

tgl_base_config_for_model() {
  local model="${1,,}"
  case "${model}" in
    tgn) echo "${REPO_ROOT}/third_party/tgl/config/frost_parity/TGN.yml" ;;
    jodie) echo "${REPO_ROOT}/third_party/tgl/config/frost_parity/JODIE.yml" ;;
    apan) echo "${REPO_ROOT}/third_party/tgl/config/frost_parity/APAN.yml" ;;
    *)
      echo "Unsupported TGL memory-based model=${model}" >&2
      return 1
      ;;
  esac
}

dyglib_model_name() {
  local model="${1,,}"
  case "${model}" in
    tgat) echo "TGAT" ;;
    cawn) echo "CAWN" ;;
    tcl) echo "TCL" ;;
    graph_mixer|graphmixer) echo "GraphMixer" ;;
    dygformer) echo "DyGFormer" ;;
    *)
      echo "Unsupported DyGLib model=${model}" >&2
      return 1
      ;;
  esac
}

dyglib_dataset_name() {
  local dataset="${1^^}"
  case "${dataset}" in
    WIKI|WIKIPEDIA) echo "wikipedia" ;;
    LASTFM) echo "lastfm" ;;
    MOOC) echo "mooc" ;;
    REDDIT) echo "reddit" ;;
    *)
      echo "Unsupported DyGLib dataset=${dataset}" >&2
      return 1
      ;;
  esac
}

write_tgl_config() {
  local src_cfg="$1"
  local dst_cfg="$2"
  local epoch="$3"
  local batch_size="$4"
  sed -E \
    -e "0,/epoch:[[:space:]]*[0-9]+/s//epoch: ${epoch}/" \
    -e "0,/batch_size:[[:space:]]*[0-9]+/s//batch_size: ${batch_size}/" \
    "${src_cfg}" > "${dst_cfg}"
}

parse_memory_summary_json() {
  local log_path="$1"
  local prefix="$2"
  uv run python - "$log_path" "$prefix" <<'PY'
import json
import math
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
text = log_path.read_text(encoding="utf-8", errors="ignore")
decoder = json.JSONDecoder()
rows = []
for match in re.finditer(re.escape(prefix) + r"\s+", text):
    start = match.end()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        continue
    rows.append(obj)
if not rows:
    raise SystemExit(f"Could not parse {prefix} from {log_path}")

by_rank = {}
for idx, row in enumerate(rows):
    rank = row.get("rank", idx)
    try:
        rank = int(rank)
    except Exception:
        rank = idx
    by_rank[rank] = row
rows = [by_rank[k] for k in sorted(by_rank)]

def get_float(row, key):
    value = row.get(key)
    if value is None:
        return float("nan")
    return float(value)

reserved_means = [get_float(row, "reserved_mib_mean") for row in rows]
reserved_p90s = [get_float(row, "reserved_mib_p90") for row in rows]
reserved_maxes = [get_float(row, "reserved_mib_max") for row in rows]
allocated_means = [get_float(row, "allocated_mib_mean") for row in rows]
memory_measured = [int(row.get("memory_measured_batches", -1)) for row in rows]

def finite(values):
    return [v for v in values if math.isfinite(v)]

reserved_means_f = finite(reserved_means)
reserved_p90s_f = finite(reserved_p90s)
reserved_maxes_f = finite(reserved_maxes)
allocated_means_f = finite(allocated_means)

print(
    "\t".join(
        [
            str(len(rows)),
            f"{max(reserved_means_f) if reserved_means_f else float('nan'):.6f}",
            f"{max(reserved_p90s_f) if reserved_p90s_f else float('nan'):.6f}",
            f"{max(reserved_maxes_f) if reserved_maxes_f else float('nan'):.6f}",
            f"{sum(reserved_means_f) if reserved_means_f else float('nan'):.6f}",
            f"{max(allocated_means_f) if allocated_means_f else float('nan'):.6f}",
            f"{sum(allocated_means_f) if allocated_means_f else float('nan'):.6f}",
            str(min(memory_measured) if memory_measured else -1),
        ]
    )
)
PY
}

append_result_row() {
  local out_tsv="$1"
  shift
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${out_tsv}"
}

echo "=== GPU Memory Footprint Benchmark ==="
echo "Run dir: ${RUN_DIR}"
echo "Distributed GPUs: ${GPU_IDS}"
echo "Serial GPU: ${SERIAL_GPU}"
echo "Warmup batches: ${WARMUP_BATCHES}"
echo "Measure batches: ${MEASURE_BATCHES}"

combo_idx=0

if (( RUN_MEMORY_BASED == 1 )); then
  echo
  echo "=== Memory-based models: FROST vs TGL train.py vs TGL train_dist.py ==="
  for model in "${MEMORY_BASED_MODELS[@]}"; do
    model_lc="${model,,}"
    model_uc="${model_lc^^}"
    base_cfg="$(tgl_base_config_for_model "${model_lc}")"
    for dataset in "${DATASETS[@]}"; do
      dataset_uc="${dataset^^}"
      dataset_lc="${dataset,,}"
      macro_batch_size="$(memory_based_macro_batch_size "${dataset_uc}")"
      per_gpu_batch_size="$(( macro_batch_size / 2 ))"
      dist_cfg="${CFG_DIR}/${model_lc}_${dataset_lc}_dist_bs${per_gpu_batch_size}.yml"
      serial_cfg="${CFG_DIR}/${model_lc}_${dataset_lc}_serial_bs${macro_batch_size}.yml"
      write_tgl_config "${base_cfg}" "${dist_cfg}" 1000 "${per_gpu_batch_size}"
      write_tgl_config "${base_cfg}" "${serial_cfg}" 1000 "${macro_batch_size}"
      for seed in "${SEEDS[@]}"; do
        combo_idx=$(( combo_idx + 1 ))
        frost_log="${LOG_DIR}/frost_mem_${model_lc}_${dataset_lc}_seed${seed}.log"
        tgl_serial_log="${LOG_DIR}/tgl_train_mem_${model_lc}_${dataset_lc}_seed${seed}.log"
        tgl_dist_log="${LOG_DIR}/tgl_dist_mem_${model_lc}_${dataset_lc}_seed${seed}.log"
        master_port=$(( 38000 + combo_idx ))

        echo
        echo "=== [Memory-based] ${model_uc} ${dataset_uc} seed=${seed} macro_batch=${macro_batch_size} ==="

        echo "[FROST] launching..."
        if ! CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
          FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH="${WARMUP_BATCHES}" \
          FROST_THROUGHPUT_MEASURE_BATCHES="${MEASURE_BATCHES}" \
          uv run python -m frost_bench.cli.train "${model_lc}" \
            --nproc-per-node 3 \
            --master-port "${master_port}" \
            "dataset=${dataset_uc}" \
            "epoch=1" \
            "macro_batch_size=${macro_batch_size}" \
            "seed=${seed}" \
            "compile=false" \
            "compute.tf32=false" \
            "tqdm=false" \
            > "${frost_log}" 2>&1; then
          echo "FROST memory run failed. See ${frost_log}" >&2
          echo "- FAILED memory-based FROST: ${model_lc}/${dataset_uc}/seed${seed}" >> "${PROGRESS_MD}"
          exit 1
        fi
        read -r frost_ngpu frost_res_mean frost_res_p90 frost_res_max frost_res_total frost_alloc_mean frost_alloc_total frost_measured < <(parse_memory_summary_json "${frost_log}" "APPENDIX_TRAIN_MEMORY_SUMMARY")
        append_result_row "${MB_RESULTS_TSV}" \
          "memory_based" "${model_lc}" "${dataset_uc}" "${seed}" "frost" "${frost_ngpu}" "${macro_batch_size}" "${per_gpu_batch_size}" \
          "${frost_res_mean}" "${frost_res_p90}" "${frost_res_max}" "${frost_res_total}" \
          "${frost_alloc_mean}" "${frost_alloc_total}" "${frost_measured}" "${frost_log}"
        echo "[FROST] reserved_mib_mean_per_gpu_max=${frost_res_mean} total_mean=${frost_res_total}"

        echo "[TGL train.py] launching..."
        if ! CUDA_VISIBLE_DEVICES="${SERIAL_GPU}" \
          uv run --project "${REPO_ROOT}/third_party/tgl" python "${REPO_ROOT}/third_party/tgl/train.py" \
            --data "${dataset_uc}" \
            --config "${serial_cfg}" \
            --gpu 0 \
            --seed "${seed}" \
            --batch_size "${macro_batch_size}" \
            --pin_memory \
            --appendix-train-e2e \
            --measure-start-epoch 0 \
            --warmup-batches "${WARMUP_BATCHES}" \
            --measure-batches "${MEASURE_BATCHES}" \
            --max-train-steps "${SERIAL_MAX_TRAIN_STEPS}" \
            > "${tgl_serial_log}" 2>&1; then
          echo "TGL train.py memory run failed. See ${tgl_serial_log}" >&2
          echo "- FAILED memory-based TGL-train: ${model_lc}/${dataset_uc}/seed${seed}" >> "${PROGRESS_MD}"
          exit 1
        fi
        read -r serial_ngpu serial_res_mean serial_res_p90 serial_res_max serial_res_total serial_alloc_mean serial_alloc_total serial_measured < <(parse_memory_summary_json "${tgl_serial_log}" "APPENDIX_TRAIN_E2E_SUMMARY")
        append_result_row "${MB_RESULTS_TSV}" \
          "memory_based" "${model_lc}" "${dataset_uc}" "${seed}" "tgl_train" "${serial_ngpu}" "${macro_batch_size}" "${macro_batch_size}" \
          "${serial_res_mean}" "${serial_res_p90}" "${serial_res_max}" "${serial_res_total}" \
          "${serial_alloc_mean}" "${serial_alloc_total}" "${serial_measured}" "${tgl_serial_log}"
        echo "[TGL train.py] reserved_mib_mean_per_gpu_max=${serial_res_mean} total_mean=${serial_res_total}"

        echo "[TGL train_dist.py] launching..."
        if ! CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
          uv run --project "${REPO_ROOT}/third_party/tgl" python "${REPO_ROOT}/third_party/tgl/train_dist.py" \
            --dataset "${dataset_lc}" \
            --config "${dist_cfg}" \
            --num_gpus 2 \
            --seed "${seed}" \
            --rnd_edim 0 \
            --rnd_ndim 0 \
            --appendix-train-e2e \
            --measure-start-epoch 0 \
            --warmup-batches "${WARMUP_BATCHES}" \
            --measure-batches "${MEASURE_BATCHES}" \
            --max-train-steps "${DIST_MAX_TRAIN_STEPS}" \
            > "${tgl_dist_log}" 2>&1; then
          echo "TGL train_dist.py memory run failed. See ${tgl_dist_log}" >&2
          echo "- FAILED memory-based TGL-dist: ${model_lc}/${dataset_uc}/seed${seed}" >> "${PROGRESS_MD}"
          exit 1
        fi
        read -r dist_ngpu dist_res_mean dist_res_p90 dist_res_max dist_res_total dist_alloc_mean dist_alloc_total dist_measured < <(parse_memory_summary_json "${tgl_dist_log}" "APPENDIX_TRAIN_E2E_SUMMARY")
        append_result_row "${MB_RESULTS_TSV}" \
          "memory_based" "${model_lc}" "${dataset_uc}" "${seed}" "tgl_dist" "${dist_ngpu}" "${macro_batch_size}" "${per_gpu_batch_size}" \
          "${dist_res_mean}" "${dist_res_p90}" "${dist_res_max}" "${dist_res_total}" \
          "${dist_alloc_mean}" "${dist_alloc_total}" "${dist_measured}" "${tgl_dist_log}"
        echo "[TGL train_dist.py] reserved_mib_mean_per_gpu_max=${dist_res_mean} total_mean=${dist_res_total}"

        echo "- DONE memory-based: ${model_lc}/${dataset_uc}/seed${seed}" >> "${PROGRESS_MD}"
      done
    done
  done
fi

if (( RUN_MEMORYLESS == 1 )); then
  echo
  echo "=== Memoryless models: FROST vs DyGLib ==="
  for model in "${MEMORYLESS_MODELS[@]}"; do
    model_lc="${model,,}"
    dyglib_model="$(dyglib_model_name "${model_lc}")"
    for dataset in "${DATASETS[@]}"; do
      dataset_uc="${dataset^^}"
      dyglib_dataset="$(dyglib_dataset_name "${dataset_uc}")"
      macro_batch_size="$(memoryless_macro_batch_size "${dataset_uc}")"
      per_gpu_batch_size="$(( macro_batch_size / 2 ))"
      for seed in "${SEEDS[@]}"; do
        combo_idx=$(( combo_idx + 1 ))
        frost_log="${LOG_DIR}/frost_memless_${model_lc}_${dyglib_dataset}_seed${seed}.log"
        dyglib_log="${LOG_DIR}/dyglib_mem_${model_lc}_${dyglib_dataset}_seed${seed}.log"
        master_port=$(( 39000 + combo_idx ))

        echo
        echo "=== [Memoryless] ${model_lc} ${dataset_uc} seed=${seed} macro_batch=${macro_batch_size} ==="

        echo "[FROST] launching..."
        if ! CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
          FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH="${WARMUP_BATCHES}" \
          FROST_THROUGHPUT_MEASURE_BATCHES="${MEASURE_BATCHES}" \
          uv run python -m frost_bench.cli.train "${model_lc}" \
            --nproc-per-node 3 \
            --master-port "${master_port}" \
            "dataset=${dataset_uc}" \
            "epoch=1" \
            "macro_batch_size=${macro_batch_size}" \
            "seed=${seed}" \
            "compile=false" \
            "compute.tf32=false" \
            "stable_mode=false" \
            "no_memory_queue_size=${FROST_MEMORYLESS_QUEUE_SIZE}" \
            "tqdm=false" \
            > "${frost_log}" 2>&1; then
          echo "FROST memoryless run failed. See ${frost_log}" >&2
          echo "- FAILED memoryless FROST: ${model_lc}/${dataset_uc}/seed${seed}" >> "${PROGRESS_MD}"
          exit 1
        fi
        read -r frost_ngpu frost_res_mean frost_res_p90 frost_res_max frost_res_total frost_alloc_mean frost_alloc_total frost_measured < <(parse_memory_summary_json "${frost_log}" "APPENDIX_TRAIN_MEMORY_SUMMARY")
        append_result_row "${ML_RESULTS_TSV}" \
          "memoryless" "${model_lc}" "${dataset_uc}" "${seed}" "frost" "${frost_ngpu}" "${macro_batch_size}" "${per_gpu_batch_size}" \
          "${frost_res_mean}" "${frost_res_p90}" "${frost_res_max}" "${frost_res_total}" \
          "${frost_alloc_mean}" "${frost_alloc_total}" "${frost_measured}" "${frost_log}"
        echo "[FROST] reserved_mib_mean_per_gpu_max=${frost_res_mean} total_mean=${frost_res_total}"

        echo "[DyGLib] launching..."
        if ! (
          cd "${REPO_ROOT}/third_party/DyGLib"
          CUDA_VISIBLE_DEVICES="${SERIAL_GPU}" \
            uv run python train_link_prediction.py \
              --seed-start "${seed}" \
              --dataset_name "${dyglib_dataset}" \
              --model_name "${dyglib_model}" \
              --load_best_configs \
              --gpu 0 \
              --num_runs 1 \
              --num_epochs 1 \
              --batch_size "${macro_batch_size}" \
              --appendix-train-e2e \
              --warmup-batches "${WARMUP_BATCHES}" \
              --measure-batches "${MEASURE_BATCHES}" \
              --max-train-batches "${DYGLIB_MAX_TRAIN_BATCHES}" \
              --features-in-host-memory
        ) > "${dyglib_log}" 2>&1; then
          echo "DyGLib memory run failed. See ${dyglib_log}" >&2
          echo "- FAILED memoryless DyGLib: ${model_lc}/${dataset_uc}/seed${seed}" >> "${PROGRESS_MD}"
          exit 1
        fi
        read -r dyglib_ngpu dyglib_res_mean dyglib_res_p90 dyglib_res_max dyglib_res_total dyglib_alloc_mean dyglib_alloc_total dyglib_measured < <(parse_memory_summary_json "${dyglib_log}" "APPENDIX_TRAIN_E2E_SUMMARY")
        append_result_row "${ML_RESULTS_TSV}" \
          "memoryless" "${model_lc}" "${dataset_uc}" "${seed}" "dyglib" "${dyglib_ngpu}" "${macro_batch_size}" "${macro_batch_size}" \
          "${dyglib_res_mean}" "${dyglib_res_p90}" "${dyglib_res_max}" "${dyglib_res_total}" \
          "${dyglib_alloc_mean}" "${dyglib_alloc_total}" "${dyglib_measured}" "${dyglib_log}"
        echo "[DyGLib] reserved_mib_mean_per_gpu_max=${dyglib_res_mean} total_mean=${dyglib_res_total}"

        echo "- DONE memoryless: ${model_lc}/${dataset_uc}/seed${seed}" >> "${PROGRESS_MD}"
      done
    done
  done
fi

uv run python - "${MB_RESULTS_TSV}" "${MB_SUMMARY_TSV}" "${ML_RESULTS_TSV}" "${ML_SUMMARY_TSV}" "${REPORT_MD}" <<'PY'
import csv
import pathlib
import statistics
import sys

mb_results = pathlib.Path(sys.argv[1])
mb_summary = pathlib.Path(sys.argv[2])
ml_results = pathlib.Path(sys.argv[3])
ml_summary = pathlib.Path(sys.argv[4])
report_path = pathlib.Path(sys.argv[5])

def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)

def write_summary(input_rows, output_path, comparison_groups):
    summary_rows = []
    grouped = {}
    for row in input_rows:
        key = (row["model"], row["dataset"], row["seed"])
        grouped.setdefault(key, {})[row["system"]] = row
    for (model, dataset, seed), systems in sorted(grouped.items()):
        for left, right in comparison_groups:
            if left not in systems or right not in systems:
                continue
            a = systems[left]
            b = systems[right]
            a_per_gpu = float(a["reserved_mib_mean_per_gpu_max"])
            b_per_gpu = float(b["reserved_mib_mean_per_gpu_max"])
            a_total = float(a["reserved_mib_mean_total"])
            b_total = float(b["reserved_mib_mean_total"])
            summary_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "lhs": left,
                    "rhs": right,
                    "lhs_reserved_mib_mean_per_gpu_max": f"{a_per_gpu:.6f}",
                    "rhs_reserved_mib_mean_per_gpu_max": f"{b_per_gpu:.6f}",
                    "lhs_reserved_mib_mean_total": f"{a_total:.6f}",
                    "rhs_reserved_mib_mean_total": f"{b_total:.6f}",
                    "ratio_per_gpu_lhs_over_rhs": f"{(a_per_gpu / b_per_gpu) if b_per_gpu > 0 else float('nan'):.6f}",
                    "ratio_total_lhs_over_rhs": f"{(a_total / b_total) if b_total > 0 else float('nan'):.6f}",
                }
            )
    if summary_rows:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        output_path.write_text("", encoding="utf-8")
    return summary_rows

mb_rows = read_rows(mb_results)
ml_rows = read_rows(ml_results)
mb_summary_rows = write_summary(mb_rows, mb_summary, [("frost", "tgl_train"), ("frost", "tgl_dist")])
ml_summary_rows = write_summary(ml_rows, ml_summary, [("frost", "dyglib")])

def fmt_ratio(xs):
    return f"{min(xs):.3f}x - {max(xs):.3f}x (mean {statistics.mean(xs):.3f}x)"

lines = []
lines.append("# GPU Memory Footprint Report")
lines.append("")
lines.append("- Metric: stable-state GPU memory from measured-window train batch-end snapshots")
lines.append("- Field used for comparison: `reserved_mib_mean_per_gpu_max` by default, plus `reserved_mib_mean_total`")
lines.append("- Interpretation: ratio < 1 means the left-hand system uses less GPU memory")
lines.append("")
if mb_summary_rows:
    lines.append("## Memory-based Models")
    lines.append("")
    lines.append("| Comparison | Ratio range (per-GPU max reserved mean) | Ratio range (total reserved mean) |")
    lines.append("| --- | ---: | ---: |")
    for lhs, rhs in [("frost", "tgl_train"), ("frost", "tgl_dist")]:
        rows = [r for r in mb_summary_rows if r["lhs"] == lhs and r["rhs"] == rhs]
        if rows:
            per_gpu = [float(r["ratio_per_gpu_lhs_over_rhs"]) for r in rows]
            total = [float(r["ratio_total_lhs_over_rhs"]) for r in rows]
            lines.append(f"| {lhs} vs {rhs} | {fmt_ratio(per_gpu)} | {fmt_ratio(total)} |")
    lines.append("")
if ml_summary_rows:
    lines.append("## Memoryless Models")
    lines.append("")
    rows = [r for r in ml_summary_rows if r["lhs"] == "frost" and r["rhs"] == "dyglib"]
    if rows:
        per_gpu = [float(r["ratio_per_gpu_lhs_over_rhs"]) for r in rows]
        total = [float(r["ratio_total_lhs_over_rhs"]) for r in rows]
        lines.append("| Comparison | Ratio range (per-GPU max reserved mean) | Ratio range (total reserved mean) |")
        lines.append("| --- | ---: | ---: |")
        lines.append(f"| frost vs dyglib | {fmt_ratio(per_gpu)} | {fmt_ratio(total)} |")
        lines.append("")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo
echo "Finished. Outputs:"
echo "  ${RUN_DIR}"
echo "  ${MB_RESULTS_TSV}"
echo "  ${MB_SUMMARY_TSV}"
echo "  ${ML_RESULTS_TSV}"
echo "  ${ML_SUMMARY_TSV}"
echo "  ${REPORT_MD}"
