import re
import csv
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
OUTPUT = Path(__file__).parent / "result.csv"

# Matches: [GPU 0] Training throughput: 12345.6 events/sec
PATTERN = re.compile(r"\[GPU (\d+)\] Training throughput: ([\d.]+) events/sec")

rows = []
for log_file in sorted(LOG_DIR.glob("dmem_*_bs_*.log")):
    # filename: dmem_<D_MEM>_bs_<BS>.log
    m = re.match(r"dmem_(\d+)_bs_(\d+)\.log", log_file.name)
    if not m:
        continue
    d_mem = int(m.group(1))
    bs = int(m.group(2))

    gpu_throughputs = {}
    with open(log_file) as f:
        for line in f:
            tm = PATTERN.search(line)
            if tm:
                gpu = int(tm.group(1))
                throughput = float(tm.group(2))
                gpu_throughputs[gpu] = throughput

    if gpu_throughputs:
        total = sum(gpu_throughputs.values())
        for gpu, tp in sorted(gpu_throughputs.items()):
            rows.append({"d_mem": d_mem, "batch_size": bs, "gpu": gpu, "throughput_events_per_sec": tp})
        rows.append({"d_mem": d_mem, "batch_size": bs, "gpu": -1, "throughput_events_per_sec": total})
    else:
        print(f"WARNING: no throughput found in {log_file.name}")

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["d_mem", "batch_size", "gpu", "throughput_events_per_sec"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {OUTPUT}")
