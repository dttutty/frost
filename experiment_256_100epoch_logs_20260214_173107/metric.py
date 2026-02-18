import glob
import re
import statistics


pattern = re.compile(r"Best Val AP: .*?\| Test AP: ([0-9.]+) \| Test AUC: ([0-9.]+)")

test_aps = []
test_aucs = []

for log_file in glob.glob("run_*.log"):
    with open(log_file, encoding="utf-8") as f:
        content = f.read()

    matches = pattern.findall(content)
    if not matches:
        print(f"[WARN] No match found in {log_file}")
        continue

    # 取最后一次出现的结果
    last_ap, last_auc = matches[-1]
    test_aps.append(float(last_ap))
    test_aucs.append(float(last_auc))

if not test_aps:
    print("No valid results found.")
else:
    avg_ap = statistics.mean(test_aps)
    avg_auc = statistics.mean(test_aucs)

    print(f"Runs counted: {len(test_aps)}")
    print(f"Average Test AP:  {avg_ap:.6f}")
    print(f"Average Test AUC: {avg_auc:.6f}")
