# Force Agg backend
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Data
mem_dim = [128, 256, 512]
speedup = [1.56529921, 2.443617021, 2.945585216]

auc_my = [0.881625, 0.893095, 0.903621]
auc_tgl = [0.883683, 0.889914, 0.892557]

ap_my = [0.86626, 0.8788, 0.891163]
ap_tgl = [0.86989, 0.876261, 0.877626]

# X labels with speedup (2 decimal places + 'x')
x_labels = [f"{m} ({s:.2f}x)" for m, s in zip(mem_dim, speedup)]
# --------- Plot 1: AP ---------
plt.figure()
plt.plot(x_labels, ap_my, color="red", marker="o")
plt.plot(x_labels, ap_tgl, color="black", marker="o")
plt.xlabel("mem_dim (speedup)")
plt.ylabel("AP")
plt.title("AP vs mem_dim")
plt.tight_layout()
plt.savefig("ap_vs_mem_dim.png", dpi=300)
plt.close()

# --------- Plot 2: AUC ---------
plt.figure()
plt.plot(x_labels, auc_my, color="red", marker="o")
plt.plot(x_labels, auc_tgl, color="black", marker="o")
plt.xlabel("mem_dim (speedup)")
plt.ylabel("AUC")
plt.title("AUC vs mem_dim")
plt.tight_layout()
plt.savefig("auc_vs_mem_dim.png", dpi=300)
plt.close()
