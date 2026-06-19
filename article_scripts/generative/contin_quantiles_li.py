import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re
from itertools import combinations

plt.style.use(["science", "notebook", "grid"])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
})

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def threshold_agreement_at_quantile(A, B, q):
    """Agreement when both select top (1-q) proportion."""
    tau_A = np.quantile(A, q)
    tau_B = np.quantile(B, q)
    return np.mean((A >= tau_A) == (B >= tau_B))

def compute_stability_curve(A, B, n_points=100):
    """Return (quantiles, agreements) for a pair."""
    qs = np.linspace(0, 1, n_points)
    agreements = [threshold_agreement_at_quantile(A, B, q) for q in qs]
    return qs, np.array(agreements)

# ------------------------------------------------------------
# Load data (same as before)
# ------------------------------------------------------------

base_path = Path("./output/WOS/lithium")
files = list(base_path.glob("R_*_IR_*_OR_*_T_0_MODEL_llama3.2_*.csv"))

configs, scores_list = [], []
for file in files:
    match = re.search(
        r"R_(.*?)_IR_(.*?)_OR_(.*?)_T_0_MODEL_llama3\.2_(\d+)\.csv",
        file.name,
    )
    if not match:
        continue
    r, ir, order = match.group(1), match.group(2), match.group(3)
    order_short = "RF" if order == "relevant_first" else "IF"
    label = rf"R{r}-IR{ir}-{order_short}"
    
    df = pd.read_csv(file)
    if r not in df.columns:
        continue
        
    configs.append(label)
    scores_list.append(df[r].values)

# ------------------------------------------------------------
# Select informative pairs to plot
# ------------------------------------------------------------

# def select_informative_pairs(configs, scores_list, n_pairs=6):
#     """Pick pairs with max/min stability and single-parameter differences."""
#     n = len(scores_list)
#     pairs = []
#     
#     # Compute overall PTS for all pairs (average over quantiles)
#     pts_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             qs, ags = compute_stability_curve(scores_list[i], scores_list[j])
#             pts_matrix[i, j] = np.mean(ags)
#     
#     # Find most/least stable pairs
#     triu_idx = np.triu_indices(n, k=1)
#     stability_vals = pts_matrix[triu_idx]
#     
#     max_idx = np.argmax(stability_vals)
#     min_idx = np.argmin(stability_vals)
#     i_max, j_max = triu_idx[0][max_idx], triu_idx[1][max_idx]
#     i_min, j_min = triu_idx[0][min_idx], triu_idx[1][min_idx]
#     
#     pairs.append((i_max, j_max, "Most stable", "green"))
#     pairs.append((i_min, j_min, "Least stable", "red"))
#     
#     # Add pairs that differ in only one parameter (e.g., same R/IR, different order)
#     
#     for i, j in combinations(range(n), 2):
#         cfg_i = configs[i].split("-")
#         cfg_j = configs[j].split("-")
#         diffs = sum(a != b for a, b in zip(cfg_i, cfg_j))
#         if diffs == 1:  # Single-parameter difference
#             # label = f"{configs[i]}↔{configs[j]}"
#             label = f"{configs[i]} $\\leftrightarrow$ {configs[j]}"
#             pairs.append((i, j, label, "blue"))
#             if len(pairs) >= n_pairs:
#                 break
#     
#     return pairs[:n_pairs]
# ------------------------------------------------------------
# Updated pair selection (no hardcoded colors)
# ------------------------------------------------------------
def select_informative_pairs(configs, scores_list, n_pairs=6):
    """Pick pairs with max/min stability and single-parameter differences."""
    n = len(scores_list)
    pairs = []
    
    # Compute overall PTS for all pairs
    pts_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            qs, ags = compute_stability_curve(scores_list[i], scores_list[j])
            pts_matrix[i, j] = np.mean(ags)
    
    # Most/least stable pairs
    triu_idx = np.triu_indices(n, k=1)
    stability_vals = pts_matrix[triu_idx]
    
    max_idx = np.argmax(stability_vals)
    min_idx = np.argmin(stability_vals)
    i_max, j_max = triu_idx[0][max_idx], triu_idx[1][max_idx]
    i_min, j_min = triu_idx[0][min_idx], triu_idx[1][min_idx]
    
    pairs.append((i_max, j_max, "Most stable"))
    pairs.append((i_min, j_min, "Least stable"))
    
    # Single-parameter differences
    for i, j in combinations(range(n), 2):
        cfg_i = configs[i].split("-")
        cfg_j = configs[j].split("-")
        diffs = sum(a != b for a, b in zip(cfg_i, cfg_j))
        if diffs == 1:
            label = f"{configs[i]} vs. {configs[j]}"
            pairs.append((i, j, label))
            if len(pairs) >= n_pairs:
                break
                
    return pairs

# ------------------------------------------------------------
# Plot continuous stability curves
# ------------------------------------------------------------
pairs_to_plot = select_informative_pairs(configs, scores_list)

# Generate distinct colors for each pair
cmap = plt.get_cmap("tab20")  # 20 highly distinguishable colors
colors = [cmap(i) for i in range(len(pairs_to_plot))]

fig, ax = plt.subplots(figsize=(9, 6))

for idx, (i, j, label) in enumerate(pairs_to_plot):
    qs, ags = compute_stability_curve(scores_list[i], scores_list[j])
    ax.plot(qs, ags, label=label, color=colors[idx], linewidth=2.5)

# Optional: overlay mean ± std across ALL pairs
all_agreements = []
for i, j in combinations(range(len(scores_list)), 2):
    _, ags = compute_stability_curve(scores_list[i], scores_list[j])
    all_agreements.append(ags)
if all_agreements:
    mean_ag = np.mean(all_agreements, axis=0)
    std_ag = np.std(all_agreements, axis=0)
    ax.plot(qs, mean_ag, color="gray", linestyle="--", linewidth=1.5, label="Mean ± SD")
    ax.fill_between(qs, mean_ag - std_ag, mean_ag + std_ag, color="gray", alpha=0.15)

# Reference lines & formatting
ax.axhline(y=1.0, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
ax.axvline(x=0.9, color="orange", linestyle=":", linewidth=0.5, alpha=0.5, label="Top 10% threshold")
ax.axvspan(0.85, 1.0, alpha=0.08, color="orange")

ax.set_xlabel("Quantile threshold $q$ (select scores $\\geq q$-th percentile)")
ax.set_ylabel("Threshold-aligned agreement")
ax.set_title("Pairwise Tokenization Stability Across Selection Thresholds")
ax.legend(frameon=True, fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("continuous_stability_curves.png", dpi=300, bbox_inches="tight")
plt.show()


pairs_to_plot = select_informative_pairs(configs, scores_list)

# ------------------------------------------------------------
# Plot continuous stability curves
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

# Plot selected pairs
for i, j, label, color in pairs_to_plot:
    qs, ags = compute_stability_curve(scores_list[i], scores_list[j])
    ax.plot(qs, ags, label=label, color=color, linewidth=2)

# Optional: overlay mean ± std across ALL pairs (light background)
all_agreements = []
for i, j in combinations(range(len(scores_list)), 2):
    _, ags = compute_stability_curve(scores_list[i], scores_list[j])
    all_agreements.append(ags)
if all_agreements:
    mean_ag = np.mean(all_agreements, axis=0)
    std_ag = np.std(all_agreements, axis=0)
    ax.plot(qs, mean_ag, color="gray", linestyle="--", linewidth=1.5, label="Mean ± SD")
    ax.fill_between(qs, mean_ag - std_ag, mean_ag + std_ag, 
                    color="gray", alpha=0.15)

# Reference lines
ax.axhline(y=1.0, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
ax.axvline(x=0.9, color="orange", linestyle=":", linewidth=0.5, alpha=0.5, 
           label="Top 10% threshold")

# Labels and formatting
ax.set_xlabel("Quantile threshold $q$ (select scores $\\geq q$-th percentile)")
ax.set_ylabel("Threshold-aligned agreement")
ax.set_title("Pairwise Tokenization Stability Across Selection Thresholds")
ax.legend(frameon=True, fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Highlight critical region for systematic reviews
ax.axvspan(0.85, 1.0, alpha=0.08, color="orange", label="High-relevance region")

plt.tight_layout()
plt.savefig("continuous_stability_curves.png", dpi=300, bbox_inches="tight")
plt.show()
