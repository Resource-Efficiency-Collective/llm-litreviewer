import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re
from scipy.stats import rankdata

plt.style.use(["science", "notebook", "grid"])


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_scores(df, col):
    return df[col].values


def matched_threshold(scores_A, scores_B, tau_A):
    frac = np.mean(scores_A <= tau_A)
    return np.quantile(scores_B, frac)


def split_agreement(A, B):
    return np.mean(A == B)


def pairwise_score(scores_A, scores_B, taus=50):
    """
    Average best-aligned split agreement across thresholds.
    """
    taus_A = np.linspace(np.min(scores_A), np.max(scores_A), taus)

    scores = []

    for tau in taus_A:
        tau_B = matched_threshold(scores_A, scores_B, tau)

        A_split = scores_A >= tau
        B_split = scores_B >= tau_B

        scores.append(split_agreement(A_split, B_split))

    return np.mean(scores)


# ------------------------------------------------------------
# Load all configs
# ------------------------------------------------------------

base_path = Path("./output/WOS/lithium")
files = list(base_path.glob("R_*_IR_*_OR_*_T_0_MODEL_llama3.2_*.csv"))

configs = []
scores_list = []

for file in files:

    match = re.search(
        r"R_(.*?)_IR_(.*?)_OR_(.*?)_T_0_MODEL_llama3\.2_(\d+)\.csv",
        file.name,
    )

    if not match:
        continue

    r, ir, order = match.group(1), match.group(2), match.group(3)

    df = pd.read_csv(file)

    if r not in df.columns:
        continue

    scores = get_scores(df, r)

    configs.append(f"R{r}-IR{ir}-{order}")
    scores_list.append(scores)


# ------------------------------------------------------------
# Pairwise matrix
# ------------------------------------------------------------

n = len(scores_list)
matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        matrix[i, j] = pairwise_score(scores_list[i], scores_list[j])


# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(configs, rotation=90)
ax.set_yticklabels(configs)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Threshold-aligned split agreement")

ax.set_title("Pairwise Tokenization Stability Matrix")

plt.tight_layout()
plt.show()
