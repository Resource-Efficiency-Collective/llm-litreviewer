import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re

plt.style.use(["science", "notebook", "grid"])
# Answers the question:
# "If I use tokenization scheme A vs. B and select the top 10% of abstracts, how often do I get the same papers?"

# ------------------------------------------------------------
# Load scores
# ------------------------------------------------------------

def get_scores(df, col):
    return df[col].values




def threshold_agreement_at_quantile(A, B, q):
    """
    Agreement on selection decisions when both configs select the top (1-q) proportion.
    q=0.9 means "select top 10%" (scores >= 90th percentile).
    """
    tau_A = np.quantile(A, q)
    tau_B = np.quantile(B, q)
    
    A_sel = A >= tau_A
    B_sel = B >= tau_B
    
    return np.mean(A_sel == B_sel)




# ------------------------------------------------------------
# Load all configurations
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

    order_short = "RF" if order == "relevant_first" else "IF"
    label = rf"R{r}-IR{ir}-{order_short}"

    df = pd.read_csv(file)

    if r not in df.columns:
        continue

    configs.append(label)
    scores_list.append(get_scores(df, r))


# ------------------------------------------------------------
# Define quantile regions
# ------------------------------------------------------------

regions = {
    "Bottom 10%": (0.0, 0.1),
    "Middle 80%": (0.1, 0.9),
    "Top 10%": (0.9, 1.0),
}


# ------------------------------------------------------------
# Plot stacked heatmaps
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

vmin, vmax = 0, 1

for ax, (title, qs) in zip(axes, regions.items()):

    n = len(scores_list)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = threshold_agreement_at_quantile(
                scores_list[i], scores_list[j], qs
            )
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")

    ax.set_title(title)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xticklabels(configs, rotation=90)
    ax.set_yticklabels(configs)

    ax.tick_params(axis="both", which="both", length=0)

    # annotate values
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < 0.6 else "black",
                fontsize=7,
            )

    ax.grid(False)

# shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
cbar.set_label("Quantile-region agreement")

plt.suptitle("Tokenization Stability Across Score Regions", fontsize=16)

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re

plt.style.use(["science", "notebook", "grid"])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_scores(df, col):
    return df[col].values

def threshold_agreement_at_quantile(A, B, q):
    """
    Agreement on selection decisions when both configs select the top (1-q) proportion.
    q=0.9 means "select top 10%" (scores >= 90th percentile).
    """
    tau_A = np.quantile(A, q)
    tau_B = np.quantile(B, q)
    
    A_sel = A >= tau_A
    B_sel = B >= tau_B
    
    return np.mean(A_sel == B_sel)

# ------------------------------------------------------------
# Load all configurations
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
    order_short = "RF" if order == "relevant_first" else "IF"
    label = rf"R{r}-IR{ir}-{order_short}"

    df = pd.read_csv(file)
    if r not in df.columns:
        continue

    configs.append(label)
    scores_list.append(get_scores(df, r))

# ------------------------------------------------------------
# Define single quantile thresholds (not ranges)
# ------------------------------------------------------------
# q=0.90 -> select top 10%
# q=0.50 -> select top 50%
# q=0.10 -> exclude bottom 10% (select top 90%)
thresholds = {
    "Top 10% Selected": 0.90,
    "Top 50% Selected": 0.50,
    "Bottom 10% Excluded": 0.10,
}

# ------------------------------------------------------------
# Plot heatmaps
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
vmin, vmax = 0, 1

for ax, (title, q) in zip(axes, thresholds.items()):
    n = len(scores_list)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = threshold_agreement_at_quantile(
                scores_list[i], scores_list[j], q
            )

    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(title, fontsize=14)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(configs, rotation=90, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(configs)
    ax.tick_params(axis="both", which="both", length=0)

    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{matrix[i, j]:.1f}",
                ha="center", va="center",
                color="white" if matrix[i, j] < 0.6 else "black",
                fontsize=7,
            )
    ax.grid(False)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
cbar.set_label("Threshold-aligned split agreement", fontsize=12)

plt.suptitle("Tokenization Stability at Key Selection Thresholds", fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Prevents suptitle overlap

plt.savefig('pts_threshold_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()
