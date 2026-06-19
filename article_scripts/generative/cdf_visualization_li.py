# \paragraph{Pairwise Tokenization Stability (PTS).}
 # We define Pairwise Tokenization Stability (PTS) to quantify the robustness of abstract screening decisions under different tokenization conventions in an LLM-based literature review filtering pipeline. For each configuration $c \in \mathcal{C}$, the model assigns each abstract $d \in \mathcal{D}$ a relevance score $s_c(d)$, defined as the probability of a designated \emph{relevance token}. Different configurations correspond to alternative tokenization or prompting conventions, which may shift or rescale these scores.
#
# Because score distributions are not directly comparable across configurations, we evaluate stability in terms of threshold-induced filtering decisions rather than raw scores. A threshold $\tau_c$ defines a selected subset
# \[
# \mathcal{S}_c(\tau_c) = \{ d \in \mathcal{D} : s_c(d) \ge \tau_c \}.
# \]
# To enable comparison, thresholds are aligned across configurations using empirical quantile matching, such that a threshold $\tau_a$ in configuration $a$ is mapped to a threshold $\tau_b$ in configuration $b$ selecting the same proportion of abstracts under their respective score distributions.
#
# We compute agreement between the resulting selected sets under aligned thresholds and average this agreement over a range of thresholds. The resulting PTS score, $\mathrm{PTS}(a,b)$, measures the expected consistency of inclusion/exclusion decisions across tokenization conventions under monotone calibration alignment.
#
# A value of $\mathrm{PTS}(a,b) = 1$ indicates that two configurations induce identical filtering behaviour up to a monotonic transformation of relevance scores, implying that tokenization affects only score calibration and not the induced selection of abstracts. Lower values indicate that tokenization alters which abstracts are selected for inclusion in a literature review.
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re
import itertools

plt.style.use(["science", "notebook", "grid"])
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 9,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def compute_cumulative_counts(df, column="1"):
    df_sorted = df.sort_values(column)
    values = df_sorted[column].unique()
    counts = [(df_sorted[column] >= v).sum() for v in values]
    return pd.DataFrame({column: values, "count_ge": counts})


# ----------------------------
# Setup
# ----------------------------

base_path = Path("./output/WOS/lithium")
files = list(base_path.glob("R_*_IR_*_OR_*_T_0_MODEL_llama3.2_*.csv"))

fig, ax = plt.subplots(figsize=(9, 6))

# stable color per (R, IR)
colors = plt.cm.tab20.colors
color_map = {}

def get_color(pair):
    if pair not in color_map:
        color_map[pair] = colors[len(color_map) % len(colors)]
    return color_map[pair]


# ----------------------------
# Plot
# ----------------------------

seen = set()

for file in files:

    match = re.search(
        r"R_(.*?)_IR_(.*?)_OR_(.*?)_T_0_MODEL_llama3\.2_(\d+)\.csv",
        file.name,
    )
    if not match:
        continue

    r, ir, order = match.group(1), match.group(2), match.group(3)

    config = (r, ir, order)
    if config in seen:
        continue
    seen.add(config)

    df = pd.read_csv(file)

    if r not in df.columns:
        continue

    plot_df = compute_cumulative_counts(df, column=r)


    order_short = "RF" if order == "relevant_first" else "IF"

    label = rf"R{r}-IR{ir}-{order_short}"

    ax.plot(
        plot_df[r],
        plot_df["count_ge"],
        color=get_color((r, ir)),
        linestyle="-" if order == "relevant_first" else "--",
        linewidth=1.8,
        alpha=0.85,
        label=label,
    )


# ----------------------------
# Formatting
# ----------------------------

ax.set_xlabel("Threshold")
ax.set_ylabel(r"Abstracts $\geq$ Threshold")

ax.legend()
ax.set_ylim(0)
# ax.legend(
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5),
#     frameon=True,
# )

# plt.tight_layout(rect=[0, 0, 0.78, 1])
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re

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
    taus_A = np.linspace(np.min(scores_A), np.max(scores_A), taus)

    scores = []

    for tau in taus_A:
        tau_B = matched_threshold(scores_A, scores_B, tau)

        A_split = scores_A >= tau
        B_split = scores_B >= tau_B

        scores.append(split_agreement(A_split, B_split))

    return np.mean(scores)
def pairwise_score_symmetric(scores_A, scores_B, n_quantiles=50):
    quantiles = np.linspace(0, 1, n_quantiles)
    scores = []
    
    for q in quantiles:
        tau_A = np.quantile(scores_A, q)
        tau_B = np.quantile(scores_B, q)  # Same quantile in both
        
        A_split = scores_A >= tau_A
        B_split = scores_B >= tau_B
        scores.append(np.mean(A_split == B_split))
    
    return np.mean(scores)


# ------------------------------------------------------------
# Load data
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
# Pairwise matrix
# ------------------------------------------------------------

n = len(scores_list)
matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        # matrix[i, j] = pairwise_score(scores_list[i], scores_list[j])

        matrix[i, j] = pairwise_score_symmetric(scores_list[i], scores_list[j])


# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
for i in range(n):
    for j in range(n):
        ax.text(
            j,
            i,
            f"{matrix[i, j]:.1f}",
            ha="center",
            va="center",
            color="white" if matrix[i, j] < 0.6 else "black",
            fontsize=8,
        )

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.tick_params(axis='both', which='both', length=0)
ax.set_xticklabels(configs, rotation=90)
ax.set_yticklabels(configs)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Threshold-aligned split agreement")

ax.set_title("Pairwise Tokenization Stability Matrix")
ax.grid(False)


plt.tight_layout()
figname='pairwise_tokenization_matrix_sym'
plt.savefig(f'{figname}.png',dpi=300, bbox_inches='tight')
plt.show()
