# paragraph{Pairwise Tokenization Stability (PTS).}
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
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 8,
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

def parse_config(label):
    """Extract (r, ir, order) from config label like 'R0.3-IR0.5-RF'."""
    parts = label.split('-')
    r = parts[0].replace('R', '')
    ir = parts[1].replace('IR', '')
    order = parts[2]  # Already 'RF' or 'IF'
    return r, ir, order

# ------------------------------------------------------------
# Load all configurations
# ------------------------------------------------------------

base_path = Path("./output/WOS/lithium")
files = list(base_path.glob("R_*_IR_*_OR_*_T_0_MODEL_llama3.2_*.csv"))

configs, scores_list, parsed_configs = [], [], []

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
    parsed_configs.append((r, ir, order_short))

# ------------------------------------------------------------
# Pair selection functions (fixed + standardized labels)
# ------------------------------------------------------------

def find_order_pairs(configs, parsed_configs, scores_list, max_pairs=6):
    """Pairs differing ONLY in ordering (RF vs IF), identical R/IR values."""
    pairs = []
    n = len(configs)
    seen_r_ir = set()  # Track which (r, ir) combos we've already plotted
    
    for i, j in combinations(range(n), 2):
        r1, ir1, ord1 = parsed_configs[i]
        r2, ir2, ord2 = parsed_configs[j]
        
        if r1 == r2 and ir1 == ir2 and ord1 != ord2:
            key = (r1, ir1)
            if key not in seen_r_ir:
                label = f"Order: {ord1} $\\leftrightarrow$ {ord2} \\quad (R={r1}, IR={ir1})"
                pairs.append((i, j, label))
                seen_r_ir.add(key)
                if len(pairs) >= max_pairs:
                    break
    return pairs

def find_swap_pairs(configs, parsed_configs, scores_list, max_pairs=6):
    """Pairs where R and IR values are swapped, same order."""
    pairs = []
    n = len(configs)
    seen = set()
    
    for i, j in combinations(range(n), 2):
        r1, ir1, ord1 = parsed_configs[i]
        r2, ir2, ord2 = parsed_configs[j]
        
        if r1 == ir2 and ir1 == r2 and ord1 == ord2 and r1 != ir1:
            key = (ord1, tuple(sorted([r1, ir1])))
            if key not in seen:
                label = f"Swap: R$\\leftrightarrow$IR \\quad (R={r1}/IR={ir1} $\\leftrightarrow$ R={ir1}/IR={r1}, {ord1})"
                pairs.append((i, j, label))
                seen.add(key)
                if len(pairs) >= max_pairs:
                    break
    return pairs

def find_tokenization_pairs(configs, parsed_configs, scores_list, max_pairs=6):
    """Pairs differing in exactly one tokenization parameter (R or IR), same order."""
    pairs = []
    n = len(configs)
    seen = set()
    
    for i, j in combinations(range(n), 2):
        r1, ir1, ord1 = parsed_configs[i]
        r2, ir2, ord2 = parsed_configs[j]
        
        if ord1 == ord2:
            # Differ in exactly one of R or IR
            if r1 == r2 and ir1 != ir2:
                param = "IR"
                key = (ord1, r1, ir1, ir2)
                label = f"Token param: {param}={ir1} $\\leftrightarrow$ {ir2} \\quad (R={r1}, {ord1})"
            elif r1 != r2 and ir1 == ir2:
                param = "R"
                key = (ord1, ir1, r1, r2)
                label = f"Token param: {param}={r1} $\\leftrightarrow$ {r2} \\quad (IR={ir1}, {ord1})"
            else:
                continue  # Differ in both or neither
                
            if key not in seen:
                pairs.append((i, j, label))
                seen.add(key)
                if len(pairs) >= max_pairs:
                    break
    return pairs

# ------------------------------------------------------------
# Collect pairs for each subplot
# ------------------------------------------------------------

order_pairs = find_order_pairs(configs, parsed_configs, scores_list)
swap_pairs = find_swap_pairs(configs, parsed_configs, scores_list)
token_pairs = find_tokenization_pairs(configs, parsed_configs, scores_list)

print(f"Found {len(order_pairs)} order pairs, {len(swap_pairs)} swap pairs, {len(token_pairs)} tokenization pairs")

# ------------------------------------------------------------
# Plotting: 3 standardized subplots
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)

# Standardized styling
subplot_titles = [
    "(a) Effect of Prompt Ordering",
    "(b) Effect of R/IR Parameter Symmetry", 
    "(c) Effect of Tokenization Thresholds"
]
pair_collections = [order_pairs, swap_pairs, token_pairs]
base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]

for ax, pairs, title, base_color in zip(axes, pair_collections, subplot_titles, base_colors):
    
    if not pairs:
        ax.text(0.5, 0.5, "No comparable pairs found", 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=10, style='italic', alpha=0.6)
    
    # Plot each pair
    for idx, (i, j, label) in enumerate(pairs):
        qs, ags = compute_stability_curve(scores_list[i], scores_list[j])
        linestyle = linestyles[idx % len(linestyles)]
        ax.plot(qs, ags, label=label, color=base_color, linewidth=2, linestyle=linestyle)
    
    # Reference lines
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5, label="Perfect agreement")
    ax.axvline(x=0.9, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.6, label="Top 10\% threshold")
    ax.axvspan(0.85, 1.0, alpha=0.06, color="#d62728", label="High-relevance region")
    
    # Standardized labels
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Quantile threshold $q$ (select scores $\\geq q$-th percentile)")
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    
    # Y-label only on leftmost
    if ax == axes[0]:
        ax.set_ylabel("Threshold-aligned agreement")
    
    # Compact legend (only if not too crowded)
    ax.legend(frameon=True, fontsize=7, loc="lower left", handlelength=1.5)

# Shared annotations
fig.text(0.5, 0.015, 
         "Note: Agreement = fraction of abstracts with consistent inclusion/exclusion decisions at aligned thresholds", 
         ha='center', fontsize=9, style='italic', alpha=0.7)

plt.suptitle("Sources of Variation in Tokenization Stability for Abstract Screening", 
             fontsize=15, y=1.03, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig("stability_by_variation_type_standardized.png", dpi=300, bbox_inches="tight")
plt.show()
# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

subplot_titles = [
    r"(a) Prompt Ordering",
    r"(b) R/IR Symmetry",
    r"(c) Token Parameters",
]

pair_collections = [
    order_pairs,
    swap_pairs,
    token_pairs,
]

colors = plt.cm.tab20.colors

linestyles = [
    '-',
    '--',
    '-.',
    ':',
    (0, (3, 1, 1, 1)),
    (0, (5, 5)),
]

markers = [
    'o',
    's',
    '^',
    'D',
    'v',
    'P',
    'X',
    '*',
]

for subplot_idx, (ax, pairs, title) in enumerate(
    zip(axes, pair_collections, subplot_titles)
):

    for idx, (i, j, label) in enumerate(pairs):

        qs, ags = compute_stability_curve(
            scores_list[i],
            scores_list[j],
        )

        r1, ir1, ord1 = parsed_configs[i]
        r2, ir2, ord2 = parsed_configs[j]

        # ----------------------------------------------------
        # MINIMAL LEGEND LABELS
        # ----------------------------------------------------

        if subplot_idx == 0:
            # ordering comparison
            short_label = rf"$R={r1},\ IR={ir1}$"

        elif subplot_idx == 1:
            # swap comparison
            short_label = rf"${r1}\leftrightarrow{ir1}$ ({ord1})"

        else:
            # token parameter comparison

            if r1 == r2:
                short_label = rf"$IR:{ir1}\leftrightarrow{ir2}$"
            else:
                short_label = rf"$R:{r1}\leftrightarrow{r2}$"

        ax.plot(
            qs,
            ags,
            label=short_label,
            color=colors[idx % len(colors)],
            linewidth=2,
            linestyle=linestyles[idx % len(linestyles)],
            marker=markers[idx % len(markers)],
            markersize=3,
            markevery=12,
            alpha=0.9,
        )

    # --------------------------------------------------------
    # Reference region
    # --------------------------------------------------------

    ax.axhline(
        y=1.0,
        color="gray",
        linestyle=":",
        linewidth=0.7,
        alpha=0.5,
    )

    ax.axvline(
        x=0.9,
        color="black",
        linestyle=":",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.axvspan(
        0.9,
        1.0,
        alpha=0.05,
        color="gray",
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(title, fontsize=12)

    ax.set_xlabel(
        r"Quantile threshold $q$"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.grid(
        alpha=0.3,
        linestyle='--',
        linewidth=0.3,
    )

    if subplot_idx == 0:
        ax.set_ylabel(
            r"Threshold-aligned agreement"
        )

    # --------------------------------------------------------
    # Internal legend
    # --------------------------------------------------------

    ax.legend(
        fontsize=7,
        frameon=True,
        loc="lower left",
        ncol=1,
        handlelength=1.4,
        borderpad=0.3,
        labelspacing=0.25,
    )

# ------------------------------------------------------------
# Global title
# ------------------------------------------------------------

plt.suptitle(
    r"Sources of Variation in Tokenization Stability",
    fontsize=15,
    y=1.02,
)

plt.tight_layout()

plt.savefig(
    "stability_by_variation_type_standardized.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re
from itertools import combinations

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------

plt.style.use(["science", "notebook", "grid"])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def threshold_agreement_at_quantile(A, B, q):
    tau_A = np.quantile(A, q)
    tau_B = np.quantile(B, q)

    return np.mean(
        (A >= tau_A) == (B >= tau_B)
    )


def compute_stability_curve(A, B, n_points=100):
    qs = np.linspace(0, 1, n_points)

    agreements = [
        threshold_agreement_at_quantile(A, B, q)
        for q in qs
    ]

    return qs, np.array(agreements)


# ------------------------------------------------------------
# Load configurations
# ------------------------------------------------------------

base_path = Path("./output/WOS/lithium")

files = list(
    base_path.glob(
        "R_*_IR_*_OR_*_T_0_MODEL_llama3.2_*.csv"
    )
)

configs = []
scores_list = []
parsed_configs = []

for file in files:

    match = re.search(
        r"R_(.*?)_IR_(.*?)_OR_(.*?)_T_0_MODEL_llama3\.2_(\d+)\.csv",
        file.name,
    )

    if not match:
        continue

    r, ir, order = (
        match.group(1),
        match.group(2),
        match.group(3),
    )

    order_short = (
        "RF"
        if order == "relevant_first"
        else "IF"
    )

    label = rf"R{r}-IR{ir}-{order_short}"

    df = pd.read_csv(file)

    if r not in df.columns:
        continue

    configs.append(label)
    scores_list.append(df[r].values)

    parsed_configs.append(
        (r, ir, order_short)
    )

# ------------------------------------------------------------
# Pair selection
# ------------------------------------------------------------

def find_order_pairs(max_pairs=6):

    pairs = []
    seen = set()

    for i, j in combinations(
        range(len(configs)),
        2,
    ):

        r1, ir1, ord1 = parsed_configs[i]
        r2, ir2, ord2 = parsed_configs[j]

        if (
            r1 == r2
            and ir1 == ir2
            and ord1 != ord2
        ):

            key = (r1, ir1)

            if key not in seen:

                pairs.append((i, j))
                seen.add(key)

                if len(pairs) >= max_pairs:
                    break

    return pairs


def find_swap_pairs(max_pairs=6):

    pairs = []
    seen = set()

    for i, j in combinations(
        range(len(configs)),
        2,
    ):

        r1, ir1, ord1 = parsed_configs[i]
        r2, ir2, ord2 = parsed_configs[j]

        if (
            r1 == ir2
            and ir1 == r2
            and ord1 == ord2
            and r1 != ir1
        ):

            key = (
                ord1,
                tuple(sorted([r1, ir1])),
            )

            if key not in seen:

                pairs.append((i, j))
                seen.add(key)

                if len(pairs) >= max_pairs:
                    break

    return pairs


def find_token_pairs(max_pairs=6):

    pairs = []
    seen = set()

    for i, j in combinations(
        range(len(configs)),
        2,
    ):

        r1, ir1, ord1 = parsed_configs[i]
        r2, ir2, ord2 = parsed_configs[j]

        if ord1 != ord2:
            continue

        if r1 == r2 and ir1 != ir2:

            key = (
                "IR",
                ord1,
                r1,
                tuple(sorted([ir1, ir2])),
            )

        elif r1 != r2 and ir1 == ir2:

            key = (
                "R",
                ord1,
                ir1,
                tuple(sorted([r1, r2])),
            )

        else:
            continue

        if key not in seen:

            pairs.append((i, j))
            seen.add(key)

            if len(pairs) >= max_pairs:
                break

    return pairs


order_pairs = find_order_pairs()
swap_pairs = find_swap_pairs()
token_pairs = find_token_pairs()

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(5, 5),
    sharex=True,
    sharey=True,
)

subplot_titles = [
    r"(a) Prompt Ordering",
    r"(b) R/IR Symmetry",
    r"(c) Token Parameters",
]

pair_sets = [
    order_pairs,
    swap_pairs,
    token_pairs,
]

colors = plt.cm.tab20.colors

linestyles = [
    '-',
    '--',
    '-.',
    ':',
    (0, (3, 1, 1, 1)),
    (0, (5, 5)),
]

for subplot_idx, (
    ax,
    pairs,
    title,
) in enumerate(
    zip(
        axes,
        pair_sets,
        subplot_titles,
    )
):

    for idx, (i, j) in enumerate(pairs):

        qs, ags = compute_stability_curve(
            scores_list[i],
            scores_list[j],
        )

        r1, ir1, ord1 = parsed_configs[i]
        r2, ir2, ord2 = parsed_configs[j]

        # ----------------------------------------------------
        # Compact labels
        # ----------------------------------------------------

        if subplot_idx == 0:

            label = rf"$R={r1},\ IR={ir1}$"

        elif subplot_idx == 1:

            label = (
                rf"${r1}"
                rf"\leftrightarrow"
                rf"{ir1}$"
                rf" ({ord1})"
            )

        else:

            if r1 == r2:

                label = (
                    rf"$IR:"
                    rf"{ir1}"
                    rf"\leftrightarrow"
                    rf"{ir2}$"
                )

            else:

                label = (
                    rf"$R:"
                    rf"{r1}"
                    rf"\leftrightarrow"
                    rf"{r2}$"
                )

        ax.plot(
            qs,
            ags,
            label=label,
            color=colors[idx % len(colors)],
            linewidth=1.5,
            linestyle=linestyles[
                idx % len(linestyles)
            ],
            alpha=0.9,
        )

    # --------------------------------------------------------
    # High-relevance threshold marker
    # --------------------------------------------------------

    ax.axvline(
        x=0.9,
        color="black",
        linestyle=":",
        linewidth=0.8,
        alpha=0.7,
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=11,
        pad=6,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.grid(
        alpha=0.25,
        linestyle='--',
        linewidth=0.3,
    )

    if subplot_idx == 1:

        ax.set_ylabel(
            r"Threshold-aligned agreement"
        )

    ax.legend(
        fontsize=8,
        frameon=True,
        loc="lower left",
        ncol=2,
        handlelength=1.2,
        borderpad=0.25,
        labelspacing=0.2,
    )

axes[-1].set_xlabel(
    r"Quantile threshold $q$"
)

plt.tight_layout()

plt.savefig(
    "stability_by_variation_type_vertical.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
