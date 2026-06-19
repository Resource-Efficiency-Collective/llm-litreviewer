import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re
from itertools import combinations


All_Info = pd.read_excel('../../data/WOS/Lithium/lithium_combined.xlsx')
All_Info.value_counts('Label_Human')

# output_directory = './output/WOS/Lithium/R_0_IR_1_OR_relevant_first_T_0_MODEL_llama3.2_5.csv'

base_path = Path("./output/WOS/lithium")
files = list(base_path.glob("R_*_IR_*_OR_*_T_0_MODEL_llama3.2_*.csv"))

configs, scores_list = [], []
dfs = {}
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
    dfs[label]=df


merge_key = 'UT (Unique WOS ID)'

# Merge All_Info with each df in dfs on merge_key for additional_information.

# On the merged dfs drop rows where "Label_Human" = nan



# Something similar to this, but this time there is only 1 topic (Lithium)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re
import os

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------

plt.style.use(["science", "notebook", "grid"])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7,
})

# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------

All_Info = pd.read_excel(
    "../../data/WOS/Lithium/lithium_combined.xlsx"
)

All_Info = All_Info.dropna(subset=["Label_Human"])
All_Info['Label_Human']
All_Info["Label_Human"] = All_Info["Label_Human"].replace(
    "BORDERLINE",True
)
# Consider borderline papers True


merge_key = "UT (Unique WOS ID)"

# ------------------------------------------------------------
# Load model outputs
# ------------------------------------------------------------

base_path = Path("./output/WOS/lithium")

files = list(
    base_path.glob(
        "R_*_IR_*_OR_*_T_0_MODEL_llama3.2_*.csv"
    )
)

dfs = {}
configs = []

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
    df.drop('Label_Human', axis=1, inplace=True)

    # --------------------------------------------------------
    # Merge with ground truth
    # --------------------------------------------------------

    df = df.merge(
        All_Info,
        on=merge_key,
        how="left",
    )

    df = df.dropna(subset=["Label_Human"])

    dfs[label] = df

# ------------------------------------------------------------
# Threshold sweep setup
# ------------------------------------------------------------

thresholds = np.arange(0, 1.01, 0.01)

results_by_config = {}

summary_rows = []

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

for label, df in dfs.items():

    # 'R1-IR0-RF'
    match = re.search(
        r"R(.*?)-IR(.*?)-(RF|IF)",
        label,
    )

    r = match.group(1)

    scores = df[r].values
    truth = df["Label_Human"].values.astype(int)

    results = []

    for tau in thresholds:

        pred = scores >= tau

        TP = np.sum((pred == 1) & (truth == 1))
        FP = np.sum((pred == 1) & (truth == 0))
        FN = np.sum((pred == 0) & (truth == 1))
        TN = np.sum((pred == 0) & (truth == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = (
            2 * TP / (2 * TP + FP + FN)
            if (2 * TP + FP + FN) > 0
            else 0
        )

        results.append({
            "Threshold": tau,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

    results_df = pd.DataFrame(results)
    results_by_config[label] = results_df

    # --------------------------------------------------------
    # Best threshold (F1)
    # --------------------------------------------------------

    best = results_df.loc[
        results_df["F1"].idxmax()
    ]

    summary_rows.append({
        "Config": label,
        "Best_Threshold": best["Threshold"],
        "Max_F1": best["F1"],
    })

    # --------------------------------------------------------
    # Plot metrics
    # --------------------------------------------------------

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.plot(
        results_df["Threshold"],
        results_df["Precision"],
        label=r"Precision",
    )

    ax.plot(
        results_df["Threshold"],
        results_df["Recall"],
        label=r"Recall",
    )

    ax.plot(
        results_df["Threshold"],
        results_df["F1"],
        label=r"F1",
    )

    ax.set_title(label)
    ax.set_xlabel(r"Threshold $\tau$")
    ax.set_ylabel(r"Score")

    ax.set_ylim(0, 1)

    ax.grid(alpha=0.3)

    ax.legend(loc='lower left')

    plt.tight_layout()

    plt.savefig(
        f"threshold_sweep_lithium_{label}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    "Max_F1",
    ascending=False,
)

summary_df.to_csv(
    "lithium_threshold_summary.csv",
    index=False,
)

print("Saved threshold sweep + summary")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import re

# ------------------------------------------------------------
# Style (journal-safe)
# ------------------------------------------------------------

plt.style.use(["science", "notebook", "grid"])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
})

# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------

All_Info = pd.read_excel(
    "../../data/WOS/Lithium/lithium_combined.xlsx"
)

All_Info = All_Info.dropna(subset=["Label_Human"])

All_Info["Label_Human"] = (
    All_Info["Label_Human"]
    .replace("BORDERLINE", True)
)

merge_key = "UT (Unique WOS ID)"
All_Info = All_Info.drop_duplicates(subset=merge_key)


# ------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------

base_path = Path("./output/WOS/lithium")

files = list(
    base_path.glob(
        "R_*_IR_*_OR_*_T_0_MODEL_llama3.2_*.csv"
    )
)

dfs = {}
configs = []

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

    df = df.drop_duplicates(subset=merge_key)
    df.drop('Label_Human', axis=1, inplace=True)



    # --------------------------------------------------------
    # Merge with ground truth
    # --------------------------------------------------------

    df = df.merge(
        All_Info,
        on=merge_key,
        how="left",
    )

    df = df.dropna(subset=["Label_Human"])

    dfs[label] = df

# ------------------------------------------------------------
# Threshold sweep
# ------------------------------------------------------------

# ------------------------------------------------------------
# Threshold sweep setup
# ------------------------------------------------------------

# thresholds = np.arange(0, 1.01, 0.01)
#
# results_by_config = {}
#
# summary_rows = []
#
# # ------------------------------------------------------------
# # Main loop
# # ------------------------------------------------------------
#
# for label, df in dfs.items():
#
#     # 'R1-IR0-RF'
#     match = re.search(
#         r"R(.*?)-IR(.*?)-(RF|IF)",
#         label,
#     )
#
#     r = match.group(1)
#
#     scores = df[r].values
#     truth = df["Label_Human"].values.astype(int)
#
#     results = []
#
#     for tau in thresholds:
#
#         pred = scores >= tau
#
#         TP = np.sum((pred == 1) & (truth == 1))
#         FP = np.sum((pred == 1) & (truth == 0))
#         FN = np.sum((pred == 0) & (truth == 1))
#         TN = np.sum((pred == 0) & (truth == 0))
#
#         precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#         recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#
#         f1 = (
#             2 * TP / (2 * TP + FP + FN)
#             if (2 * TP + FP + FN) > 0
#             else 0
#         )
#
#         results.append({
#             "Threshold": tau,
#             "TP": TP,
#             "FP": FP,
#             "FN": FN,
#             "TN": TN,
#             "Precision": precision,
#             "Recall": recall,
#             "F1": f1,
#         })
#
#     results_df = pd.DataFrame(results)
#     results_by_config[label] = results_df
#
#     # --------------------------------------------------------
#     # Best threshold (F1)
#     # --------------------------------------------------------
#
#     best = results_df.loc[
#         results_df["F1"].idxmax()
#     ]
#
#     summary_rows.append({
#         "Config": label,
#         "Best_Threshold": best["Threshold"],
#         "Max_F1": best["F1"],
#     })
thresholds = np.arange(0, 1.01, 0.01)

results_by_config = {}
summary_rows = []

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

for label, df in dfs.items():

    # 'R1-IR0-RF'
    match = re.search(
        r"R(.*?)-IR(.*?)-(RF|IF)",
        label,
    )

    r = match.group(1)

    scores = df[r].values
    truth = df["Label_Human"].values.astype(int)

    results = []


    for tau in thresholds:

        pred = scores >= tau

        TP = np.sum((pred == 1) & (truth == 1))
        FP = np.sum((pred == 1) & (truth == 0))
        FN = np.sum((pred == 0) & (truth == 1))
        TN = np.sum((pred == 0) & (truth == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = (
            2 * TP / (2 * TP + FP + FN)
            if (2 * TP + FP + FN) > 0
            else 0
        )

        results.append({
            "Threshold": tau,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

    results_df = pd.DataFrame(results)
    results_by_config[label] = results_df

    # --------------------------------------------------------
    # Best threshold (F1 with tie-break on Recall)
    # --------------------------------------------------------

    best = (
        results_df[results_df["F1"] == results_df["F1"].max()]
        .sort_values("Recall", ascending=False)
        .iloc[0]
    )

    summary_rows.append({
        "Config": label,
        "Best_Threshold": best["Threshold"],
        "Max_F1": best["F1"],
        "TP": best["TP"],
        "FP": best["FP"],
        "FN": best["FN"],
        "TN": best["TN"],
        "Precision_at_best": best["Precision"],
        "Recall_at_best": best["Recall"],
    })


# ------------------------------------------------------------
# Select configs → summary table
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("Max_F1", ascending=False)


# ------------------------------------------------------------
# Select 12 configs → split into 2 figures
# ------------------------------------------------------------


configs_sorted = summary_df["Config"].tolist()

fig_sets = [
    configs_sorted[:6],
    configs_sorted[6:12],
]

# ------------------------------------------------------------
# Plot function (shared)
# ------------------------------------------------------------

def plot_figure(configs, fname):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7, 4.5),
        sharex=True,
        sharey=True
    )

    for i, config in enumerate(configs):

        ax = axes.flat[i]
        df_res = results_by_config[config]

        # best_tau = summary_df[
        #     summary_df["Config"] == config
        # ]["Best_Threshold"].values[0]
        best_f1 = summary_df[
            summary_df["Config"] == config
        ]["Max_F1"].values[0]

        best_tau = summary_df[
            summary_df["Config"] == config
        ]["Best_Threshold"].values[0]

        # curves
        opt_label = rf"Opt: $F1={best_f1:.2f},\ \tau={best_tau:.2f}$"
        ax.plot(df_res["Threshold"], df_res["Precision"], label=r"Precision")
        ax.plot(df_res["Threshold"], df_res["Recall"], label=r"Recall")
        ax.plot(df_res["Threshold"], df_res["F1"], label=r"F1")

        # optimal line
        ax.axvline(best_tau, linestyle="--", color="black", linewidth=0.8)

        opt_label = rf"Opt: $F1={best_f1:.2f},\ \tau={best_tau:.2f}$"
        opt_handle, = ax.plot([], [], linestyle="--", color="black", label=opt_label)

        # legend logic
        if i == 0:
            ax.legend(loc="lower left")
        else:
            ax.legend(
                handles=[opt_handle],
                labels=[opt_label],
                loc="lower left"
            )

        ax.set_title(config,fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        if i % 3 == 0:
            ax.set_ylabel(r"Score")

        if i >= 3:
            ax.set_xlabel(r"Token Threshold")


    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------
# Generate both figures
# ------------------------------------------------------------

plot_figure(fig_sets[0], "lithium_threshold_fig_1.png")
plot_figure(fig_sets[1], "lithium_threshold_fig_2.png")

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

summary_df.to_csv(
    "lithium_threshold_summary.csv",
    index=False,
)

print("Saved 2 figures (12 configs total) + summary")
