import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")
# -------------------------------------------------------------------- #
# Setup
topics = ["financial markets", "polar bears", "rabbits", "cows"]
token_conventions = [
    ["1", "0", "relevant_first", "0", "llama"],
    # ["1", "0", "irrelevant_first", "0", "llama"],
    # ["0", "1", "irrelevant_first", "0", "llama"],
    # ["0", "1", "relevant_first", "0", "llama"],
    # ["5", "3", "relevant_first", "0", "llama"],
    # ["3", "5", "relevant_first", "0", "llama"],
    # ["6", "3", "relevant_first", "0", "llama"],
    # ["3", "6", "relevant_first", "0", "llama"],
    # ["1", "0", "relevant_first", "0", "qwen"],
    # ["1", "0", "irrelevant_first", "0", "qwen"],
    # ["0", "1", "irrelevant_first", "0", "qwen"],
    # ["0", "1", "relevant_first", "0", "qwen"],
    # ["5", "3", "relevant_first", "0", "qwen"],
    # ["3", "5", "relevant_first", "0", "qwen"],
    # ["6", "3", "relevant_first", "0", "qwen"],
    # ["3", "6", "relevant_first", "0", "qwen"],
    # ["9", "2", "relevant_first", "0", "qwen"],
    # ["2", "9", "relevant_first", "0", "qwen"],
    # ["2", "3", "relevant_first", "0", "qwen"],
    # ["3", "2", "relevant_first", "0", "qwen"],
    # ["9", "2", "relevant_first", "0", "llama"],
    # ["2", "9", "relevant_first", "0", "llama"],
    # ["2", "3", "relevant_first", "0", "llama"],
    # ["3", "2", "relevant_first", "0", "llama"],
]

base_path = "./output"
os.makedirs(base_path, exist_ok=True)

# -------------------------------------------------------------------- #
# Load data for all token conventions
data_conventions = {}
for relevant_token, irrelevant_token, order, temp, model in token_conventions:
    key = f"R_{relevant_token}_IR_{irrelevant_token}_OR_{order}_T_{temp}_MODEL_{model}"
    data_conventions[key] = {}
    for topic in topics:
        path = os.path.join(
            base_path,
            f"{topic}_R_{relevant_token}_IR_{irrelevant_token}_OR_{order}_T_{temp}_MODEL_{model}.csv",
        )
        try:
            data_conventions[key][topic] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue

# -------------------------------------------------------------------- #
# Threshold sweep and plotting
thresholds = np.arange(0, 1.01, 0.01)  # 0 to 1 in steps of 0.01
summary_rows = []  # Store max F1 info for all conventions

for convention_key, topic_data in data_conventions.items():
    relevant_token = convention_key.split("_")[1]
    irrelevant_token = convention_key.split("_")[3]
    order = convention_key.split("_")[5]
    temp = convention_key.split("_")[8]
    model = convention_key.split("_")[10]

    results = []
    for source in topics:
        # source = "cows"
        if source not in topic_data:
            print(f"Skipping {source} for {convention_key} (no data)")
            continue

        combined_df = topic_data[source]
        source_df_string = source.replace(" ", "")
        scores = combined_df[relevant_token]  # Based on relevance tokens
        scores_2 = combined_df[irrelevant_token]  # Based on relevance tokens
        sum_probabilities = scores + scores_2
        avg_non_relevant = (1 - (sum_probabilities)).mean()
        max_non_relevant = (1 - (sum_probabilities)).max()
        print(f"Max Non Relevant: {max_non_relevant}")

        for threshold in thresholds:
            predicted_relevance = scores > threshold

            TP = (
                (combined_df["source"] == source_df_string) & predicted_relevance
            ).sum()
            FP = (
                (combined_df["source"] != source_df_string) & predicted_relevance
            ).sum()
            FN = (
                (combined_df["source"] == source_df_string) & (~predicted_relevance)
            ).sum()
            TN = (
                (combined_df["source"] != source_df_string) & (~predicted_relevance)
            ).sum()

            results.append(
                {
                    "Source": source,
                    "Threshold": threshold,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TN": TN,
                    "Precision": TP / (TP + FP) if (TP + FP) > 0 else 0,
                    "Recall": TP / (TP + FN) if (TP + FN) > 0 else 0,
                    "F1": 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0,
                }
            )

    # ---------------------------------------------------------------- #
    # Save per-convention threshold sweep
    results_df = pd.DataFrame(results)
    output_file = os.path.join(
        base_path,
        f"threshold_sweep_R_{relevant_token}_IR_{irrelevant_token}_OR_{order}_T_{temp}_MODEL_{model}.csv",
    )
    results_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # ---------------------------------------------------------------- #
    # Compute max F1 and corresponding threshold per topic
    for source in results_df["Source"].unique():
        df_source = results_df[results_df["Source"] == source]
        if df_source.empty:
            continue
        max_row = df_source.loc[df_source["F1"].idxmax()]
        summary_rows.append(
            {
                "Convention": convention_key,
                "Source": source,
                "Max_F1": max_row["F1"],
                "Best_Threshold": max_row["Threshold"],
            }
        )

    # ---------------------------------------------------------------- #
    # Multi-metric plot (Precision, Recall, F1)
    sources = results_df["Source"].unique()
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    metrics = ["Precision", "Recall", "F1"]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, source in enumerate(sources):
            df_source = results_df[results_df["Source"] == source]
            ax.plot(
                df_source["Threshold"],
                df_source[metric],
                label=source.capitalize(),
                color=colors[j % len(colors)],
            )
        ax.set_title(metric)
        ax.set_xlabel("Threshold")
        ax.grid(True)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("Score")
        if i == 1:
            ax.legend(loc="lower left")

    plt.suptitle(f"Convention: {convention_key}", y=0.95)
    plt.tight_layout()
    fig_name = os.path.join(base_path, f"metrics_vs_threshold_{convention_key}")
    plt.savefig(f"{fig_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots for {convention_key}")

# -------------------------------------------------------------------- #
# Combine and save summary of max F1 per topic & convention
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("Max_F1", ascending=False)
rounded_df = summary_df.round(2)
summary_file = os.path.join(base_path, "max_f1_summary_2.csv")
rounded_df.to_csv(summary_file, index=False)
print(f"Saved summary: {summary_file}")

summary_df = summary_df.sort_values("Max_F1")
summary_df.to_csv("summary_df_f1.csv")
