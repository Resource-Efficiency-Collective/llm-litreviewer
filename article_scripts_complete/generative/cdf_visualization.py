import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Use SciencePlots style with LaTeX support
plt.style.use(["science", "notebook", "grid"])
plt.rcParams.update(
    {
        "text.usetex": True,  # Enable LaTeX rendering
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def compute_cumulative_counts(df, column="1"):
    """Return DataFrame of unique values and counts >= each value."""
    df_sorted = df.sort_values(column)
    values = df_sorted[column].unique()
    counts = [(df_sorted[column] >= v).sum() for v in values]
    return pd.DataFrame({column: values, "count_ge": counts})


def plot_cumulative_counts_multiple_topics(
    topics, models, base_path="../output/llm_methods_paper", column="1"
):
    """
    Plot cumulative >= counts for multiple topics and models in a 2x2 grid.

    Parameters
    ----------
    topics : list of str
        List of topic names (e.g. ["cows", "rabbits"])
    models : list of str
        List of model names (e.g. ["qwen", "llama"])
    base_path : str
        Folder where CSV files are stored
    column : str
        Column name to use for comparison
    """
    n = len(topics)
    rows, cols = 2, 2  # 2x2 grid
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), sharey=True, sharex=True)
    axes = axes.flatten()

    for ax, topic in zip(axes, topics):
        for model in models:
            df = pd.read_csv(
                f"{base_path}/{topic}_R_1_IR_0_OR_irrelevant_first_T_0_MODEL_{model}.csv"
            )
            plot_df = compute_cumulative_counts(df, column)
            ax.plot(
                plot_df[column],
                plot_df["count_ge"],
                marker="o",
                label=rf"\textbf{{{model.capitalize()}}}",
            )

        # Horizontal reference line
        ax.axhline(25, color="k", linestyle="--", label="True Relevant Abstracts")

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_title(rf"\textbf{{{topic.capitalize()}}}")
        ax.grid(True)
        ax.set_xlabel(r"Relevant Token Threshold")
        ax.set_ylabel(r"Number of Abstracts $\geq$ Threshold")

    # Remove any unused subplots (if topics < 4)
    for j in range(len(topics), len(axes)):
        fig.delaxes(axes[j])

    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(models) + 1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave room for legend
    # plt.show()


# Example usage
topics = ["cows", "rabbits", "polar bears", "financial markets"]
models = ["qwen", "llama"]

plot_cumulative_counts_multiple_topics(topics, models)
filename = "qwen_llama_comparison"
plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{filename}.eps")

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Use SciencePlots style with LaTeX support
plt.style.use(["science", "notebook", "grid"])
plt.rcParams.update(
    {
        "text.usetex": True,  # Enable LaTeX rendering
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def compute_cumulative_counts(df, column="1"):
    """Return DataFrame of unique values and counts >= each value."""
    df_sorted = df.sort_values(column)
    values = df_sorted[column].unique()
    counts = [(df_sorted[column] >= v).sum() for v in values]
    return pd.DataFrame({column: values, "count_ge": counts})


def plot_cumulative_counts_multiple_topics(
    topics, models, base_path="../output/llm_methods_paper", column="1"
):
    """
    Plot cumulative >= counts for multiple topics and models in a 2x2 grid.

    Parameters
    ----------
    topics : list of str
        List of topic names (e.g. ["cows", "rabbits"])
    models : list of str
        List of model names (e.g. ["qwen", "llama"])
    base_path : str
        Folder where CSV files are stored
    column : str
        Column name to use for comparison
    """
    n = len(topics)
    rows, cols = 2, 2  # 2x2 grid
    fig, axes = plt.subplots(rows, cols, figsize=(9, 6.5), sharey=True, sharex=True)
    axes = axes.flatten()

    for i, (ax, topic) in enumerate(zip(axes, topics)):
        for model in models:
            df = pd.read_csv(
                f"{base_path}/{topic}_R_1_IR_0_OR_irrelevant_first_T_0_MODEL_{model}.csv"
            )
            plot_df = compute_cumulative_counts(df, column)
            ax.plot(
                plot_df[column],
                plot_df["count_ge"],
                marker="o",
                label=rf"\textbf{{{model.capitalize()}}}",
            )

        # Horizontal reference line
        ax.axhline(25, color="k", linestyle="--", label="True Relevant Abstracts")

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_title(rf"\textbf{{{topic.capitalize()}}}")
        ax.grid(True)

        # Remove repeated axis labels
        if i % cols != 0:
            ax.set_ylabel("")  # not first column → hide y-label
        else:
            ax.set_ylabel(r"Abstracts $\geq$ Threshold")

        if i < (rows - 1) * cols:
            ax.set_xlabel("")  # not bottom row → hide x-label
        else:
            ax.set_xlabel(r"Relevant Token Threshold")

    # Remove any unused subplots (if topics < 4)
    for j in range(len(topics), len(axes)):
        fig.delaxes(axes[j])

    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(models) + 1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave room for legend


# -------------------------------------------------------------------- #
# Create Qwen / Llama comparison Figure 
topics = ["cows", "rabbits", "polar bears", "financial markets"]
models = ["qwen", "llama"]

plot_cumulative_counts_multiple_topics(topics, models)

filename = "qwen_llama_comparison"
plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{filename}.eps", bbox_inches="tight")
