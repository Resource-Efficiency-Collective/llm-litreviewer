import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def freedman_diaconis_bins(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    return max(1, int((data.max() - data.min()) / bin_width))  # Ensure at least 1 bin


def compute_cdf(data):
    sorted_data = np.sort(data)
    sorted_data = sorted_data[::-1]
    cdf = np.arange(1, len(data) + 1)
    return sorted_data, cdf


class Result_Plotter:
    def __init__(self, file_name=None, data=None):
        if file_name:
            self.data = pd.read_csv(file_name)
        else:
            self.data = data

        self.bins_irrelevant = None
        self.bins_relevant = None
        self.bins_neither = None

    def plot_pdf(
        self,
        prompt,
        temperature,
        axes=None,
        outname="",
        save_plot=False,
        show_plot=False,
        alpha=1,
        use_existing_bins=False,
        column_1_name=None,
        column_2_name=None,
        swap_binary=False,
        label=None,
    ):
        if column_1_name is not None:
            relevant_data = self.data[column_1_name].values
        else:

            relevant_data = self.data[
                f"{prompt}_temp_{temperature}_{1}"
            ].values  # 1 is relevant
        if column_2_name is not None:
            irrelevant_data = self.data[column_2_name]
        else:

            irrelevant_data = self.data[
                f"{prompt}_temp_{temperature}_{0}"
            ].values  # 0 is irrelevant

        if swap_binary is True:
            relevant_data, irrelevant_data = irrelevant_data, relevant_data

        neither_data = 1 - (relevant_data + irrelevant_data)
        if use_existing_bins is False or self.bins_relevant is None:
            bins_relevant = freedman_diaconis_bins(relevant_data)
            bins_irrelevant = freedman_diaconis_bins(irrelevant_data)
            bins_neither = freedman_diaconis_bins(neither_data)

            self.bins_relevant = bins_relevant
            self.bins_irrelevant = bins_irrelevant
            self.bins_neither = bins_neither
        else:
            print("Using previous bins")
            bins_relevant = self.bins_relevant
            bins_irrelevant = self.bins_irrelevant
            bins_neither = self.bins_neither

        new_fig = False
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            new_fig = True  # Indicates the function created the figure
        # if axes is None:
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Share y-axis between ax[0] and ax[1] only
        axes[1].sharey(axes[0])

        # Plot histograms
        axes[0].hist(relevant_data, bins=bins_relevant, edgecolor="black", alpha=alpha)
        axes[0].set_xlabel("Token Probability")
        axes[0].set_ylabel("Number of Occurrences")
        axes[0].set_title("Relevance Probability")

        axes[1].hist(
            irrelevant_data, bins=bins_irrelevant, edgecolor="black", alpha=alpha
        )
        axes[1].set_xlabel("Token Probability")
        axes[1].set_title("Irrelevance Probability")
        if label is None:
            label = f"Prompt: {prompt}, T: {temperature}"

        axes[2].hist(
            neither_data,
            bins=bins_neither,
            edgecolor="black",
            alpha=alpha,
            label=label,
        )
        axes[2].set_xlabel("Token Probability")
        axes[2].set_title("Neither Probability")

        axes[2].legend()
        # Adjust layout
        plt.tight_layout()

        # Save plot if enabled
        if save_plot:
            plt.savefig(
                f"token_probs_comparison_{outname}.png", dpi=300, bbox_inches="tight"
            )

        # Show plot if enabled
        if show_plot and new_fig:
            plt.show()

    # def plot_ccdf(self,prompt,temperature,ax=None,outname='',show_plot=False,save_plot=False):
    #
    #     relevant_data = self.data[f'{prompt}_temp_{temperature}_{1}'].values # 1 is relevant
    #     irrelevant_data = self.data[f'{prompt}_temp_{temperature}_{0}'].values # 0 is irrelevant
    #     # neither_data = 1 - (relevant_data + irrelevant_data)
    #     x_relevant, cdf_relevant = compute_cdf(relevant_data)
    #     x_irrelevant, cdf_irrelevant = compute_cdf(irrelevant_data)
    #
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(6, 5))
    #
    #     # Plot CDFs
    #     ax.plot(x_relevant, cdf_relevant, label=f'Relevance ({prompt}:T{temperature}', linestyle='-', marker='', color='blue')
    #     ax.plot(x_irrelevant, cdf_irrelevant, label=f'Irrelevance ({prompt}:T{temperature}', linestyle='-', marker='', color='red')
    #     # ax.plot(x_irrelevant,cdf_irrelevant+cdf_relevant[::-1],label='Total Abstracts',color='k')
    #     ax.axvline(0.5,color='k')
    #     # Labels and title
    #     ax.set_xlabel('Token Probability (p)')
    #     ax.set_ylabel('Number of Abstracts (>p)')
    #
    #     ax.legend(edgecolor='k')
    #     num_abstracts_rel = np.sum(relevant_data > 0.50)
    #     ax.axhline(num_abstracts_rel,color='blue',linestyle='--')
    #     num_abstracts_irr = np.sum(irrelevant_data > 0.50)
    #     # ax.scatter(0.5,num_abstracts,label='Irrelevant Abstracts')
    #     ax.axhline(num_abstracts_irr,color='r',linestyle = '--')
    #     # Layout and save
    #     plt.tight_layout()
    #     if save_plot is True:
    #         plt.savefig(f'token_cdfs_{outname}.png', dpi=300, bbox_inches='tight')
    #     if show_plot is True:
    #         plt.show()

    def plot_ccdf(
        self,
        prompt,
        temperature,
        ax=None,
        relevant_color="red",
        irrelevant_color="blue",
        outname="",
        show_plot=False,
        save_plot=False,
        plot_relevant=True,
        plot_irrelevant=True,
        column_1_name=None,
        column_2_name=None,
        swap_binary=False,
        relevant_label=None,
        irrelevant_label=None,
    ):
        """Plot the Complementary Cumulative Distribution Function (CCDF) for relevance and irrelevance.

        Ensures that line and reference colors remain consistent.

        Parameters:
            prompt (str): The prompt key used in the dataset.
            temperature (float): The temperature setting.
            ax (matplotlib.axes.Axes, optional): Axes object for plotting. If None, creates new.
            outname (str, optional): Output filename suffix for saving.
            show_plot (bool, optional): Whether to display the plot. Default is False.
            save_plot (bool, optional): Whether to save the plot as an image file. Default is False.
        """
        # Define consistent colors
        colors = {
            "relevant": relevant_color,  # Matplotlib default blue
            "irrelevant": irrelevant_color,  # Matplotlib default red
        }

        if column_1_name is not None:
            relevant_data = self.data[column_1_name].values
        else:

            relevant_data = self.data[
                f"{prompt}_temp_{temperature}_{1}"
            ].values  # 1 is relevant
        if column_2_name is not None:
            irrelevant_data = self.data[column_2_name]
        else:

            irrelevant_data = self.data[
                f"{prompt}_temp_{temperature}_{0}"
            ].values  # 0 is irrelevant

        if swap_binary is True:
            relevant_data, irrelevant_data = irrelevant_data, relevant_data
        # # Get data
        # relevant_data = self.data[
        #     f"{prompt}_temp_{temperature}_1"
        # ].values  # 1 = relevant
        # irrelevant_data = self.data[
        #     f"{prompt}_temp_{temperature}_0"
        # ].values  # 0 = irrelevant

        # Compute CDF
        x_relevant, cdf_relevant = compute_cdf(relevant_data)
        x_irrelevant, cdf_irrelevant = compute_cdf(irrelevant_data)

        # Create figure if needed
        new_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            new_fig = True  # Track if we created the figure

        # Plot CCDFs using consistent colors
        if relevant_label is None:
            relevant_label = f"Relevance ({prompt}:T{temperature})"
        if irrelevant_label is None:
            irrelevant_label = f"Irrelevance ({prompt}:T{temperature})"

        if plot_relevant is True:
            ax.plot(
                x_relevant,
                cdf_relevant,
                label=relevant_label,
                linestyle="-",
                marker="",
                color=colors["relevant"],
            )
        if plot_irrelevant is True:
            ax.plot(
                x_irrelevant,
                cdf_irrelevant,
                label=irrelevant_label,
                linestyle="-",
                marker="",
                color=colors["irrelevant"],
            )

        # Vertical reference line at 0.5
        ax.axvline(0.5, color="k", linestyle="-")

        # Compute number of abstracts above threshold (0.5 probability)
        num_abstracts_rel = np.sum(relevant_data > 0.50)
        num_abstracts_irr = np.sum(irrelevant_data > 0.50)

        # Horizontal reference lines using the same colors
        if plot_relevant is True:
            ax.axhline(num_abstracts_rel, color=colors["relevant"], linestyle="--")
        if plot_irrelevant is True:
            ax.axhline(num_abstracts_irr, color=colors["irrelevant"], linestyle="--")

        # Labels and title
        ax.set_xlabel("Token Probability (p)")
        ax.set_ylabel("Number of Abstracts (>p)")
        ax.legend(edgecolor="k")

        # Layout and save
        plt.tight_layout()
        if save_plot:
            plt.savefig(f"token_cdfs_{outname}.png", dpi=300, bbox_inches="tight")

        if show_plot and new_fig:
            plt.show()

        # Close figure if created here and not showing
        if new_fig and not show_plot:
            plt.close(fig)


def extract_relevant_probabilities(response):
    """
    Extracts probabilities for the relevant item (0 or 1) from log probabilities.

    Args:
        response (dict): API response containing log probability data.

    Returns:
        dict: Dictionary with '0' and '1' probabilities at each token position.
    """
    logprobs_data = response["choices"][0]["logprobs"]["top_logprobs"]
    relevant_probs = {"0": [], "1": []}

    for token_info in logprobs_data:
        prob_0 = np.exp(
            token_info.get("0", float("-inf"))
        )  # Convert logprob to probability
        prob_1 = np.exp(
            token_info.get("1", float("-inf"))
        )  # Convert logprob to probability

        relevant_probs["0"].append(prob_0)
        relevant_probs["1"].append(prob_1)

    return relevant_probs

#
def plot_token_probabilities_with_highlighted_choices(response, outname):
    # Extract log probabilities and actual outcome
    logprobs_data = response["choices"][0]["logprobs"]["top_logprobs"]
    outcome = response["choices"][0][
        "text"
    ]  # Actually selected text based on constraints and logprobs
    outcome_tokens = response["choices"][0]["logprobs"][
        "tokens"
    ]  # List of Selected tokens

    # Initialize matrices and variables
    tokens_matrix = []
    logprobs_matrix = []
    max_choices = max(len(token_info) for token_info in logprobs_data)
    outcome_index = 0
    selected_tokens = []
    selected_mask = np.zeros((max_choices, len(logprobs_data)))  # Mask for highlighting

    # Extract tokens and log probabilities, and identify selected tokens
    for col_idx, token_info in enumerate(logprobs_data):
        tokens = list(token_info.keys())
        logprobs = list(token_info.values())

        # # Find selected token by checking against the outcome text
        # selected_token = None
        # for token in tokens:
        #     if outcome.startswith(token, outcome_index):  # Ensure correct subword alignment
        #         selected_token = token
        #         outcome_index += len(token)  # Move forward in the outcome text
        #         break  # Stop once we find the matching token
        #
        # if selected_token is None:
        #     selected_token = tokens[0]  # Fallback to the first token if no match found

        # selected_tokens.append(selected_token)

        # Pad with None if fewer than max_choices
        while len(tokens) < max_choices:
            tokens.append(None)
            logprobs.append(None)

        tokens_matrix.append(tokens)
        logprobs_matrix.append(logprobs)

        # Mark the position of the selected token in the mask
        for row_idx, token in enumerate(tokens):
            if token == outcome_tokens[col_idx]:
                selected_mask[row_idx, col_idx] = 1  # Highlight this cell

    # Convert matrices to DataFrames
    tokens_df = pd.DataFrame(tokens_matrix).T
    logprobs_df = pd.DataFrame(logprobs_matrix).T

    # Rename DataFrame columns to indicate token positions
    tokens_df.columns = [f"Token {i}" for i in range(tokens_df.shape[1])]
    logprobs_df.columns = [f"Token {i}" for i in range(logprobs_df.shape[1])]

    # Rename rows to indicate choice rank (1st, 2nd, 3rd, etc.)
    tokens_df.index = [
        (
            f"{i+1}st Choice"
            if i == 0
            else (
                f"{i+1}nd Choice"
                if i == 1
                else f"{i+1}rd Choice" if i == 2 else f"{i+1}th Choice"
            )
        )
        for i in range(max_choices)
    ]
    logprobs_df.index = tokens_df.index  # Same index as tokens_df

    # Convert log probabilities to normal probabilities
    prob_df = np.exp(logprobs_df)

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        prob_df,
        annot=tokens_df,
        fmt="",
        cmap="coolwarm",
        linewidths=0.5,
        annot_kws={"rotation": 90},  # Rotate text annotations vertically
    )

    # Highlight selected tokens by overlaying yellow boxes
    edgecolor = "black"  # Use yellow for highlighting
    for row in range(max_choices):
        for col in range(len(logprobs_data)):
            if selected_mask[row, col] == 1:  # If this is the selected token
                ax.add_patch(
                    Rectangle((col, row), 1, 1, fill=False, edgecolor=edgecolor, lw=3)
                )

    # Add colorbar label
    cbar = ax.collections[0].colorbar  # Get colorbar object from the heatmap
    cbar.set_label("Probability", rotation=270, labelpad=15)  # Set label for colorbar

    # Final adjustments and save plot
    plt.title("Token Probability Heatmap with Selected Tokens Highlighted")
    plt.xlabel("Token Position")
    plt.ylabel("Choice Rank")
    plt.tight_layout()
    plt.savefig(f"tokens_highlighted_{outname}.png", dpi=300, bbox_inches="tight")
    return outcome
