import numpy as np


# def extract_relevant_probabilities(response):
#     """
#     Extracts probabilities for the relevant item (0 or 1) from log probabilities.
#
#     Args:
#         response (dict): API response containing log probability data.
#
#     Returns:
#         dict: Dictionary with '0' and '1' probabilities at each token position.
#     """
#     logprobs_data = response["choices"][0]["logprobs"]["top_logprobs"]
#     relevant_probs = {"0": [], "1": []}
#
#     for token_info in logprobs_data:
#         prob_0 = np.exp(
#             token_info.get("0", float("-inf"))
#         )  # Convert logprob to probability
#         prob_1 = np.exp(
#             token_info.get("1", float("-inf"))
#         )  # Convert logprob to probability
#
#         relevant_probs["0"].append(prob_0)
#         relevant_probs["1"].append(prob_1)
#
#     return relevant_probs


def extract_relevant_probabilities(response):
    """
    Extracts probabilities for the relevant item (0 or 1) from log probabilities,
    but only for tokens where both '0' and '1' exist.

    Args:
        response (dict): API response containing log probability data.

    Returns:
        list: List of tuples (P(0), P(1)) for tokens containing both 0 and 1.
    """
    logprobs_data = response["choices"][0]["logprobs"]["top_logprobs"]
    relevant_probs = []

    for token_info in logprobs_data:
        if "0" in token_info and "1" in token_info:
            prob_0 = np.exp(token_info["0"])  # Convert logprob to probability
            prob_1 = np.exp(token_info["1"])  # Convert logprob to probability
            relevant_probs.append((prob_0, prob_1))

    return relevant_probs


def extract_probabilities_from_index(response, index):
    logprobs_data = response["choices"][0]["logprobs"]["top_logprobs"]
    bool_token = logprobs_data[index]
    prob_0 = bool_token.get("0")
    prob_1 = bool_token.get("1")
    relevant_probs = [np.exp(prob_0), np.exp(prob_1)]
    return relevant_probs


def get_value_index(tokens, key):
    """
    Finds the index of the value corresponding to a given key in a tokenized JSON list.

    Parameters:
    tokens (list): A list of JSON tokens.
    key (str): The key whose value index we want to find.

    Returns:
    int: The index of the value if found, otherwise -1.
    """
    # Reconstruct potential split words
    reconstructed_tokens = []
    i = 0
    while i < len(tokens):
        if (
            i > 0 and tokens[i - 1].isalpha() and tokens[i].isalpha()
        ):  # Handle split words
            reconstructed_tokens[-1] += tokens[i]
        else:
            reconstructed_tokens.append(tokens[i])
        i += 1

    try:
        # Find the index of the key in the reconstructed list
        key_index = reconstructed_tokens.index(key)

        # Find the colon (":") immediately after the key
        index_colon = key_index + 1
        while (
            index_colon < len(reconstructed_tokens)
            and reconstructed_tokens[index_colon] != '":'
        ):
            index_colon += 1

        if index_colon >= len(reconstructed_tokens):
            return -1  # Colon not found

        # Find the value (skipping whitespace tokens)
        index_value = index_colon + 1
        while (
            index_value < len(reconstructed_tokens)
            and reconstructed_tokens[index_value].strip() == ""
        ):
            index_value += 1

        return (
            tokens.index(reconstructed_tokens[index_value])
            if index_value < len(reconstructed_tokens)
            else -1
        )

    except ValueError:
        return -1  # Key not found
