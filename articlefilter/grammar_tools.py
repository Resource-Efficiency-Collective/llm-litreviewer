import numpy as np


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


def extract_probabilities_from_index(
    response, index, relevant_token="1", irrelevant_token="0"
):
    logprobs_data = response["choices"][0]["logprobs"]["top_logprobs"]
    bool_token = logprobs_data[index]
    # print(bool_token)
    # print(relevant_token)
    # print(irrelevant_token)
    prob_0 = bool_token.get(str(irrelevant_token), np.nan)
    # print(prob_0)
    prob_1 = bool_token.get(str(relevant_token), np.nan)
    # print(np.exp(prob_0), np.exp(prob_1))
    # print(prob_1)
    # relevant_probs = [np.exp(prob_0), np.exp(prob_1)]
    relevant_probs = {irrelevant_token: np.exp(prob_0), relevant_token: np.exp(prob_1)}

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


# def get_value_index(tokens, key):
#     """
#     Finds the index of the value corresponding to a given key in a tokenized JSON list.
#     Parameters:
#     tokens (list): A list of JSON tokens.
#     key (str): The key whose value index we want to find.
#     Returns:
#     int: The index of the value if found, otherwise -1.
#     """
#     # Search for the key by reconstructing it from tokens
#     i = 0
#     while i < len(tokens):
#         # Try to match the key starting from position i
#         reconstructed = ""
#         j = i
#         while j < len(tokens):
#             # Extract alphanumeric parts from token
#             alpha_part = "".join(c for c in tokens[j] if c.isalpha())
#             if alpha_part:
#                 reconstructed += alpha_part
#                 if reconstructed == key:
#                     # Found the key, now find the value
#                     # Move past the key tokens and look for ":"
#                     k = j + 1
#                     while k < len(tokens):
#                         if '":' in tokens[k] or tokens[k] == ":":
#                             # Found colon, now skip to next non-whitespace, non-quote token
#                             k += 1
#                             while k < len(tokens):
#                                 token = tokens[k]
#                                 # Skip whitespace and quote-only tokens
#                                 if token.strip() and token.strip() not in [
#                                     '"',
#                                     "'",
#                                     ' "',
#                                     '" ',
#                                 ]:
#                                     # Check if this token contains the actual value
#                                     # (not just quotes/punctuation)
#                                     if any(c.isalnum() for c in token):
#                                         return k
#                                 k += 1
#                             return -1
#                         k += 1
#                     return -1
#                 j += 1
#             else:
#                 break
#         i += 1
#
#     return -1
