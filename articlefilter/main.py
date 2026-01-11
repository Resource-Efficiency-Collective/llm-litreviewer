from llama_cpp import Llama, LlamaGrammar
import numpy as np
import json
import os
import pandas as pd
from tqdm import tqdm
# from .plotting import plot_token_probabilities_with_highlighted_choices
from .probs import extract_relevant_probabilities
from .probs import extract_probabilities_from_index
from .probs import get_value_index
import pickle


def merge_datasets(df1, df2):
    merged_df = pd.merge(df1, df2, on="Abstract_key", how="outer")

    # Ensure 'Abstract' from df1 is retained (assuming identical)
    if "Abstract_x" in merged_df.columns and "Abstract_y" in merged_df.columns:
        merged_df["Abstract"] = merged_df["Abstract_x"]  # Keep from df1
        merged_df.drop(columns=["Abstract_x", "Abstract_y"], inplace=True)
    return merged_df


def read_txt(filepath):
    with open(filepath, "r") as txt_file:
        txt_content = txt_file.read()
    return txt_content


def importUserPrompt(filepath):
    user_prompt = read_txt(filepath)
    return user_prompt


def importSystemPrompt(filepath):
    system_prompt = read_txt(filepath)
    return system_prompt


def get_ollama_model_blob_path(model="llama3.2", version="latest"):
    """
    Returns the path to the model blob (GGUF file) for the specified Ollama model.

    Parameters:
        model (str): The name of the model (default is 'llama3.2').
        version (str): The version of the model (default is 'latest').

    Returns:
        str: The absolute path to the model blob file.
    """
    # Construct the manifest file path
    manifest_path = (
        f"~/.ollama/models/manifests/registry.ollama.ai/library/{model}/{version}"
    )
    manifest_path = os.path.expanduser(manifest_path)

    # Read the manifest file
    try:
        with open(manifest_path, "r") as file:
            data = json.load(file)

        # Extract the model blob digest
        model_blob = next(
            layer["digest"]
            for layer in data["layers"]
            if layer["mediaType"] == "application/vnd.ollama.image.model"
        )

        # Format the blob path
        model_blob = model_blob.replace(":", "-")  # Replace ':' with '-' in digest
        blob_path = f"~/.ollama/models/blobs/{model_blob}"
        return os.path.expanduser(blob_path)

    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    except (KeyError, StopIteration):
        raise ValueError("Model blob digest not found in manifest file.")


def load_ollama_model(
    model="llama3.2",
    version="latest",
    max_context_size=2048,
    verbose=False,
    n_threads=4,
):
    manifest_path = (
        f"~/.ollama/models/manifests/registry.ollama.ai/library/{model}/{version}"
    )
    manifest_path = os.path.expanduser(manifest_path)

    with open(manifest_path, "r") as file:
        data = json.load(file)

    # Extract model blob (first layer with "application/vnd.ollama.image.model")
    model_blob = next(
        layer["digest"]
        for layer in data["layers"]
        if layer["mediaType"] == "application/vnd.ollama.image.model"
    )

    print("Model Blob Name:", model_blob)
    model_blob = model_blob.replace(":", "-")
    # Replace : with - in model_blob
    blob_path = f"~/.ollama/models/blobs/{model_blob}"  # gguf file

    blob_path = os.path.expanduser(blob_path)
    # n_threads = 4
    llm = Llama(
        model_path=blob_path,
        logits_all=True,
        max_context_size=max_context_size,
        n_ctx=max_context_size,
        verbose=verbose,
        n_threads=n_threads,
    )
    print(f"Lllama Init Settings:")
    print(f"Max Content Size={max_context_size}")
    print(f"n_ctx = {max_context_size}")
    print(f"n_threads = {n_threads}")
    return llm


def calculate_probabilities(
    llm,
    input_text,
    verbose=False,
    max_tokens=10,
    temperature=0.2,
    top_p=1,
):
    output = llm.create_completion(
        input_text,
        top_p=top_p,
        max_tokens=max_tokens,
        echo=False,
        temperature=temperature,
        # stop=["Relevant", "Irrelevant"],
        logprobs=5,
    )  # to return top 5 tokens

    results_first_token = output["choices"][0]["logprobs"]["top_logprobs"][0]
    # results_second_token = output['choices'][0]['logprobs']['top_logprobs'][1]
    token_probabilities = {
        token: np.exp(logit) for token, logit in results_first_token.items()
    }
    if verbose:
        print(f"{input_text}...")
        for token, p in token_probabilities.items():
            print(f"Token: {token}:  Probability {p}")
    # token_probabilities = {}
    # for token,logit in results_first_token.items():
    #
    #     # p = 1/(1+np.exp(logit))
    #     p = np.exp(logit)
    #     token_probabilities[token] = p
    #     # print(f'Token: {token},{logit},{p}')
    #     if verbose:
    #         print(f'Token: {token}:  Probability {p}')

    return token_probabilities


class LLMProcessor:
    def __init__(
        self,
        temperatures,
        model="llama3.2",
        version="latest",
        max_context_size=2048,
        verbose=False,
        n_threads=4,
    ):
        self.model = model
        self.version = version
        self.max_context_size = max_context_size
        self.verbose = verbose
        self.llm = load_ollama_model(
            model, version, max_context_size, verbose, n_threads=n_threads
        )
        self.user_prompts = dict()
        self.system_prompt = None
        self.abstracts = dict()
        self.prob_dict = dict()
        self.temperatures = temperatures
        self.grammar = None
        self.grammar_explanation = None
        self.responses = dict()

    # def load_grammar(self,grammar):
    #     self.grammar = grammar

    def extract_boolean(self):
        for key, response in self.responses.items():
            relevant_probs = extract_relevant_probabilities(response)
            print(relevant_probs)

    def extract_bool_2(self, key):
        response = self.responses[key]
        tokens = response["choices"][0]["logprobs"]["tokens"]
        index = get_value_index(tokens, key="relevance")
        relevant_probs = extract_probabilities_from_index(response, index)
        print(relevant_probs)
        return relevant_probs

    def save_responses(self, filename="responses.pkl"):
        """
        Save the responses dictionary to a pickle file.
        :param filename: Name of the file where the responses will be saved.
        """
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.responses, f)
            print(f"Responses saved to {filename}")
        except Exception as e:
            print(f"Error saving responses: {e}")

    def load_responses(self, filename="responses.pkl"):
        """
        Load the responses dictionary from a pickle file.
        :param filename: Name of the file to load the responses from.
        """
        try:
            with open(filename, "rb") as f:
                self.responses = pickle.load(f)
            print(f"Responses loaded from {filename}")
        except Exception as e:
            print(f"Error loading responses: {e}")

    def print_selected(self):
        for key, response in self.responses.items():

            outcome = response["choices"][0][
                "text"
            ]  # Actually selected text based on constraints and logprobs
            print(f"{key}: {outcome}")

    def load_grammar(self, grammar_directory):
        grammar_text = grammar_directory + "/grammar.gbnf"
        explanation_text = grammar_directory + "/grammar_explanation.txt"

        # Load the grammar
        self.grammar = LlamaGrammar.from_file(grammar_text)

        # Load the explanation text file
        with open(explanation_text, "r", encoding="utf-8") as f:
            self.grammar_explanation = f.read()

    def import_user_prompts(self, filepath, user_prompt_key="default"):
        """Import a user prompt from a file and associate it with a key."""
        user_prompt = read_txt(filepath)
        self.user_prompts[user_prompt_key] = (
            user_prompt  # Store the prompt with its key
        )
        return user_prompt

    def import_system_prompt(self, filepath):
        """Import system prompt from a file."""
        self.system_prompt = read_txt(filepath)
        return self.system_prompt

    # def load_abstract(self,abstract_id,abstract):
    #     self.abstracts[abstract_id] = abstract

    def load_abstract(self, abstract_id, file_path):
        # Open the file in read mode
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                abstract_content = file.read()  # Read the entire content of the file
            self.abstracts[abstract_id] = (
                abstract_content  # Store the content with the given abstract_id
            )
            print(f"Abstract {abstract_id} loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
        except Exception as e:
            print(f"Error while loading abstract: {str(e)}")

    def batch_load_abstracts(self, abstracts):
        """Load abstracts into self.abstracts.

        - If `abstracts` is a **file path**, it loads the CSV into a DataFrame.
        - If `abstracts` is a **DataFrame**, it converts it to a dictionary.
        - If `abstracts` is a **dictionary**, it assigns it directly.
        """
        # If `abstracts` is a file path (string), attempt to load it
        if isinstance(abstracts, str):
            if not os.path.isfile(abstracts):
                raise FileNotFoundError(f"File not found: {abstracts}")
            abstracts = pd.read_csv(abstracts)

        # Process DataFrame
        if isinstance(abstracts, pd.DataFrame):
            if (
                "UT (Unique WOS ID)" in abstracts.columns
                and "Abstract" in abstracts.columns
            ):
                self.abstracts = dict(
                    zip(abstracts["UT (Unique WOS ID)"], abstracts["Abstract"])
                )
            else:
                raise ValueError(
                    "DataFrame must contain 'UT (Unique WOS ID)' and 'Abstract' columns."
                )

        # Process Dictionary
        elif isinstance(abstracts, dict):
            self.abstracts = abstracts

        else:
            raise TypeError(
                "Input should be a file path (CSV), a pandas DataFrame, or a dictionary."
            )
        # def batch_load_abstracts(self, abstracts):
        #     """Load abstracts into self.abstracts.
        #
        #     Either as a dictionary of 'UT (Unique WOS ID)' -> 'Abstract' or
        #     as a pandas DataFrame with a column 'UT (Unique WOS ID)' and 'Abstract'."""
        #     if isinstance(abstracts, pd.DataFrame):
        #         # If abstracts is a DataFrame, convert to dictionary
        #         if (
        #             "UT (Unique WOS ID)" in abstracts.columns
        #             and "Abstract" in abstracts.columns
        #         ):
        #             self.abstracts = dict(
        #                 zip(abstracts["UT (Unique WOS ID)"], abstracts["Abstract"])
        #             )
        #         else:
        #             raise ValueError(
        #                 "DataFrame must contain 'UT (Unique WOS ID)' and 'Abstract' columns."
        #             )
        #     elif isinstance(abstracts, dict):
        #         # If abstracts is already a dictionary, just assign
        #         self.abstracts = abstracts
        #     else:
        #         raise TypeError(
        #             "Input should be either a pandas DataFrame or a dictionary."
        #         )

    def load_ollama_model(
        self, model="llama3.2", version="latest", max_context_size=2048, verbose=False
    ):
        """Load the Llama model."""
        self.llm = load_ollama_model(model, version, max_context_size, verbose)

    def calculate_probability(
        self,
        abstract_key,
        user_prompt,
        temperature,
        max_tokens=5,
        top_p=1,
    ):
        """Calculate token probabilities for a given abstract using a specific user prompt."""
        abstract = self.abstracts[abstract_key]
        input_text = user_prompt.replace(
            "{abstract}", abstract
        )  # Replace placeholder with abstract text
        probs = calculate_probabilities(
            self.llm,
            input_text,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return probs

    def heat_maps(self):
        for name, response in self.responses.items():
            out1 = plot_token_probabilities_with_highlighted_choices(
                response, outname=f"{name}"
            )
            print(f"{name}:  {out1}")

    def calc_grammar(
        self,
        abstract_key,
        user_prompt,
        max_tokens=500,
        logprobs=3,
        top_p=1,
        temperature=0,
        presence_penalty=0,
        repeat_penalty=1,
        frequency_penalty=1,
    ):

        print(f"Abstract: {abstract_key}")
        print(f"Running with t_pop = {top_p} and temp = {temperature}")
        print(f"Presence Penalty = {presence_penalty}")
        print(f"Repeat Penalty = {repeat_penalty}")
        print(f"Frequency Penalty = {frequency_penalty}")
        abstract = self.abstracts[abstract_key]
        input_text = user_prompt.replace("{abstract}", abstract)
        response = self.llm(
            input_text,
            grammar=self.grammar,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_p=top_p,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
        )
        # response = self.llm(input_text, max_tokens=max_tokens,logprobs=3)
        return response

    def iterate_abstracts_grammar(
        self,
        max_tokens=500,
        logprobs=3,
        top_p=1,
        temperature=0,
        presence_penalty=0,
        repeat_penalty=1,
        frequency_penalty=1,
        save_to_csv=False,
        csv_filename="abstract_probabilities.csv",
        save_every_n_abstracts=None,
        verbose=False,
    ):
        data_for_csv = []
        header_written = False
        print(f"Running with top_p = {top_p} and temp = {temperature}")

        # Track index for periodic saving
        for idx, (abstract_key, abstract) in enumerate(
            tqdm(self.abstracts.items(), total=len(self.abstracts))
        ):
            abstract_probabilities = {}  # Store probabilities for this abstract

            for prompt_key, prompt in self.user_prompts.items():
                input_text = (
                    prompt.replace("{abstract}", abstract) + self.grammar_explanation
                )
                if verbose is True:
                    print(f"{input_text}")

                response = self.llm(
                    input_text,
                    grammar=self.grammar,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    top_p=top_p,
                    temperature=temperature,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repeat_penalty=repeat_penalty,
                )

                self.responses[f"{abstract_key}_{prompt_key}_temp_{temperature}"] = (
                    response
                )

                outcome = response["choices"][0][
                    "text"
                ]  # Actually selected text based on constraints and logprobs

                # Extract the text from the response
                outcome = response["choices"][0]["text"]

                # Convert the outcome string to JSON
                try:
                    outcome_json = json.loads(outcome)
                    explanation = outcome_json["explanation"]
                    result = outcome_json["relevance"]
                    # abstract_probabilities["explanation"] = explanation
                    # print(f"Explanation: {explanation}")
                    # print(f"Result: {result}")
                    # print("Successfully parsed JSON:", outcome_json)
                except json.JSONDecodeError as e:
                    # print("Error parsing JSON:", e)
                    explanation = "Unable to Parse"
                    result = "Unable to Parse"
                    # abstract_probabilities["explanation"] = explanation

                # Extract relevant probabilities
                # Removed (can cause issues if 0/1 as low probability tokens elsewhere
                # relevant_probs = extract_relevant_probabilities(response)

                tokens = response["choices"][0]["logprobs"]["tokens"]
                index = get_value_index(tokens, key="relevance")
                relevant_probs = extract_probabilities_from_index(response, index)
                prob_0 = relevant_probs[0]
                prob_1 = relevant_probs[1]
                # Returns a list of tuples

                # if relevant_probs:  # Ensure it's not empty
                #     prob_0, prob_1 = zip(
                #         *relevant_probs
                #     )  # Unpack tuples into two lists
                # else:
                #     prob_0, prob_1 = [], []  # Handle empty case

                abstract_probabilities[f"{prompt_key}_temp_{temperature}_1"] = prob_1
                abstract_probabilities[f"{prompt_key}_temp_{temperature}_0"] = prob_0
                abstract_probabilities[f"{prompt_key}_temp_{temperature}_relevance"] = (
                    result
                )
                abstract_probabilities[
                    f"{prompt_key}_temp_{temperature}_explanation"
                ] = explanation

            # Store results for this abstract
            self.prob_dict[abstract_key] = abstract_probabilities

            # Prepare CSV row
            row = {
                "Abstract_key": abstract_key,
                "Abstract": abstract,
            }

            for prompt_key in self.user_prompts.keys():
                # print(f"prompt_key: {prompt_key}")
                row[f"{prompt_key}_temp_{temperature}_1"] = abstract_probabilities.get(
                    f"{prompt_key}_temp_{temperature}_1", None
                )
                row[f"{prompt_key}_temp_{temperature}_0"] = abstract_probabilities.get(
                    f"{prompt_key}_temp_{temperature}_0", None
                )
                row[f"{prompt_key}_temp_{temperature}_explanation"] = (
                    abstract_probabilities.get(
                        f"{prompt_key}_temp_{temperature}_explanation", None
                    )
                )

                row[f"{prompt_key}_temp_{temperature}_relevance"] = (
                    abstract_probabilities.get(
                        f"{prompt_key}_temp_{temperature}_relevance", None
                    )
                )

            # print(f"Row keys:  {row}")
            # print(f"Row keys:  {row.keys()}")
            data_for_csv.append(row)

            # Save to CSV periodically if needed
            if (
                save_to_csv
                and save_every_n_abstracts
                and (idx + 1) % save_every_n_abstracts == 0
            ):
                df = pd.DataFrame(data_for_csv)
                df.to_csv(
                    csv_filename, index=False, mode="a", header=not header_written
                )
                data_for_csv = []  # Reset buffer
                header_written = True

        # Final CSV save
        if save_to_csv and data_for_csv:
            df = pd.DataFrame(data_for_csv)
            # print(df.columns)
            df.to_csv(csv_filename, index=False, mode="a", header=not header_written)
            print(f"Saved to CSV {csv_filename}")

    # def iterate_abstracts_grammar(
    #     self,
    #     max_tokens=500,
    #     logprobs=3,
    #     top_p=1,
    #     temperature=0,
    #     presence_penalty=0,
    #     repeat_penalty=1,
    #     frequency_penalty=1,
    #     save_to_csv=False,
    #     csv_filename="abstract_probabilities.csv",
    #     save_every_n_abstracts=None,
    # ):
    #
    #     data_for_csv = []
    #     header_written = False
    #     print(f"Running with t_pop = {top_p} and temp = {temperature}")
    #     for abstract_key, abstract in tqdm(
    #         self.abstracts.items(), total=len(self.abstracts.keys())
    #     ):
    #
    #         abstract_probabilities = (
    #             {}
    #         )  # Store probabilities for each abstract with different prompts and temperatures
    #         for prompt_key, prompt in self.user_prompts.items():
    #             input_text = prompt.replace("{abstract}", abstract)
    #             input_text = input_text + self.grammar_explanation
    #             print(f"{input_text}")
    #             response = self.llm(
    #                 input_text,
    #                 grammar=self.grammar,
    #                 max_tokens=max_tokens,
    #                 logprobs=logprobs,
    #                 top_p=top_p,
    #                 temperature=temperature,
    #                 presence_penalty=presence_penalty,
    #                 frequency_penalty=frequency_penalty,
    #                 repeat_penalty=repeat_penalty,
    #             )
    #
    #             prob_0, prob_1 = extract_relevant_probabilities(response)
    #
    #             abstract_probabilities[f"{user_prompt_key}_temp_{temperature}_1"] = (
    #                 prob_1
    #             )
    #             abstract_probabilities[f"{user_prompt_key}_temp_{temperature}_0"] = (
    #                 prob_0
    #             )
    #
    #         # Store results for this abstract
    #         self.prob_dict[abstract_key] = (
    #             abstract_probabilities  # Store results for this abstract
    #         )
    #
    #         # Prepare the row for this abstract, to be added to the CSV
    #         row = {
    #             "Abstract_key": abstract_key,
    #             "Abstract": self.abstracts[abstract_key],
    #         }
    #
    #         # Add probabilities for each user prompt and temperature combination
    #         for user_prompt_key, user_prompt in self.user_prompts.items():
    #             for temperature in self.temperatures:
    #                 prob_1 = abstract_probabilities.get(
    #                     f"{user_prompt_key}_temp_{temperature}_1", None
    #                 )
    #                 prob_0 = abstract_probabilities.get(
    #                     f"{user_prompt_key}_temp_{temperature}_0", None
    #                 )
    #                 row[f"{user_prompt_key}_temp_{temperature}_1"] = prob_1
    #                 row[f"{user_prompt_key}_temp_{temperature}_0"] = prob_0
    #
    #         # Append the row to the list
    #         data_for_csv.append(row)
    #
    #         # Save to CSV periodically if required
    #         if (
    #             save_to_csv
    #             and save_every_n_abstracts
    #             and (idx + 1) % save_every_n_abstracts == 0
    #         ):
    #             df = pd.DataFrame(data_for_csv)
    #             df.to_csv(
    #                 csv_filename, index=False, mode="a", header=not header_written
    #             )
    #             data_for_csv = []  # Reset data for next batch
    #             header_written = True
    #
    #     # If save_to_csv is True, save the results to a CSV file after all abstracts are processed
    #     if save_to_csv and data_for_csv:
    #         df = pd.DataFrame(data_for_csv)
    #         df.to_csv(csv_filename, index=False, mode="a", header=not header_written)
    #         print(f"Saved to CSV {csv_filename}")
    #
    #         # self.responses[f"{abstract_key}_{prompt_key}_temp_{temperature}"] = (
    #         #     response
    #         # )

    def iterate_abstracts(
        self,
        max_tokens=5,
        top_p=1,
        save_to_csv=False,
        csv_filename="abstract_probabilities.csv",
        save_every_n_abstracts=None,
    ):
        """Iterate over all abstracts and calculate probabilities for each using all user prompts."""
        data_for_csv = []
        header_written = False
        for idx, key in tqdm(
            enumerate(self.abstracts.keys()), total=len(self.abstracts.keys())
        ):
            abstract_probabilities = (
                {}
            )  # Store probabilities for each abstract with different prompts and temperatures
            for user_prompt_key, user_prompt in self.user_prompts.items():
                for temperature in self.temperatures:
                    probabilities = self.calculate_probability(
                        key,
                        user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    prob_1 = probabilities.get(
                        "1", None
                    )  # Default to None if not present
                    prob_0 = probabilities.get(
                        "0", None
                    )  # Default to None if not present
                    # Store probabilities for each combination of user prompt and temperature
                    abstract_probabilities[
                        f"{user_prompt_key}_temp_{temperature}_1"
                    ] = prob_1
                    abstract_probabilities[
                        f"{user_prompt_key}_temp_{temperature}_0"
                    ] = prob_0

            # Store results for this abstract
            self.prob_dict[key] = (
                abstract_probabilities  # Store results for this abstract
            )

            # Prepare the row for this abstract, to be added to the CSV
            row = {"Abstract_key": key, "Abstract": self.abstracts[key]}

            # Add probabilities for each user prompt and temperature combination
            for user_prompt_key, user_prompt in self.user_prompts.items():
                for temperature in self.temperatures:
                    prob_1 = abstract_probabilities.get(
                        f"{user_prompt_key}_temp_{temperature}_1", None
                    )
                    prob_0 = abstract_probabilities.get(
                        f"{user_prompt_key}_temp_{temperature}_0", None
                    )
                    row[f"{user_prompt_key}_temp_{temperature}_1"] = prob_1
                    row[f"{user_prompt_key}_temp_{temperature}_0"] = prob_0

            # Append the row to the list
            data_for_csv.append(row)

            # Save to CSV periodically if required
            if (
                save_to_csv
                and save_every_n_abstracts
                and (idx + 1) % save_every_n_abstracts == 0
            ):
                df = pd.DataFrame(data_for_csv)
                df.to_csv(
                    csv_filename, index=False, mode="a", header=not header_written
                )
                data_for_csv = []  # Reset data for next batch
                header_written = True

        # If save_to_csv is True, save the results to a CSV file after all abstracts are processed
        if save_to_csv and data_for_csv:
            df = pd.DataFrame(data_for_csv)
            df.to_csv(csv_filename, index=False, mode="a", header=not header_written)
            print(f"Saved to CSV {csv_filename}")

    # def iterate_abstracts(self,max_tokens=5,save_to_csv=False,csv_filename="abstract_probabilities.csv"):
    #     """Iterate over all abstracts and calculate probabilities for each using all user prompts."""
    #     data_for_csv = []
    #     for key in tqdm(self.abstracts.keys(), total=len(self.abstracts.keys())):
    #         abstract_probabilities = {}  # Store probabilities for each abstract with different prompts and temperatures
    #         for user_prompt_key, user_prompt in self.user_prompts.items():
    #             for temperature in self.temperatures:
    #                 probabilities = self.calculate_probability(key, user_prompt, temperature=temperature,max_tokens=max_tokens)
    #                 prob_1 = probabilities.get('1', None)  # Default to None if not present
    #                 prob_0 = probabilities.get('0', None)  # Default to None if not present
    #                 # Store probabilities for each combination of user prompt and temperature
    #                 abstract_probabilities[f"{user_prompt_key}_temp_{temperature}"] = probabilities
    #                 self.prob_dict[key] = abstract_probabilities  # Store results for this abstract
    #                 if save_to_csv:
    #                     row = {
    #                         'Abstract_key': key,
    #                         'Abstract': self.abstracts[key],
    #                         f'{user_prompt_key}_temp_{temperature}_1': prob_1,
    #                         f'{user_prompt_key}_temp_{temperature}_0': prob_0
    #                     }
    #                     data_for_csv.append(row)
    #
    #     if save_to_csv:
    #         df = pd.DataFrame(data_for_csv)
    #         df.to_csv(csv_filename, index=False)
    #
    #
