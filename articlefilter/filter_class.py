import pandas as pd
import os
from tqdm import tqdm
import yaml
import numpy as np
import logging
import json

from sklearn.metrics.pairwise import cosine_similarity
from articlefilter.embeddings import hex_to_embedding
from articlefilter.embeddings import embedding_to_hex
from articlefilter.grammar_tools import extract_probabilities_from_index
from articlefilter.grammar_tools import get_value_index


def chunk_list(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


class LLMProcessor_Pure:
    def __init__(
        self,
        # temperature=None,
        # model="llama3.2",
        # version="latest",
        # max_context_size=2048,
        # verbose=False,
        # n_threads=4,
    ):
        # Setting up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Adjust level (DEBUG, INFO, etc.)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Other initialization
        self.input_file = None
        self.abstract_column = "Abstract"
        self.abstract_key = "UT (Unique WOS ID)"
        self.keep_columns = []
        self.log_stats = False
        self.evaluation_settings = {}
        self.system_message = None
        self.token_options = None
        self.think_tag = False
        self.grammar_prompt = None
        self.grammar_string = None
        self.grammar_file = None

        self.results_df = None

    @classmethod
    def from_config(cls, config_path: str, logits: bool, embedding: bool):
        # Load YAML
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Create instance
        instance = cls()

        # Load model
        model_cfg = config["model"]
        llama_cpp_settings = model_cfg.get("llama_cpp_settings", {})
        grammar_cfg = config.get("grammar", None)
        if grammar_cfg:
            instance.load_grammar(**grammar_cfg)
        else:
            instance.grammar_file = None

        # if grammar_cfg:
        #     grammar_file = grammar_cfg.get("grammar_file", None)
        #     grammar_prompt = grammar_cfg.get("grammar_prompt", None)
        #     if grammar_prompt:
        #         with open(grammar_prompt, "r") as f:
        #             grammar_prompt = f.read().strip()
        # else:
        #     grammar_file = None
        #     grammar_prompt = None

        print(f"llama_cpp_settings: {llama_cpp_settings}")
        instance.load_model(
            model_name=model_cfg["name"],
            model_version=model_cfg.get("version"),
            model_provider=model_cfg.get("provider", "ollama"),
            logits=logits,
            embedding=embedding,
            grammar_file=instance.grammar_file,
            **llama_cpp_settings,
        )
        instance.think_tag = model_cfg.get("thinktag", False)

        # Load abstracts
        abstracts_cfg = config["abstracts"]
        instance.load_abstracts(
            input_file=abstracts_cfg["input_file"],
        )

        # Load prompts

        prompt_cfg = config.get("prompts", None)
        if prompt_cfg:
            instance.load_prompt(
                system_prompt_path=prompt_cfg.get("system_prompt_path"),
                user_prompt_path=prompt_cfg.get("user_prompt_path"),
            )
        instance.token_options = prompt_cfg.get("token_options", None)

        # Prepare output files
        output_cfg = config["output"]
        instance.prepare_output_files(
            output_dir=output_cfg["output_dir"],
            run_name=output_cfg["run_name"],
            log_dir=output_cfg.get("log_dir"),
            keep_columns=output_cfg.get("keep_columns", []),
            relevance_label_col_name=output_cfg.get(
                "relevance_label_col_name", "Label"
            ),
            other_cols=output_cfg.get("other_cols", []),
        )

        # Optional settings
        instance.log_stats = config.get("log_stats", False)

        # Evaluation Settings

        evalulation_cfg = config.get("evaluation", {})
        instance.evaluation_settings = evalulation_cfg

        return instance

    def print_settings(self):
        print(
            "# -------------------------------------------------------------------- #"
        )
        print("PROMPT (user_message):")
        print(self.user_message)
        print("PROMPT (system_message):")
        print(self.system_message)
        print("Grammar:")
        print(self.grammar_prompt)
        print(self.grammar_file)

    def load_grammar(
        self,
        grammar_folder=None,
        grammar_file=None,
        grammar_prompt=None,
    ):
        if grammar_folder:
            grammar_file = grammar_folder + "/grammar.gbnf"
            grammar_prompt = grammar_folder + "/grammar_explanation.txt"

        if grammar_prompt:
            with open(grammar_prompt, "r") as f:
                grammar_prompt = f.read().strip()
        self.grammar_prompt = grammar_prompt
        self.grammar_file = grammar_file
        print(self.grammar_file)

    def load_model(
        self,
        model_name,
        model_version=None,
        model_provider="ollama",
        logits=False,
        embedding=False,
        api_key=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.model_provider_name = model_provider
        if model_provider == "gpt":
            from articlefilter.providers.openai_provider import OpenAIProvider

            self.model_provider = OpenAIProvider(model_name)
        elif model_provider == "ollama":
            from articlefilter.providers.ollama_provider import OllamaProvider

            self.model_provider = OllamaProvider(model_name, model_version)

        elif model_provider == "llama_cpp":
            from articlefilter.providers.LlamaCppProvider import LlamaCppProvider

            self.model_provider = LlamaCppProvider(
                model_name, model_version, logits, embedding, **kwargs
            )
        elif model_provider == "gemini":
            from articlefilter.providers.gemini_provider import GeminiProvider

            self.model_provider = GeminiProvider(model_name, api_key=api_key)

        else:
            raise ValueError(f"Invalid model_provider: {model_provider}")

        print("Model loaded successfully")

    # def load_model_advanced(self, model_name, model_version=None, **kwargs):
    #     self.model_name = model_name
    #     self.model_provider_name = "llama_cpp"
    #     from articlefilter.providers.LlamaCppProvider import LlamaCppProvider
    #
    #     self.model_provider = LlamaCppProvider(model_name, model_version, **kwargs)

    def load_abstracts(self, input_file=None, df=None):
        """
        Load abstracts from a CSV file or an existing pandas DataFrame.

        Parameters:
        -----------
        input_file : str, optional
            Path to a CSV file containing abstracts.
        df : pandas.DataFrame, optional
            A pre-loaded DataFrame containing abstracts.
        """
        if df is not None:
            # If a DataFrame is provided, use it directly
            self.abstract_df_all = df.copy()
        else:
            # Otherwise, load from a CSV file
            if input_file:
                self.input_file = input_file
            if not hasattr(self, "input_file") or self.input_file is None:
                raise ValueError(
                    "Either a DataFrame or an input_file must be provided."
                )
            self.abstract_df_all = pd.read_csv(self.input_file, low_memory=False)

        print("Abstracts Loaded")
        print(self.abstract_df_all.head(4))

        # Confirm that abstract_column, abstract_key and keep_columns are in the dataframe.

    def load_prompt(
        self,
        system_prompt_path=None,
        system_prompt_str=None,
        user_prompt_path=None,
        user_prompt_str=None,
    ):
        # Load system message
        if system_prompt_path:
            with open(system_prompt_path, "r") as f:
                self.system_message = f.read().strip()
        elif system_prompt_str is not None:
            self.system_message = system_prompt_str.strip()

        # Load user message
        if user_prompt_path:
            with open(user_prompt_path, "r") as f:
                self.user_message = f.read()
        elif user_prompt_str is not None:
            self.user_message = user_prompt_str

    # def load_prompt(self, system_prompt_path=None, user_prompt_path=None):
    #
    #     # Load system message from the specified text file
    #     if system_prompt_path:
    #         with open(system_prompt_path, "r") as f:
    #             self.system_message = f.read().strip()
    #
    #     if user_prompt_path:
    #         with open(user_prompt_path, "r") as f:
    #             self.user_message = f.read()

    def prepare_output_files(
        self,
        output_dir,
        run_name,
        log_dir=None,
        keep_columns=[],
        relevance_label_col_name="Label",
        other_cols=[],
    ):
        self.output_file = os.path.join(
            output_dir, f"output_{run_name}_{self.model_name}.csv"
        )

        if self.log_stats == True and log_dir is not None:
            self.output_log = os.path.join(
                log_dir, f"output_log_{run_name}_{self.model_name}.csv"
            )

        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        if self.log_stats == True and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        self.keep_columns = keep_columns
        self.relevance_label_col_name = relevance_label_col_name
        self.other_cols = other_cols

    def run(self, batch_size=10, **evaluation_settings):
        # Check Current Status

        evaluation_settings = self.evaluation_settings | evaluation_settings

        # Handle existing output files
        if os.path.exists(self.output_file):
            current_results = pd.read_csv(self.output_file)
            current_length = len(current_results)
            self.abstract_df_select = self.abstract_df_all.iloc[current_length:]
        else:
            current_length = 0
            self.abstract_df_select = self.abstract_df_all

        print(f"{current_length}/{len(self.abstract_df_all)} already processed")

        if self.log_stats == True and os.path.exists(self.output_log):
            current_log = pd.read_csv(self.output_log)

        # Set up storage arrays
        results = []
        result_log = []
        batch_count = 0

        if self.system_message:
            system_words = len(self.system_message.split())

        for i, row in tqdm(
            self.abstract_df_select.iterrows(), total=len(self.abstract_df_select)
        ):
            abstract = row["Abstract"]

            if pd.isna(abstract):
                abstract = "NO ABSTRACT PROVIDED"
                self.logger.warning(f"Row {i} missing abstract")
            user_message_filled = self.user_message.format(abstract=abstract)
            # print(self.evaluation_settings)
            result_text, response = self.model_provider.chat(
                self.system_message, user_message_filled, **self.evaluation_settings
            )
            # print(result_text)

            # If Logging Stats
            if self.log_stats == True and self.model_provider_name != "gpt":
                prompt_stats = {
                    f"{self.abstract_key}": row[self.abstract_key],
                    "eval_count": response["eval_count"],
                    "eval_duration": response["eval_duration"],
                    "eval_speed": response["eval_count"]
                    / response["eval_duration"]
                    * 1e9,
                    "words_system": system_words,
                    "user_system": len(self.user_message.split()),
                    "abstract_length": len(abstract.split()),
                    "prompt_eval_count": response["prompt_eval_count"],
                    "prompt_eval_duration": response["prompt_eval_duration"],
                    "prompt_speed": response["prompt_eval_count"]
                    / response["eval_duration"]
                    * 1e9,
                    "total_duration": response["total_duration"],
                }
                result_log.append(prompt_stats)
            classification = {col: row[col] for col in self.keep_columns}
            classification["Abstract"] = abstract  # override if needed
            classification["Label"] = None  # default label
            for other_col in self.other_cols:
                classification[other_col] = None  # default for additional columns

            # Parse the LLM output dynamically
            field_mapping = {
                self.relevance_label_col_name: self.relevance_label_col_name
            }
            for col in self.other_cols:
                field_mapping[col] = col

            for line in result_text.split("\n"):
                for field in field_mapping.keys():
                    if line.startswith(f"{field}:") or line.startswith(f"`{field}:"):
                        classification[field_mapping[field]] = line.split(":", 1)[
                            1
                        ].strip()

            # Check for missing fields
            missing_fields = []
            for field in field_mapping.keys():
                if classification[field_mapping[field]] is None:
                    missing_fields.append(field)

            # Log missing fields
            if missing_fields:
                self.logger.warning(
                    f"Row {i} missing required fields in result_text: {', '.join(missing_fields)}"
                )

            results.append(classification)

            # Write to CSV every `batch_size` rows
            batch_count += 1
            if batch_count % batch_size == 0:
                # pd.DataFrame(results).to_csv(output_file, mode='a', index=False, header=(batch_count == batch_size))
                pd.DataFrame(results).to_csv(
                    self.output_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(self.output_file),
                )
                results = []  # Clear the batch results after writing
                # print(f"Appended {batch_size} rows to {output_file}")

                if self.log_stats is True:
                    pd.DataFrame(result_log).to_csv(
                        self.output_log,
                        mode="a",
                        index=False,
                        header=not os.path.exists(self.output_log),
                    )
                    result_log = []

        # Final write for any remaining rows
        if results:
            pd.DataFrame(results).to_csv(
                self.output_file, mode="a", index=False, header=False
            )
            print(f"Final append: {len(results)} rows to {self.output_file}")

    def run_binary(
        self, batch_size=10, prob_type="prob", rerun=False, **evaluation_settings
    ):

        # Handle existing output files
        if os.path.exists(self.output_file) and rerun == False:
            current_results = pd.read_csv(self.output_file)
            current_length = len(current_results)
            self.abstract_df_select = self.abstract_df_all.iloc[current_length:]
        else:
            current_length = 0
            self.abstract_df_select = self.abstract_df_all

        print(f"{current_length}/{len(self.abstract_df_all)} already processed")

        if self.log_stats == True and os.path.exists(self.output_log):
            current_log = pd.read_csv(self.output_log)

        # Set up storage arrays
        results = []
        result_log = []
        batch_count = 0

        evaluation_settings = self.evaluation_settings | evaluation_settings
        print(evaluation_settings)

        for i, row in tqdm(
            self.abstract_df_select.iterrows(), total=len(self.abstract_df_select)
        ):
            abstract = row["Abstract"]

            classification = {col: row[col] for col in self.keep_columns}
            classification["Abstract"] = abstract  # override if needed

            for other_col in self.other_cols:
                classification[other_col] = None  # default for additional columns
            if pd.isna(abstract):
                abstract = "NO ABSTRACT PROVIDED"
                self.logger.warning(f"Row {i} missing abstract")

            user_message_filled = self.user_message.format(abstract=abstract)
            input_text = user_message_filled
            print(input_text)
            # print(f"User Message: {input_text}")
            result_text, response = self.model_provider.boolean_probs(
                input_text=input_text, **evaluation_settings
            )
            # print(response)
            # print(response)
            # print(self.token_options)

            # tokens = response["choices"][0]["logprobs"]["tokens"]
            logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
            print(logprobs)

            if prob_type == "prob":
                logprobs = {token: np.exp(logit) for token, logit in logprobs.items()}
            elif prob_type == "logit":
                pass
            max_key = max(logprobs, key=logprobs.get)
            print(logprobs)
            selected_tokens = {
                token: logprobs.get(str(token), np.nan) for token in self.token_options
            }
            print(selected_tokens)
            classification = {**classification, **selected_tokens}

            classification[self.relevance_label_col_name] = max_key

            results.append(classification)

            # print(logprobs)
            # Write to CSV every `batch_size` rows
            batch_count += 1
            if batch_count % batch_size == 0:
                # pd.DataFrame(results).to_csv(output_file, mode='a', index=False, header=(batch_count == batch_size))
                pd.DataFrame(results).to_csv(
                    self.output_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(self.output_file),
                )
                results = []  # Clear the batch results after writing
                # print(f"Appended {batch_size} rows to {output_file}")

                if self.log_stats is True:
                    pd.DataFrame(result_log).to_csv(
                        self.output_log,
                        mode="a",
                        index=False,
                        header=not os.path.exists(self.output_log),
                    )
                    result_log = []

        # Final write for any remaining rows
        if results:
            pd.DataFrame(results).to_csv(
                self.output_file, mode="a", index=False, header=False
            )
            print(f"Final append: {len(results)} rows to {self.output_file}")
        return results

    def runStructured(
        self,
        write_csv=False,
        batch_size=10,
        relevant_token="1",
        irrelevant_token="0",
        **evaluation_settings,
    ):

        # Handle existing output files
        if os.path.exists(self.output_file):
            current_results = pd.read_csv(self.output_file)
            current_length = len(current_results)
            self.abstract_df_select = self.abstract_df_all.iloc[current_length:]
        else:
            current_length = 0
            self.abstract_df_select = self.abstract_df_all

        print(f"{current_length}/{len(self.abstract_df_all)} already processed")

        # Set up storage arrays
        results = []
        result_log = []
        batch_count = 0
        all_columns_to_extract = self.other_cols + [self.relevance_label_col_name]

        evaluation_settings = self.evaluation_settings | evaluation_settings

        for i, row in tqdm(
            self.abstract_df_select.iterrows(), total=len(self.abstract_df_select)
        ):
            # print(i)
            abstract = row["Abstract"]

            if pd.isna(abstract):
                abstract = "NO ABSTRACT PROVIDED"
                self.logger.warning(f"Row {i} missing abstract")
            user_message_filled = self.user_message.format(abstract=abstract)

            # outcome = response["choices"][0]["text"]
            input_text = user_message_filled
            if self.grammar_prompt:

                input_text += "\n" + self.grammar_prompt
            # print(input_text)
            # print(input_text)

            result_text, response = self.model_provider.calc_grammar(
                input_text=input_text,
                grammar=self.grammar_file,
                grammar_string=self.grammar_string,
                **evaluation_settings,
            )
            tokens = response["choices"][0]["logprobs"]["tokens"]
            # print(response["choices"][0]["text"])
            index = get_value_index(tokens, key="relevance")
            # print(f"Index: {index}")

            # print(response, index)

            relevant_probs = extract_probabilities_from_index(
                response,
                index,
                relevant_token=relevant_token,
                irrelevant_token=irrelevant_token,
            )  # Dictionary
            # return tokens
            # print(index, relevant_probs)

            # Convert the outcome string to JSON
            try:
                outcome_json = json.loads(result_text)
                result = {
                    column: outcome_json.get(column, "Column not returned")
                    for column in all_columns_to_extract
                }
            except json.JSONDecodeError as e:
                result = {
                    column: "Unable to Parse" for column in all_columns_to_extract
                }
                print("Error parsing JSON:", e)

            orig_columns_to_keep = {col: row[col] for col in self.keep_columns}
            row_dict = orig_columns_to_keep | result | relevant_probs

            results.append(row_dict)
            if write_csv == True:

                # Write to CSV every `batch_size` rows
                batch_count += 1
                if batch_count % batch_size == 0:
                    # pd.DataFrame(results).to_csv(output_file, mode='a', index=False, header=(batch_count == batch_size))
                    pd.DataFrame(results).to_csv(
                        self.output_file,
                        mode="a",
                        index=False,
                        header=not os.path.exists(self.output_file),
                    )
                    results = []  # Clear the batch results after writing

        # Final write for any remaining rows
        if results and write_csv == True:
            pd.DataFrame(results).to_csv(
                self.output_file, mode="a", index=False, header=False
            )
            print(f"Final append: {len(results)} rows to {self.output_file}")
        else:
            return pd.DataFrame(results)

            #
            # print("RESULT: ")
            # print(result_text)
            # print("----------")
            # return result_text

    # def runEmbedding(self, batch_size=10):
    #
    #     # Handle existing output files
    #     if os.path.exists(self.output_file):
    #         current_results = pd.read_csv(self.output_file)
    #         current_length = len(current_results)
    #         self.abstract_df_select = self.abstract_df_all.iloc[current_length:]
    #     else:
    #         current_length = 0
    #         self.abstract_df_select = self.abstract_df_all
    #
    #     print(f"{current_length}/{len(self.abstract_df_all)} already processed")
    #
    #     abstracts = self.abstract_df_select["Abstract"].to_list()
    #     chunked_abstracts = chunk_list(abstracts, batch_size)
    #
    #     batch_count = 0
    #     for batch in tqdm(chunked_abstracts, total=len(self.abstract_df_select)):
    #         res = self.model_provider.llm.create_embedding(batch)
    #         embeddings = [
    #             d["embedding"] for d in res["data"]
    #         ]  # Produces 1 Embedding per abstract
    #
    #         # Contains hexadecimal representation of embedding for csv column
    #         hexadecimal_array = [embedding_to_hex(emb) for emb in embeddings]
    #
    #
    #         # Write hexadecimal array to 'Embeddings' column of self.output_file
    #
    #         if batch_count % batch_size == 0:
    #             # pd.DataFrame(results).to_csv(output_file, mode='a', index=False, header=(batch_count == batch_size))
    #             pd.DataFrame(results).to_csv(
    #                 self.output_file,
    #                 mode="a",
    #                 index=False,
    #                 header=not os.path.exists(self.output_file),
    #             )
    def queryEmbedding(
        self,
        query,
        embedding_df=None,
        write_csv=False,
        outname="embedding_query",
        result_col="CD",
    ):
        # Get embedding for Query
        res = self.model_provider.llm.create_embedding(query)
        query_embedding = [
            d["embedding"] for d in res["data"]
        ]  # Produces 1 Embedding per sentence

        # Get embeddings to query against
        if embedding_df is None:
            embedding_df = self.results_df
        embeddings = embedding_df["Embeddings"]

        emb_vectors = [hex_to_embedding(emb) for emb in embeddings]
        emb_vectors_np = np.array(emb_vectors)  # shape: (n_samples, embedding_dim)

        query_vec = np.array(query_embedding[0]).reshape(1, -1)
        query_similarities = cosine_similarity(query_vec, emb_vectors_np).flatten()
        sorted_indices = np.argsort(query_similarities)[::-1]
        embedding_df[result_col] = query_similarities

        query_results = embedding_df.iloc[sorted_indices].reset_index(drop=True)
        if write_csv:
            query_results.to_csv(outname)

        return query_results

    def runEmbedding(self, batch_size=10, write_csv=False):

        # Handle existing output files
        self.abstract_df_all["Abstract"] = self.abstract_df_all["Abstract"].fillna(
            "NO ABSTRACT PROVIDED"
        )
        if os.path.exists(self.output_file):
            current_results = pd.read_csv(self.output_file)
            current_length = len(current_results)
            self.abstract_df_select = self.abstract_df_all.iloc[current_length:]
        else:
            current_length = 0
            self.abstract_df_select = self.abstract_df_all

        print(f"{current_length}/{len(self.abstract_df_all)} already processed")

        # Extract abstracts and chunk them
        abstracts = self.abstract_df_select["Abstract"].to_list()
        chunked_abstracts = chunk_list(abstracts, batch_size)

        batch_count = 0
        all_results = []
        for batch_start in tqdm(
            range(0, len(chunked_abstracts)), total=len(chunked_abstracts)
        ):
            batch = chunked_abstracts[batch_start]
            # Generate embeddings for the current batch
            res = self.model_provider.llm.create_embedding(batch)

            # Extract the embeddings from the response
            embeddings = [
                d["embedding"] for d in res["data"]
            ]  # Produces 1 embedding per abstract

            # Convert embeddings to hexadecimal representation
            hexadecimal_array = [embedding_to_hex(emb) for emb in embeddings]

            # Prepare the results DataFrame with abstracts and embeddings
            results = pd.DataFrame({"Abstract": batch, "Embeddings": hexadecimal_array})

            # If self.keep_columns exists, add those columns to the results
            if self.keep_columns:
                # Get the indices of the current batch in self.abstract_df_select
                batch_indices = self.abstract_df_select.index[
                    batch_start * batch_size : (batch_start + 1) * batch_size
                ].tolist()

                for col in self.keep_columns:

                    # Add the additional columns based on the batch indices
                    results[col] = self.abstract_df_select.loc[
                        batch_indices, col
                    ].values

            cols = [
                col for col in results.columns if col not in ["Abstract", "Embeddings"]
            ]
            results = results[cols + ["Abstract", "Embeddings"]]
            if write_csv:
                # Write the results to CSV (append mode)
                results.to_csv(
                    self.output_file,
                    mode="a",  # Append mode
                    index=False,
                    header=(
                        batch_count == 0 and not os.path.exists(self.output_file)
                    ),  # Add header only for the first batch
                )
            else:
                all_results.append(results)

            batch_count += 1  # Increment batch counter
        if not write_csv:
            self.results_df = pd.concat(all_results, ignore_index=True)
            return self.results_df
