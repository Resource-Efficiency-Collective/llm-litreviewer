import articlefilter as af
from llama_cpp import Llama, LlamaGrammar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------------------- #
# This is the main file for each token probability variation in the paper.
# Plots are generated in the threshold_sweep_plots.py and cdf_visualization.py scripts. 

# -------------------------------------------------------------------- #
# Functions
def create_prompt(topic, relevant_token=1, irrelevant_token=0):
    prompt = (
        f"You are an expert in {topic}, and are conducting a literature review.  "
        f"You must read through the abstract below, and determine whether it is directly relevant to some aspect of {topic}.  "
        f"Your answer should be {relevant_token} (relevant) or {irrelevant_token} (irrelevant).  "
        f"Provide a BRIEF explanation.\nAbstract: {{abstract}}\n"
    )
    return prompt


def create_grammar_explanation(
    relevant_token=1, irrelevant_token=0, order="relevant_first"
):
    if order == "relevant_first":
        explanation = (
            f"Generate a structured JSON object following this schema: "
            f'{{ "relevance": boolean, "explanation": string }}. '
            f"Ensure boolean values are either {relevant_token} or {irrelevant_token}, "
            f"and the decision should be explained concisely."
        )
    elif order == "irrelevant_first":
        explanation = (
            f"Generate a structured JSON object following this schema: "
            f'{{ "relevance": boolean, "explanation": string }}. '
            f"Ensure boolean values are either {irrelevant_token} or {relevant_token}, "
            f"the decision should be explained concisely."
        )
    else:
        print("Invalid order")
        return

    return explanation


def create_grammar_gbnf(relevant_token=1, irrelevant_token=0):
    """
    Creates a GBNF grammar for JSON output with relevance and explanation fields.

    Parameters:
    relevant_token: Token for relevant content (number or string)
    irrelevant_token: Token for irrelevant content (number or string)

    Returns:
    str: GBNF grammar string

    Note: If using strings, both tokens must be strings for valid JSON output.
    """
    # Check if tokens are strings (not numbers)
    relevant_is_string = isinstance(relevant_token, str)
    irrelevant_is_string = isinstance(irrelevant_token, str)

    # Validate that both are the same type
    if relevant_is_string != irrelevant_is_string:
        raise ValueError(
            "Both tokens must be either numbers or strings, not mixed types"
        )

    # Create boolean rule based on type
    if relevant_is_string:
        # Add quotes around string tokens for valid JSON
        boolean_rule = f'"\\"{relevant_token}\\"" | "\\"{irrelevant_token}\\""'
    else:
        # No quotes for numeric tokens
        boolean_rule = f'"{relevant_token}" | "{irrelevant_token}"'

    gbnf = rf"""root ::= AbstractInfo
AbstractInfo ::= "{{" ws "\"relevance\":" ws boolean "," ws "\"explanation\":" ws string ws "}}"
string ::= "\"" ([^"]*) "\""
boolean ::= {boolean_rule}
ws ::= [ \t\n]*
"""
    return gbnf


# -------------------------------------------------------------------- #
# Load Test Abstracts
grouped = pd.read_csv('../../data/WOS/WOS_combined.csv',index_col=0)

# -------------------------------------------------------------------- #
# Load Model
name = "all_abstracts"
model_name = "llama3.2"
model_version = None

# model_name = "qwen2.5"
# model_version = "3b"

model_provider = "llama_cpp"
# input_file = "../tests/data/all_abstracts.csv"
# keep_columns = ["Title", "ID", "Abstract", "source"]

keep_columns = ["Article Title", "UT (Unique WOS ID)", "Abstract", "source"]
output_dir = "./output"
run_name = "all_test"
# user_prompt_path = "../user_messages/user_message_template.txt"
# grammar_folder = "../grammars/relevance_explanation_simplified"

# grammar_folder = "../grammars/relevance_explanation"


# -------------------------------------------------------------------- #
# Tested Combinations
combinations = []
# relevant_token, irrelevant_token, order, temperature,
# Done
combinations.append([1, 0, "relevant_first", 0])
# combinations.append([1, 0, "irrelevant_first", 0])
# combinations.append([0, 1, "irrelevant_first", 0])
# combinations.append([0, 1, "relevant_first", 0])
# combinations.append([3, 5, "relevant_first", 0])
# combinations.append([5, 3, "relevant_first", 0])
#
# combinations.append([6, 3, "relevant_first", 0])
# combinations.append([3, 6, "relevant_first", 0])

# combinations.append([9, 2, "relevant_first", 0])
# combinations.append([2, 9, "relevant_first", 0])
#
# combinations.append([2, 3, "relevant_first", 0])
# combinations.append([3, 2, "relevant_first", 0])

# Todo
# combinations.append([1, 0, "irrelevant_first", 0])
# combinations.append([1, 0, "irrelevant_first", 0])

# -------------------------------------------------------------------- #
# Run for all topics
# relevant_token = 1
# irrelevant_token = 0
# topic = "financial markets"
topics = ["financial markets", "rabbits", "cows", "polar bears"]

llm_processor = af.LLMProcessor_Pure()
llm_processor.load_model(
    model_name=model_name,
    model_provider=model_provider,
    model_version=model_version,
    logits=True,
)


llm_processor.load_abstracts(df=grouped)


# llm_processor.load_grammar(grammar_folder="../grammars/test_move_boolean")


# llm_processor.load_abstracts(df=financial_markets)
# llm_processor.load_abstracts(df=grouped.iloc[5:8])

# llm_processor.load_grammar(grammar_folder=grammar_folder)


all_results = []
for combination in combinations:
    print("Processing Combination")
    relevant_token = combination[0]
    irrelevant_token = combination[1]
    order = combination[2]
    temp = combination[3]
    other_params = {
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_penalty": 1.0,
    }
    results = {}

    for topic in topics:
        # topic = "financial markets"
        print(topic)

        prompt = create_prompt(topic, relevant_token, irrelevant_token)
        llm_processor.load_prompt(user_prompt_str=prompt)
        llm_processor.prepare_output_files(
            output_dir=output_dir,
            run_name=run_name,
            keep_columns=keep_columns,
            other_cols=["explanation"],
            relevance_label_col_name="relevance",
        )

        grammar_prompt = create_grammar_explanation(
            relevant_token=relevant_token,
            irrelevant_token=irrelevant_token,
            order=order,
        )

        # llm_processor.load_grammar(grammar_folder=grammar_folder)

        # llm_processor.load_grammar(grammar_folder="../grammars/test_move_boolean")

        llm_processor.grammar_prompt = grammar_prompt
        gbnf = create_grammar_gbnf(
            relevant_token=relevant_token, irrelevant_token=irrelevant_token
        )
        #
        # #
        llm_processor.grammar_string = gbnf
        llm_processor.grammar_file = None

        # llm_processor.load_grammar(grammar_folder=grammar_folder)

        returned_structure = llm_processor.runStructured(
            max_tokens=600,
            logprobs=3,
            relevant_token=relevant_token,
            irrelevant_token=irrelevant_token,
            temperature=temp,
            # **other_params,
        )
        returned_structure.to_csv(
            f"./output/{topic}_R_{relevant_token}_IR_{irrelevant_token}_OR_{order}_T_{temp}_MODEL_llama.csv"
        )
        results[topic] = returned_structure
    all_results.append(results)
