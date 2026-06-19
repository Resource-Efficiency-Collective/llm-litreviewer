import articlefilter as af
from llama_cpp import Llama, LlamaGrammar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------- #
# FUNCTIONS
# -------------------------------------------------------------------- #

def create_prompt(
    review_topic,
    inclusion_criteria,
    relevant_token=1,
    irrelevant_token=0,
):
    prompt = (
        f"You are assisting with a systematic literature review.\n\n"
        f"Review Topic:\n{review_topic}\n\n"
        f"Inclusion Criteria:\n{inclusion_criteria}\n\n"
        f"Read the abstract below and determine whether the paper "
        f"is relevant to the review topic.\n\n"
        f"Return {relevant_token} if the abstract is relevant.\n"
        f"Return {irrelevant_token} if the abstract is irrelevant.\n\n"
        f"Provide a brief explanation.\n\n"
        f"Abstract:\n{{abstract}}\n"
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
# LOAD REAL ABSTRACT DATASET
# -------------------------------------------------------------------- #

# grouped = pd.read_csv(
#     "../../data/WOS/lithium_supply_chain_dataset.csv",
#     index_col=0,
# )
#
# grouped = pd.read_csv('../../data/WOS/Lithium/lithium_combined.csv')

grouped = pd.read_excel('../../data/WOS/Lithium/lithium_combined.xlsx')

merge_key = "UT (Unique WOS ID)"
grouped = grouped.drop_duplicates(subset=merge_key)

grouped = grouped.dropna(subset=["Label_Human"])
# grouped = grouped.iloc[:3]

# Expected columns:
#
# "Article Title"
# "Abstract"
# "Label_Human"
# "source"
#
# ground_truth:
# 1 = relevant
# 0 = irrelevant


# -------------------------------------------------------------------- #
# REVIEW QUESTION
# -------------------------------------------------------------------- #

review_topic = (
    "Lithium supply chain resilience and disruption"
)

inclusion_criteria = """
Include papers that discuss:
- lithium supply chains
- lithium trade dependencies
- lithium supply disruptions or resilience
- geopolitical risks affecting lithium availability
- EV battery supply-chain vulnerability involving lithium

Exclude papers focused primarily on:
- lithium battery chemistry or electrochemistry
- cathode/anode/electrolyte materials
- medical lithium use
- non-lithium critical minerals
- generic supply chain optimization
"""


# -------------------------------------------------------------------- #
# LOAD MODEL
# -------------------------------------------------------------------- #

# model_name = "llama3.2"
# model_version = None

model_name = "qwen2.5"
model_version = "3b"

model_provider = "llama_cpp"

keep_columns = [
    "Article Title",
    "UT (Unique WOS ID)",
    "Abstract",
    "Label_Human",
]

output_dir = "./output"
run_name = "lithium_supply_chain_screening"


# -------------------------------------------------------------------- #
# TOKEN CONFIGURATIONS
# -------------------------------------------------------------------- #

combinations = []

combinations.append([1, 0, "relevant_first", 0])

# Suggested additions for revised paper:
combinations.append([1, 0, "irrelevant_first", 0])
combinations.append([3, 5, "relevant_first", 0])
combinations.append([5, 3, "relevant_first", 0])

combinations.append([0, 1, "irrelevant_first", 0])
combinations.append([0, 1, "relevant_first", 0])

#
combinations.append([6, 3, "relevant_first", 0])
combinations.append([3, 6, "relevant_first", 0])
#
combinations.append([9, 2, "relevant_first", 0])
combinations.append([2, 9, "relevant_first", 0])
# #
combinations.append([2, 3, "relevant_first", 0])
combinations.append([3, 2, "relevant_first", 0])


# -------------------------------------------------------------------- #
# LOAD PROCESSOR
# -------------------------------------------------------------------- #

llm_processor = af.LLMProcessor_Pure()

llm_processor.load_model(
    model_name=model_name,
    model_provider=model_provider,
    model_version=model_version,
    logits=True,
)

llm_processor.load_abstracts(df=grouped)


# -------------------------------------------------------------------- #
# MAIN LOOP
# -------------------------------------------------------------------- #

all_results = []

i = 0

for combination in combinations:

    print("Processing Combination")

    relevant_token = combination[0]
    irrelevant_token = combination[1]
    order = combination[2]
    temp = combination[3]

    prompt = create_prompt(
        review_topic=review_topic,
        inclusion_criteria=inclusion_criteria,
        relevant_token=relevant_token,
        irrelevant_token=irrelevant_token,
    )

    llm_processor.load_prompt(
        user_prompt_str=prompt
    )

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

    llm_processor.grammar_prompt = grammar_prompt

    gbnf = create_grammar_gbnf(
        relevant_token=relevant_token,
        irrelevant_token=irrelevant_token,
    )

    llm_processor.grammar_string = gbnf
    llm_processor.grammar_file = None

    returned_structure = llm_processor.runStructured(
        max_tokens=300,
        logprobs=5,
        relevant_token=relevant_token,
        irrelevant_token=irrelevant_token,
        temperature=temp,
    )

    # ---------------------------------------------------------------- #
    # SAVE RESULTS
    # ---------------------------------------------------------------- #

    output_file = (
        f"./output/WOS/lithium/"
        f"R_{relevant_token}_"
        f"IR_{irrelevant_token}_"
        f"OR_{order}_"
        f"T_{temp}_"
        f"MODEL_{model_name}_{i}.csv"
    )

    returned_structure.to_csv(output_file)

    all_results.append(returned_structure)

    i += 1
