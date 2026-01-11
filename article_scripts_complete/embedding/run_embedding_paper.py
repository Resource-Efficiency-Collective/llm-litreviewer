import articlefilter as af
from llama_cpp import Llama, LlamaGrammar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

llm_processor = af.LLMProcessor_Pure()
model_name = "nomic-embed-text"
model_version = None
model_provider = "llama_cpp"
name = "rabbits"

keep_columns = ["Article Title", "UT (Unique WOS ID)", "Abstract", "source"]
output_dir = "./output"
run_name = f"{name}"

llm_processor.load_model(
    model_name=model_name,
    model_provider=model_provider,
    model_version=model_version,
    logits=False,
    embedding=True,
)

llm_processor.prepare_output_files(
    output_dir=output_dir,
    run_name=run_name,
    keep_columns=keep_columns,
    other_cols=["explanation"],
    relevance_label_col_name="relevance",
)

input_WOS_abstracts = '../../data/WOS/WOS_combined.csv'
llm_processor.load_abstracts(input_file=input_WOS_abstracts)

df = llm_processor.runEmbedding(write_csv=False)
df.to_csv("./output/with_embedding.csv")
