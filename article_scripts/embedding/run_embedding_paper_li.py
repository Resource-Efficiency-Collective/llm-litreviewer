import articlefilter as af
from llama_cpp import Llama, LlamaGrammar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

llm_processor = af.LLMProcessor_Pure()
model_name = "nomic-embed-text"
model_version = None
model_provider = "llama_cpp"
name = ""

keep_columns = ["Article Title", "UT (Unique WOS ID)", "Abstract","Label_Human"]
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

# input_WOS_abstracts = '../../data/WOS/WOS_combined.csv'
input_WOS_abstracts = '../../data/WOS/Lithium/lithium_combined.xlsx'
df = pd.read_excel(input_WOS_abstracts)

llm_processor.load_abstracts(df=df)

df = llm_processor.runEmbedding(write_csv=False)
df.to_csv("./output/with_embedding_li.csv")

