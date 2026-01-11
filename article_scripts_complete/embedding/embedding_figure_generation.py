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

keep_columns = ["Title", "ID", "Abstract", "source"]
output_dir = "./tests/output"
run_name = f"{name}"

llm_processor.load_model(
    model_name=model_name,
    model_provider=model_provider,
    model_version=model_version,
    logits=False,
    embedding=True,
)

# -------------------------------------------------------------------- #
# Results from running embedding
df = pd.read_csv('./output/with_embedding.csv')


queries = ["rabbits", "cows", "polar bears", "financial markets"]
for query in queries:
    df = llm_processor.queryEmbedding(query,embedding_df = df, result_col=f"CD_{query}")


sources = ["cows", "rabbits", "polar bears", "financial markets"]
thresholds = np.arange(0, 1.01, 0.01)  # From 0 to 1 in steps of 0.01
results = []

for source in sources:
    source_df_string = source.replace(" ", "")
    score_col = f"CD_{source}"

    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found in DataFrame.")

    scores = df[score_col]

    for threshold in thresholds:
        predicted_relevance = scores > threshold

        TP = ((df["source"] == source_df_string) & (predicted_relevance)).sum()
        FP = ((df["source"] != source_df_string) & (predicted_relevance)).sum()
        FN = ((df["source"] == source_df_string) & (~predicted_relevance)).sum()
        TN = ((df["source"] != source_df_string) & (~predicted_relevance)).sum()

        results.append(
            {
                "Source": source,
                "Threshold": threshold,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "Precision": TP / (TP + FP) if TP + FP > 0 else 0,
                "Recall": TP / (TP + FN) if TP + FN > 0 else 0,
                "F1": 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0,
            }
        )

# Create a DataFrame of results
results_df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), sharey=True)
metrics = ["Precision", "Recall", "F1"]
sources = results_df["Source"].unique()
colors = plt.cm.tab10.colors  # Up to 10 distinct colors

for i, metric in enumerate(metrics):
    ax = axes[i]
    for j, source in enumerate(sources):
        df_source = results_df[results_df["Source"] == source]
        ax.plot(
            df_source["Threshold"],
            df_source[metric],
            label=source.capitalize(),
            color=colors[j],
        )
    ax.set_title(metric)
    ax.set_xlabel("Cosine Similarity")
    ax.grid(True)
    ax.set_ylim(0)
    if i == 0:
        ax.set_ylabel("Score")
    if i == 1:
        ax.legend(loc="lower left")

plt.tight_layout()
figure_name = "f1_per_recall_embeddings"
plt.savefig(f"{figure_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{figure_name}.eps")
plt.show()
