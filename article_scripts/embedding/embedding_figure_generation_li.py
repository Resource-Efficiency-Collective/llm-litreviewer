import articlefilter as af
from llama_cpp import Llama, LlamaGrammar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize processor
llm_processor = af.LLMProcessor_Pure()

model_name = "nomic-embed-text"
model_version = None
model_provider = "llama_cpp"

llm_processor.load_model(
    model_name=model_name,
    model_provider=model_provider,
    model_version=model_version,
    logits=False,
    embedding=True,
)

# -------------------------------------------------------------------- #
# Load data
df = pd.read_csv('./output/with_embedding_li.csv')

df= df.dropna(subset=["Label_Human"])

merge_key = "UT (Unique WOS ID)"
df= df.drop_duplicates(subset=merge_key)

df["Label_Human"] = (
    df["Label_Human"]
    .replace("BORDERLINE", True)
)

# -------------------------------------------------------------------- #
# Single query
# query = "Lithium Supplychain D"
# query = "Lithium trade network vulnerability"
# query="Recent years have seen increasing concerns on supply chain risks of lithium, a critical material for achieving e-mobility transition and climate ambitions. These risks propagate both along the life cycle and across national boundaries in a multilayer network. However, most previous studies are either only based on static network measures or focused on individual layers, ignoring dynamic cascading risks and interconnected and interdependent relationships along life cycle stages and across economies. Here, we integrated trade-linked material flow and complex network analyses to investigate intricate interconnections, interdependencies, and systematic risks of the global lithium supply chain. Both static and dynamic measures of the global lithium supply chain network exhibit a robust-yet-fragile property: robust for random shocks yet fragile for targeted shocks and robust for small or local disruptions yet fragile for large or cascading failures. Portugal, Brazil, Singapore, Canada, Finland, Norway, South Africa, Israel, Hungary, and the United Arab Emirates are most likely to be affected by supply disruptions. A hypothetical USA-China trade decoupling will increase the severity and susceptibility of network-wide failures by around 5%. Our results call for global collaborations and collective efforts to balance efficiency and security and avoid a zero-sum game in securing the lithium supply chain."
query = (
    "Lithium supply chain resilience and disruption (Not chemistry nor medical)"
)
#
# query= """
# - lithium supply chains
# - lithium trade 
# - supply disruptions or resilience
# - geopolitical risks 
#
# - NOT battery chemistry or electrochemistry
# - NOT medical 
# - NOT non-lithium 
# - NOT generic supply chain optimization
# """
df = llm_processor.queryEmbedding(
    query,
    embedding_df=df,
    result_col="CD_lithium"
)
# df.to_csv('./output/lithium_embedding_df.csv')

# -------------------------------------------------------------------- #
# Evaluate across thresholds
# source_label = "lithium"
thresholds = np.arange(0, 1.01, 0.001)

results = []
scores = df["CD_lithium"]

for threshold in thresholds:
    predicted_relevance = scores > threshold

    TP = ((df["Label_Human"] == True) & predicted_relevance).sum()
    FP = ((df["Label_Human"] !=True ) & predicted_relevance).sum()
    FN = ((df["Label_Human"] ==True ) & (~predicted_relevance)).sum()
    TN = ((df["Label_Human"] !=True ) & (~predicted_relevance)).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1',ascending=False)


# -------------------------------------------------------------------- #
# Single combined plot
plt.figure(figsize=(6, 4))

plt.plot(results_df["Threshold"], results_df["Precision"], label="Precision")
plt.plot(results_df["Threshold"], results_df["Recall"], label="Recall")
plt.plot(results_df["Threshold"], results_df["F1"], label="F1")

plt.xlabel("Cosine Similarity Threshold"e
plt.ylabel("Score")
plt.title("Lithium Supply Chain Query Performance")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

figure_name = "lithium_embedding_metrics"
plt.tight_layout()
plt.savefig(f"{figure_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{figure_name}.eps")
plt.show()
