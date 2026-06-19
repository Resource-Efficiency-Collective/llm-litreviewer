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

df = df.dropna(subset=["Label_Human"])

merge_key = "UT (Unique WOS ID)"
df = df.drop_duplicates(subset=merge_key)


#
# df["Label_Human"] = (
#     df["Label_Human"]
#     .replace("BORDERLINE", True)
# )

# Ensure labels are boolean
# df["Label_Human"] = df["Label_Human"].astype(bool)
df["Label_Human"] = (
    df["Label_Human"]
    .replace("BORDERLINE", True)
    .replace({
        "True": True,
        "False": False,
        True: True,
        False: False
    })
)

# -------------------------------------------------------------------- #
# Single query
query = (
    "Lithium supply chain resilience and disruption "
    "(Not chemistry nor medical)"
)

df = llm_processor.queryEmbedding(
    query,
    embedding_df=df,
    result_col="CD_lithium"
)

# -------------------------------------------------------------------- #
# Evaluate across thresholds
thresholds = np.arange(0, 1.01, 0.01)

results = []
scores = df["CD_lithium"]

for threshold in thresholds:

    predicted_relevance = scores > threshold

    TP = ((df["Label_Human"] == True) & predicted_relevance).sum()
    FP = ((df["Label_Human"] != True) & predicted_relevance).sum()
    FN = ((df["Label_Human"] == True) & (~predicted_relevance)).sum()
    TN = ((df["Label_Human"] != True) & (~predicted_relevance)).sum()

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

# -------------------------------------------------------------------- #
# Find optimal threshold (max F1)
best_row = results_df.loc[results_df["F1"].idxmax()]

best_threshold = best_row["Threshold"]
best_f1 = best_row["F1"]

print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best F1 Score: {best_f1:.3f}")

# -------------------------------------------------------------------- #
# Create final LLM include/exclude label at optimal threshold
df["EMB_Label"] = df["CD_lithium"] > best_threshold

# Optional: inspect confusion matrix at optimal threshold
TP = ((df["Label_Human"] == True) & (df["EMB_Label"] == True)).sum()
FP = ((df["Label_Human"] != True) & (df["EMB_Label"] == True)).sum()
FN = ((df["Label_Human"] == True) & (df["EMB_Label"] == False)).sum()
TN = ((df["Label_Human"] != True) & (df["EMB_Label"] == False)).sum()

print("\nConfusion Matrix at Optimal Threshold")
print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"TN: {TN}")
print(f"FN: {FN}")

# -------------------------------------------------------------------- #
# Save outputs
df.to_csv("./output/lithium_embedding_with_llm_labels.csv", index=False)
results_df.to_csv("./output/lithium_threshold_results.csv", index=False)

# -------------------------------------------------------------------- #
# Sort for plotting if desired
results_df = results_df.sort_values("Threshold")

# -------------------------------------------------------------------- #
# Single combined plot
plt.figure(figsize=(6, 4))

plt.plot(results_df["Threshold"], results_df["Precision"], label="Precision")
plt.plot(results_df["Threshold"], results_df["Recall"], label="Recall")
plt.plot(results_df["Threshold"], results_df["F1"], label="F1")

# Mark optimal threshold
plt.axvline(
    best_threshold,
    linestyle="--",
    alpha=0.7,
    label=f"Best Threshold = {best_threshold:.3f}"
)

plt.xlabel("Cosine Similarity Threshold")
plt.ylabel("Score")
plt.title("Lithium Supply Chain Query Performance")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

figure_name = "lithium_embedding_metrics_2"

plt.tight_layout()
# plt.savefig(f"{figure_name}.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"{figure_name}.eps")
plt.show()
