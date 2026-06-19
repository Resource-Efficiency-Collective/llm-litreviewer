# import pandas as pd
# import numpy as np
# import re
# from statsmodels.stats.contingency_tables import mcnemar
# from pathlib import Path
#
# # ------------------------------------------------------------
# # Load Generative Results
# # ------------------------------------------------------------
#
# generative_results_path = '../generative/lithium_threshold_summary.csv'
# results_df = pd.read_csv(generative_results_path)
# best_row = results_df.loc[results_df["Max_F1"].idxmax()]
# tau_emb = best_row["Best_Threshold"]
#
#
# summary_df = pd.read_csv("summary_df.csv")
#
# # ------------------------------------------------------------
# # Get BEST embedding threshold
# # ------------------------------------------------------------
#
# best_row = results_df.loc[results_df["Max_F1"].idxmax()]
# tau_emb = best_row["Best_Threshold"]
#
# # ------------------------------------------------------------
# # Load embedding scored dataset (must exist)
# # ------------------------------------------------------------
#
# embedding_df = '../embedding/output/lithium_embedding_with_llm_labels.csv'
# df_emb = pd.read_csv(embedding_df,index_col=0)
#
# # emb_score_col = "CD_lithium"
# #
# # df_emb["emb_pred"] = df_emb[emb_score_col] >= tau_emb
#
# # ------------------------------------------------------------
# # Helper: McNemar
# # ------------------------------------------------------------
#
# def mcnemar_test(y_true, pred_a, pred_b):
#
#     a_correct = (pred_a == y_true)
#     b_correct = (pred_b == y_true)
#
#     b = np.sum(a_correct & ~b_correct)  # A correct, B wrong
#     c = np.sum(~a_correct & b_correct)  # A wrong, B correct
#
#     table = [[0, b],
#              [c, 0]]
#
#     res = mcnemar(table, exact=True)
#
#     return b, c, res.pvalue
#
# # ------------------------------------------------------------
# # Loop over generative models
# # ------------------------------------------------------------
#
# base_path = Path("../generative/labelled_csvs")
# files = list(base_path.glob("R0-IR1-IF_labelled.csv")) # File template
#
# file_template = R3-IR2-RF-llama3.2_labelled.csv
# files = list(base_path.glob(file_template)) # File template
# # R3-IR2-RF-llama3.2_labelled.csv
#
# results = []
#
# for file in files:
#
#     match = re.search(
#         r"R_(.*?)_IR_(.*?)_OR_(.*?)_T_0_MODEL_llama3\.2_(\d+)\.csv",
#         file.name
#     )
#
#     if not match:
#         continue
#
#     r = match.group(1)
#     label = f"R{r}"
#
#     df_gen = pd.read_csv(file)
#
#     # --------------------------------------------------------
#     # Merge embedding + generative on document ID
#     # --------------------------------------------------------
#
#     df = df_emb[["UT (Unique WOS ID)", "LLM_Label"]].merge(
#         df_gen[["UT (Unique WOS ID)", r]],
#         on="UT (Unique WOS ID)",
#         how="inner"
#     )
#
#     df = df.dropna()
#
#     # --------------------------------------------------------
#     # Get generative threshold
#     # --------------------------------------------------------
#
#     gen_tau = summary_df.loc[
#         summary_df["Config"].str.contains(label)
#     ]["Best_Threshold"].values[0]
#
#     df["gen_pred"] = df[r] >= gen_tau
#
#     # --------------------------------------------------------
#     # Ground truth MUST exist in either dataset
#     # (assumed already present in embedding df OR generative df)
#     # --------------------------------------------------------
#
#     if "Label_Human" in df_emb.columns:
#         y_true = df_emb.set_index("UT (Unique WOS ID)").loc[
#             df["UT (Unique WOS ID)"], "Label_Human"
#         ].astype(int).values
#     else:
#         raise ValueError("Label_Human must exist in embedding dataset")
#
#     # --------------------------------------------------------
#     # McNemar test
#     # --------------------------------------------------------
#
#     b, c, p = mcnemar_test(y_true, df["emb_pred"].values, df["gen_pred"].values)
#
#     results.append({
#         "Config": label,
#         "b_emb_correct_gen_wrong": b,
#         "c_emb_wrong_gen_correct": c,
#         "p_value": p,
#         "win": "embedding" if b < c else "generative"
#     })
#
# # ------------------------------------------------------------
# # Save results
# # ------------------------------------------------------------
#
# out = pd.DataFrame(results).sort_values("p_value")
# out.to_csv("mcnemar_embedding_vs_generative.csv", index=False)
#
# print(out)
# print("\nSaved: mcnemar_embedding_vs_generative.csv")
#



import pandas as pd
import numpy as np
import re
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

# ------------------------------------------------------------
# Load embedding dataset
# ------------------------------------------------------------

emb_df = '../embedding/output/lithium_embedding_with_llm_labels.csv'
emb_df = pd.read_csv(emb_df,index_col=0)

emb_df = emb_df.dropna(subset=["Label_Human", "EMB_Label"])

# ensure binary
emb_df["Label_Human"] = emb_df["Label_Human"].replace("BORDERLINE", True).astype(int)
emb_df["EMB_Label"] = emb_df["EMB_Label"].astype(int)

# index for fast alignment
emb_df = emb_df.set_index("UT (Unique WOS ID)")

# ------------------------------------------------------------
# Generative datasets
# ------------------------------------------------------------

base_path = Path("../generative/labelled_csvs")
files = list(base_path.glob("*.csv"))

results = []

# ------------------------------------------------------------
# McNemar helper
# ------------------------------------------------------------

def mcnemar_test(y_true, pred_a, pred_b):
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    b = np.sum(a_correct & ~b_correct)  # A correct, B wrong
    c = np.sum(~a_correct & b_correct)  # A wrong, B correct

    table = [[0, b],
             [c, 0]]

    res = mcnemar(table, exact=True)

    return b, c, res.pvalue

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

for file in files:

    df_gen = pd.read_csv(file)

    if "Label_Gen" not in df_gen.columns:
        continue

    df_gen = df_gen.dropna(subset=["Label_Human", "Label_Gen"])

    df_gen["Label_Human"] = df_gen["Label_Human"].replace("BORDERLINE", True).astype(int)
    df_gen["Label_Gen"] = df_gen["Label_Gen"].astype(int)

    df_gen = df_gen.set_index("UT (Unique WOS ID)")

    # --------------------------------------------------------
    # Align embedding + generative
    # --------------------------------------------------------

    common_ids = emb_df.index.intersection(df_gen.index)

    df = pd.DataFrame(index=common_ids)

    df["y_true"] = emb_df.loc[common_ids, "Label_Human"]
    df["emb"] = emb_df.loc[common_ids, "EMB_Label"]
    df["gen"] = df_gen.loc[common_ids, "Label_Gen"]

    # --------------------------------------------------------
    # McNemar test
    # --------------------------------------------------------

    b, c, p = mcnemar_test(
        df["y_true"].values,
        df["emb"].values,
        df["gen"].values
    )

    # parse config name
    label = file.stem  # full filename without .csv

    results.append({
        "Config": label,
        "b_emb_correct_gen_wrong": b,
        "c_emb_wrong_gen_correct": c,
        "p_value": p,
        "win": "embedding" if b > c else "generative"
    })

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

out = pd.DataFrame(results).sort_values("p_value")

out.to_csv("mcnemar_embedding_vs_generative.csv", index=False)

print(out)
print("\nSaved: mcnemar_embedding_vs_generative.csv")



latex_df = out.copy()

# ------------------------------------------------------------
# 1. Clean config names
# ------------------------------------------------------------
latex_df["Config"] = latex_df["Config"].str.replace(
    "_labelled", "", regex=False
)

# ------------------------------------------------------------
# 2. Rename columns (readable headers)
# ------------------------------------------------------------
latex_df = latex_df.rename(columns={
    "Config": "Model",
    "b_emb_correct_gen_wrong": "Emb correct / Gen wrong",
    "c_emb_wrong_gen_correct": "Emb wrong / Gen correct",
    "p_value": "McNemar p-value",
    "win": "Winner"
})

# ------------------------------------------------------------
# 3. Format numbers properly
# ------------------------------------------------------------

# round integer-like columns
int_cols = [
    "Emb correct / Gen wrong",
    "Emb wrong / Gen correct"
]
latex_df[int_cols] = latex_df[int_cols].astype(int)

# format p-values nicely (avoid 0.000000)
latex_df["McNemar p-value"] = latex_df["McNemar p-value"].apply(
    lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}"
)

# ------------------------------------------------------------
# 4. Export LaTeX
# ------------------------------------------------------------
latex_table = latex_df.to_latex(
    index=False,
    escape=False,
    column_format="lcccccc",
)

with open("mcnemar_summary.tex", "w") as f:
    f.write(latex_table)

print("Saved: mcnemar_summary.tex")
