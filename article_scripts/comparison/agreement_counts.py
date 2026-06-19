import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Load all generative runs
# ------------------------------------------------------------

base_path = Path("../generative/labelled_csvs")
files = list(base_path.glob("*.csv"))

all_runs = []

for file in files:

    df = pd.read_csv(file)

    if "Label_Gen" not in df.columns:
        continue

    df = df.dropna(subset=["Label_Gen"])

    df["Label_Gen"] = df["Label_Gen"].astype(int)

    df = df.set_index("UT (Unique WOS ID)")

    run_name = file.stem.replace("_labelled", "")

    all_runs.append(
        df[["Label_Gen"]].rename(
            columns={"Label_Gen": run_name}
        )
    )

# ------------------------------------------------------------
# Create agreement matrix
# ------------------------------------------------------------

agreement_df = pd.concat(all_runs, axis=1)

# keep only abstracts present in all runs
agreement_df = agreement_df.dropna()

n_runs = len(agreement_df.columns)

# ------------------------------------------------------------
# Agreement statistics
# ------------------------------------------------------------

agreement_df["True_Count"] = agreement_df.sum(axis=1)

agreement_df["False_Count"] = (
    n_runs - agreement_df["True_Count"]
)

agreement_df["Agreement_Rate"] = (
    agreement_df[["True_Count", "False_Count"]]
    .max(axis=1)
    / n_runs
)

# ------------------------------------------------------------
# Categorise
# ------------------------------------------------------------

def classify(row):

    t = row["True_Count"]

    if t == n_runs:
        return "All True"

    if t == 0:
        return "All False"

    if t > n_runs / 2:
        return "Majority True"

    if t < n_runs / 2:
        return "Majority False"

    return "Split"

agreement_df["Category"] = agreement_df.apply(
    classify,
    axis=1
)


agreement_df["Human"] = emb_df.loc[
    agreement_df.index,
    "Label_Human"
]
# agreement_df["Human"] = human_labels

# agreement_df["Unanimous"] = (
#     (agreement_df["True_Count"] == n_runs) |
#     (agreement_df["True_Count"] == 0)
# )
#
agreement_df["Consensus_Label"] = (
    agreement_df["True_Count"] > (n_runs / 2)
).astype(int)
agreement_df["Consensus_Agrees_With_Human"] = (
    agreement_df["Consensus_Label"] ==
    agreement_df["Human"]
)
disagreement_df = agreement_df[
    ~agreement_df["Consensus_Agrees_With_Human"]
].copy()
text_df = pd.read_csv(files[0])

text_cols = [
    "UT (Unique WOS ID)",
    "Article Title_x",
    "Abstract_x"
]

text_df = text_df[text_cols].set_index(
    "UT (Unique WOS ID)"
)

disagreement_df = disagreement_df.join(text_df)
disagreement_df.to_csv('disagree.csv')


agreement_df.to_csv('concensus.csv')

# ------------------------------------------------------------
# Summary counts
# ------------------------------------------------------------

summary = (
    agreement_df["Category"]
    .value_counts()
    .rename_axis("Category")
    .reset_index(name="Count")
)

print(summary)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

agreement_df.to_csv(
    "convention_agreement_per_abstract.csv"
)

summary.to_csv(
    "convention_agreement_summary.csv",
    index=False
)

print("\nSaved:")
print("  convention_agreement_per_abstract.csv")
print("  convention_agreement_summary.csv")
