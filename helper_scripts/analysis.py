# Script used to create graphs from testing data!

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

exclude_list = []

# Data analysis on combined_eval.csv

df = pd.read_csv("combined_eval.csv")

df["filename"] = df["filename"].str.replace(r'^eval_', '', regex=True)\
                               .str.replace(r'\.\.json$', '', regex=True)
df["model"] = df["filename"].str.extract(r'_(gemma3|llava|gpt-5-nano|gemini-2\.5-flash|allenai_olmOCR-7B-0225-preview|minicpm-v)_')
df["timestamp"] = df["filename"].str.extract(r'(\d{4}-\d{2}-\d{2}-T\d{6})')
df["filename"] = df["filename"].str.replace(
    r'_(gemma3|llava|gpt-5-nano|gemini-2\.5-flash|allenai_olmOCR-7B-0225-preview|minicpm-v)_?\d{4}-\d{2}-\d{2}-T\d{6}$',
    '',
    regex=True
)

doc_type_map = {
    "handwritten_story": "Handwritten",
    "02_handwritten_story": "Handwritten",
    "a_christmas_carol": "Long",
    "newspaper_extracts": "Multi-Column",
    "financial_times": "Multi-Column",
    "tom": "Multi-Column",
    "forerunner": "Short",
    "madman": "Short",
    "gitanjali": "Short"
}

df_clipped = df.copy()
df_clipped["WER"] = df_clipped["WER"].clip(upper=1.0) 
df_clipped["CER"] = df_clipped["CER"].clip(upper=1.0) 
df_clipped["doc_category"] = df_clipped["filename"].map(doc_type_map)
df_clipped = df_clipped[df_clipped["filename"] != "forerunner_llama3.2-vision_2025-10-15-T172040"]

category_means = (
    df_clipped.groupby(["doc_category", "model"], as_index=False)[["WER", "CER"]]
    .mean()
)

def categorize_performance(wer, cer):
    avg = (wer + cer) / 2
    if avg < 0.10:
        return "High Performing (<0.1)"
    elif avg < 0.20:
        return "Medium Performing (0.1–0.2)"
    else:
        return "Low Performing (>0.2)"

category_means["Category"] = category_means.apply(
    lambda row: categorize_performance(row["WER"], row["CER"]),
    axis=1
)

for doc_cat, subset in category_means.groupby("doc_category"):
    print(f"\nCATEGORY: {doc_cat.upper()}")
    
    for category, cat_subset in subset.groupby("Category"):
        n = len(cat_subset)
        models = ", ".join(cat_subset["model"])
        mean_wer = cat_subset["WER"].mean()
        mean_cer = cat_subset["CER"].mean()
        
        print(f"- {category} (n={n}): {models}")
        print(f"    Avg WER = {mean_wer:.2f}, CER = {mean_cer:.2f}")

df_long = df_clipped.melt(
    id_vars=["filename", "model"],
    value_vars=["WER", "CER", "match_percentage"],
    var_name="Metric",
    value_name="Error"
)

df_long.loc[df_long["Metric"] == "match_percentage", "Error"] = (
    1 - df_long.loc[df_long["Metric"] == "match_percentage", "Error"]
)



# Compute mean Error for each model & metric
model_means = (
    df_long
    .groupby(["model", "Metric"], as_index=False)["Error"]
    .mean()
)

# Pivot so each metric is a column
model_pivot = model_means.pivot(index="model", columns="Metric", values="Error").reset_index()

# Compute overall mean (simple average of all three metrics)
model_pivot["overall_mean"] = model_pivot[["WER", "CER", "match_percentage"]].mean(axis=1)

# Rank: lower is better (1 = best)
model_pivot["Rank"] = model_pivot["overall_mean"].rank(method="dense", ascending=True).astype(int)

# Reorder columns: Rank first
model_pivot = model_pivot[["Rank", "model", "WER", "CER", "match_percentage", "overall_mean"]]

# Round numeric columns to 2 decimal places
model_pivot = model_pivot.round(2)

# Sort by Rank
model_pivot = model_pivot.sort_values("Rank").reset_index(drop=True)

# Clean up column names (optional readability)
model_pivot = model_pivot.rename(columns={
    "model": "Model",
    "WER": "WER",
    "CER": "CER",
    "match_percentage": "Markdown Mismatch",
    "overall_mean": "Overall Mean"
})
print(model_pivot.to_string(index=False))
#print(model_pivot.to_latex())

def frange(start, stop, step):
    values = []
    while start <= stop:
        values.append(round(start, 3))  # round to avoid floating point errors
        start += step
    return values

plt.figure(figsize=(11, 7.5))

model_order = sorted(df_long["model"].unique())

# Barplot for mean Error per model and metric
ax = sns.barplot(
    data=df_long,
    x="model",
    y="Error",
    hue="Metric",
    errorbar='sd',
    palette="pastel",
    dodge=True,
    order=model_order
)

# Overlay individual points using swarmplot
sns.stripplot(
    data=df_long,
    x="model",
    y="Error",
    hue="Metric",
    dodge=True,
    size=6,
    alpha=0.7,
    palette=["gray"],
    ax=ax,
    legend=False,
    order=model_order
)

# Annotate the mean for each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 4,
        height + 0.02,
        f"{height:.2f}",
        ha='center',
        fontsize=6,
        fontweight='bold',
        color="purple"
    )
ax.set_yticks(frange(0, 1.05, 0.05))
# Adjust legend (remove duplicate entries from swarmplot)
handles, labels = ax.get_legend_handles_labels()
label_map = {
"WER": "Word Error Rate",
"CER": "Char Error Rate",
"match_percentage": "Markdown Feature Mismatch %"
}
ax.legend(handles, [label_map[l] for l in labels], title="Metrics")
plt.title(f"WER, CER and Markdown Feature Misatch by Model — All Files", fontsize=16)
plt.xlabel("Model")
plt.ylabel("Percentage")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"test_data/plots/all_files.png", dpi=300)

# PLOTS FOR INDIVIDUAL FILES!!

# Loop over each filename
for fname, subset in df_long.groupby("filename"):
    plt.figure(figsize=(11, 7.5))

    model_order = sorted(subset["model"].unique())
    
    # Barplot for mean Error per model and metric
    ax = sns.barplot(
        data=subset,
        x="model",
        y="Error",
        hue="Metric",
        errorbar='sd',
        palette="pastel",
        dodge=True,
        order=model_order
    )
    
    # Overlay individual points using swarmplot
    sns.stripplot(
        data=subset,
        x="model",
        y="Error",
        hue="Metric",
        dodge=True,
        size=6,
        alpha=0.7,
        palette=["gray"],
        ax=ax,
        legend=False,
        order=model_order
    )
    
    # Annotate the mean for each bar
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 4,
            height + 0.02,
            f"{height:.2f}",
            ha='center',
            fontsize=6,
            fontweight='bold',
            color="purple"
        )
    ax.set_yticks(frange(0, 1.05, 0.05))
    # Adjust legend (remove duplicate entries from swarmplot)
    handles, labels = ax.get_legend_handles_labels()
    label_map = {
    "WER": "Word Error Rate",
    "CER": "Char Error Rate",
    "match_percentage": "Markdown Feature Mismatch %"
    }
    ax.legend(handles, [label_map[l] for l in labels], title="Metrics")
    plt.title(f"WER, CER and Markdown Feature Misatch by Model — {fname}.pdf", fontsize=16)
    plt.xlabel("Model")
    plt.ylabel("Percentage")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"test_data/plots/{fname}.png", dpi=300)
    #plt.show()
