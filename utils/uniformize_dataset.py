import pandas as pd

df = pd.read_csv("../datasets/articles/articles_annotated_sentences.csv")
df = df.drop(["Unnamed: 0"], axis=1)

initial_value_counts = list(df.label.value_counts())
min_entries_per_label = min(initial_value_counts)

for k in range(len(initial_value_counts)):
    df[df.label == k+1] = df[df.label == k+1][:min_entries_per_label]
    
df = df.dropna()
df = df.sample(frac=1).reset_index(drop=True)

print(df.label.value_counts())
df.to_csv("equal_annotated_sentences.csv")