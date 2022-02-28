import pandas as pd

# df = pd.read_csv("datasets/articles/text_dataset.csv")

df = pd.read_csv("../datasets/articles/articles_annotated_sentences.csv")
print(df.head())

# split_border_1 = int(len(df)*0.7)
# split_border_2 = int(len(df)*0.85)

test_split = 1557
validation_split = 1891

train_df = df[df.article_id < test_split]
test_df = df[(df.article_id >= test_split) & (df.article_id < validation_split)]
validation_df = df[df.article_id >= validation_split]

print(train_df.shape)
print(test_df.shape)
print(validation_df.shape)

train_df.drop("Unnamed: 0", 1).to_csv("datasets/sentences/train_art_ann_sen.csv")
test_df.drop("Unnamed: 0", 1).to_csv("datasets/sentences/test_art_ann_sen.csv")
validation_df.drop("Unnamed: 0", 1).to_csv("datasets/sentences/validation_art_ann_sen.csv")

# train_df = df.loc[:split_border_1]
# test_df = df.loc[split_border_1:split_border_2].to_numpy()
# validation_df = df.loc[split_border_2:].to_numpy()

# print(test_df[0][0])
# print(validation_df[0][0])