import pandas as pd

df = pd.read_csv("../dataset/clean_feedback.csv")

# find rows containing "bad"
bad_rows = df[df["feedback"].str.lower().str.contains("bad", na=False)]

print("Total rows containing 'bad':", len(bad_rows))
print("\nSentiment distribution for rows containing 'bad':\n")
print(bad_rows["sentiment"].value_counts())

print("\nSample rows:\n")
print(bad_rows.sample(10))
