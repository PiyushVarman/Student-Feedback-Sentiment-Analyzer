import pandas as pd

df = pd.read_csv("../dataset/student_feedback.csv")

print(df.head())
print(df.columns)
print(df["sentiment"].value_counts() if "sentiment" in df.columns else "No sentiment column found")