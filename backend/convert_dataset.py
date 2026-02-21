import pandas as pd

# Load your dataset
df = pd.read_csv("../dataset/student_feedback.csv")

# Column pairs: (label_column, text_column)
pairs = [
    ("teaching", "teaching.1"),
    ("coursecontent", "coursecontent.1"),
    ("examination", "Examination"),
    ("labwork", "labwork.1"),
    ("library_facilities", " library_facilities"),
    ("extracurricular", "extracurricular.1")
]

all_data = []

for label_col, text_col in pairs:
    temp = df[[label_col, text_col]].copy()
    temp.columns = ["label", "feedback"]
    all_data.append(temp)

# Combine all into one dataset
final_df = pd.concat(all_data, ignore_index=True)

# Drop missing feedback
final_df.dropna(inplace=True)

# Convert numeric label to sentiment text
final_df["sentiment"] = final_df["label"].map({
    1: "positive",
    0: "neutral",
    -1: "negative"
})

# Drop the numeric label column
final_df = final_df.drop(columns=["label"])

# Remove empty feedback rows
final_df = final_df[final_df["feedback"].str.strip() != ""]

# Save cleaned dataset
final_df.to_csv("../dataset/clean_feedback.csv", index=False)

print("✅ Converted dataset saved as dataset/clean_feedback.csv")
print(final_df.head())
print(final_df["sentiment"].value_counts())
