import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

# Load test cases
df = pd.read_csv("stress_test_cases.csv")

total = len(df)
correct_sentiment = 0
correct_flag = 0

mismatches = []

for index, row in df.iterrows():
    payload = {
        "score": int(row["score"]),
        "feedback": row["feedback"]
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
    except Exception as e:
        print(f"Error at row {index}: {e}")
        continue

    predicted_sentiment = result.get("sentiment")
    predicted_flag = result.get("flag")

    expected_sentiment = row["expected_sentiment"]
    expected_flag = row["expected_flag"]

    sentiment_match = predicted_sentiment == expected_sentiment
    flag_match = predicted_flag == expected_flag

    if sentiment_match:
        correct_sentiment += 1

    if flag_match:
        correct_flag += 1

    if not sentiment_match or not flag_match:
        mismatches.append({
            "Row": index,
            "Score": row["score"],
            "Feedback": row["feedback"],
            "Expected Sentiment": expected_sentiment,
            "Predicted Sentiment": predicted_sentiment,
            "Expected Flag": expected_flag,
            "Predicted Flag": predicted_flag
        })

print("\n========== TEST RESULTS ==========")
print(f"Total Test Cases: {total}")
print(f"Sentiment Accuracy: {correct_sentiment}/{total} "
      f"({round(correct_sentiment/total*100,2)}%)")
print(f"Flag Accuracy: {correct_flag}/{total} "
      f"({round(correct_flag/total*100,2)}%)")

if mismatches:
    print("\n========== MISMATCHES ==========")
    for m in mismatches:
        print("\n----------------------------")
        for key, value in m.items():
            print(f"{key}: {value}")
else:
    print("\nAll test cases passed successfully 🎉")