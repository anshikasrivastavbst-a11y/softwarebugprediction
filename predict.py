"""
predict.py — CLI batch prediction using the stacking ensemble.

Usage:
    python predict.py <csv_file>

Output:
    predictions_output.csv  (original columns + Bug_Probability + Prediction + Status)
"""

import sys
import os
import pandas as pd
from inference import predict_dataframe

if len(sys.argv) < 2:
    print("Usage: python predict.py <csv_file>")
    sys.exit(1)

file_path = sys.argv[1]

if not os.path.isfile(file_path):
    print(f"Error: File not found — '{file_path}'")
    sys.exit(1)

df_original = pd.read_csv(file_path)

if df_original.empty:
    print("Error: The input CSV file is empty.")
    sys.exit(1)

print(f"Loaded {len(df_original)} rows from '{file_path}'")

predictions, probabilities = predict_dataframe(df_original)

output_df = df_original.copy()
output_df["Bug_Probability"] = probabilities.round(4)
output_df["Prediction"]      = predictions
output_df["Status"]          = output_df["Prediction"].map({1: "Buggy", 0: "Not Buggy"})

print(output_df[["Bug_Probability", "Prediction", "Status"]].head(10))

output_df.to_csv("predictions_output.csv", index=False)
print(f"\nSaved: predictions_output.csv  ({predictions.sum()} Buggy / {(predictions==0).sum()} Not Buggy)")
