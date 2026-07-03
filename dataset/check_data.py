import pandas as pd
import os

folder_path = "dataset"
total_buggy, total_clean = 0, 0

for file in sorted(os.listdir(folder_path)):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(folder_path, file))
        for col in ['bug','defects','label']:
            if col in df.columns:
                buggy = (df[col] > 0).sum()
                clean = (df[col] == 0).sum()
                total_buggy += buggy
                total_clean += clean
                print(f"{file:25s} → Buggy:{buggy:4d}  Clean:{clean:4d}  ({buggy/(buggy+clean):.0%} buggy)")
                break

total_samples = total_buggy + total_clean
print(f"\nTOTAL → Buggy:{total_buggy}  Clean:{total_clean}  ({total_buggy/total_samples:.0%} buggy)")