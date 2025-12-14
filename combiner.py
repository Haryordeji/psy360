import pandas as pd
from tkinter import Tk, filedialog

# Open file dialog
Tk().withdraw()  # hides the root window
csv_files = filedialog.askopenfilenames(
    title="Select CSV files",
    filetypes=[("CSV Files", "*.csv")]
)

df_list = [pd.read_csv(f) for f in csv_files]
merged = pd.concat(df_list, ignore_index=True)

cols = ["SubjectID", "ConditionCode", "Format", "Complexity",
        "N1", "N2", "CorrectAnswer", "UserAnswer",
        "Accuracy", "RT_ms", "WasEdited"]
merged = merged[cols]

merged.to_csv("merged_output.csv", index=False)
print("Merged", len(csv_files), "files.")