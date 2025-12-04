import pandas as pd
import sys
import os

def calculate_averages(file_path):
    print(f"--- Analyzing: {os.path.basename(file_path)} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Filter Data
    # We only filter for Accuracy=1 because reaction time is typically undefined for incorrect answers.
    # No other outliers or edited trials are removed.
    df_clean = df[df['Accuracy'] == 1].copy()

    # 3. Group and Calculate Mean
    # We group by Format (digit/word) and Complexity (no/yes)
    averages = df_clean.groupby(['Format', 'Complexity'])['RT_ms'].mean().reset_index()
    
    # Round to 2 decimal places for readability
    averages['RT_ms'] = averages['RT_ms'].round(2)

    # 4. Print Results
    print("\nAverage Reaction Times (ms):")
    print("-" * 40)
    print(f"{'Format':<10} | {'Carry':<10} | {'Mean RT':<10}")
    print("-" * 40)
    
    for index, row in averages.iterrows():
        print(f"{row['Format']:<10} | {row['Complexity']:<10} | {row['RT_ms']:<10}")
    print("-" * 40)

    # Optional: Print total trials used
    print(f"\nTotal correct trials analyzed: {len(df_clean)} / {len(df)}")

if __name__ == "__main__":
    # Check if a file was dropped onto the script
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Drag and drop your CSV file here: ").strip().replace("'", "").replace('"', "")
    
    if os.path.exists(file_path):
        calculate_averages(file_path)
    else:
        print("File not found.")