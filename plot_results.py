import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

# --- CONFIGURATION ---
# Define the Model Predictions (Total Model Units from your paper)
MODEL_PREDICTIONS = {
    ('digit', 'no'): 6,
    ('digit', 'yes'): 8,
    ('word', 'no'): 10,  # Based on your final decision (10 steps)
    ('word', 'yes'): 12  # Based on your final decision (12 steps)
}

def analyze_and_plot(file_path):
    print(f"--- Loading Data from: {file_path} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        return

    # 2. Data Cleaning
    initial_count = len(df)
    # Filter: Only Correct trials
    df_clean = df[df['Accuracy'] == 1].copy()
    # Filter: Reasonable RTs (e.g., between 200ms and 10000ms)
    df_clean = df_clean[(df_clean['RT_ms'] > 200) & (df_clean['RT_ms'] < 10000)]
    
    print(f"Trials kept: {len(df_clean)} / {initial_count}")
    print(f"Removed {initial_count - len(df_clean)} trials (incorrect or outliers).")

    # 3. Add Model Predictions to the Dataframe
    # Map the tuple (Format, Complexity) to the predicted unit count
    df_clean['ModelUnits'] = df_clean.apply(
        lambda x: MODEL_PREDICTIONS.get((x['Format'], x['Complexity']), 0), axis=1
    )

    # 4. Calculate Means per Condition
    condition_means = df_clean.groupby(['Format', 'Complexity'])['RT_ms'].mean().reset_index()
    # Add model units to this summary table for the scatter plot
    condition_means['ModelUnits'] = condition_means.apply(
        lambda x: MODEL_PREDICTIONS.get((x['Format'], x['Complexity']), 0), axis=1
    )

    print("\n--- Mean RT per Condition ---")
    print(condition_means)

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # FIGURE 1: Bar Chart (The Experimental Results)
    # Shows the interaction between Format and Complexity
    sns.barplot(
        data=df_clean, 
        x="Format", 
        y="RT_ms", 
        hue="Complexity", 
        errorbar='se', # Standard Error bars
        capsize=.1, 
        palette="viridis", 
        ax=axes[0]
    )
    axes[0].set_title("Impact of Format & Complexity on RT", fontsize=14)
    axes[0].set_ylabel("Reaction Time (ms)")
    axes[0].set_xlabel("Presentation Format")

    # FIGURE 2: Model Fit (ACT-R Validation)
    # Linear regression between Model Units (X) and Human RT (Y)
    
    # Calculate Linear Regression Stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        condition_means['ModelUnits'], condition_means['RT_ms']
    )
    
    sns.regplot(
        data=condition_means, 
        x="ModelUnits", 
        y="RT_ms", 
        color="darkblue", 
        ax=axes[1],
        marker="s", # Square markers
        scatter_kws={"s": 100} # Size of markers
    )
    
    # Annotate the plot with the R-squared value
    axes[1].text(
        0.05, 0.9, 
        f'$R^2 = {r_value**2:.3f}$\nSlope = {slope:.1f} ms/unit', 
        transform=axes[1].transAxes, 
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    axes[1].set_title("ACT-R Model Fit", fontsize=14)
    axes[1].set_xlabel("Predicted Procedural Units (ACT-R Steps)")
    axes[1].set_ylabel("Actual Mean RT (ms)")
    axes[1].set_xticks([6, 8, 10, 12]) # Ensure only our model steps are shown
    axes[1].set_xlim(5, 13)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Allow user to drop file in terminal or use default
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Prompt the user if no argument is given
        csv_path = input("Please enter the path to your CSV file (drag and drop here): ").strip().replace("'", "").replace('"', "")
    
    if os.path.exists(csv_path):
        analyze_and_plot(csv_path)
    else:
        print("Invalid file path.")