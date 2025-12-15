import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import seaborn as sns


file_name = 'merged_output.csv' 

try:
    df = pd.read_csv(file_name)
    print(f"Successfully loaded data with {len(df)} trials.")
except FileNotFoundError:
    print(f"Error: Could not find {file_name}. Please check the filename.")
    exit()

# Create a combined 'Condition' column for easier plotting/grouping
# e.g., "Digit-NoCarry", "Word-Carry"
df['Condition'] = df['Format'].str.capitalize() + '-' + \
                  df['Complexity'].replace({'no': 'NoCarry', 'yes': 'Carry'})

# ==========================================
# 2. DESCRIPTIVE STATISTICS
# ==========================================
print("\n" + "="*40)
print("DESCRIPTIVE STATISTICS")
print("="*40)

# Calculate Accuracy & Editing Rates (using ALL trials)
stats_all = df.groupby(['Format', 'Complexity'])[['Accuracy', 'WasEdited']].mean().reset_index()
stats_all['ErrorRate'] = 1.0 - stats_all['Accuracy']

# Calculate Response Times (using CORRECT trials only)
df_correct = df[df['Accuracy'] == 1].copy()

def filter_outliers(group):
    mean, std = group.mean(), group.std()
    # Mask is True if value is within +/- 3 SDs
    mask = (group <= mean + 3*std) & (group >= mean - 3*std)
    # Return the values where mask is True, otherwise NaN
    return group.where(mask)

# Group by Subject AND Condition to respect individual baselines
df_correct['RT_Clean'] = df_correct.groupby(['SubjectID', 'Condition'])['RT_ms'].transform(filter_outliers)

# Drop the rows that became NaN (the outliers)
df_clean = df_correct.dropna(subset=['RT_Clean'])

# Print how many trials were removed
removed_count = len(df_correct) - len(df_clean)
print(f"Removed {removed_count} outliers (based on +/- 3 SD per participant per condition).")

rt_stats = df_clean.groupby(['Format', 'Complexity'])['RT_Clean'].agg(['mean', 'std', 'count']).reset_index()

# Merge tables
summary_table = pd.merge(stats_all, rt_stats, on=['Format', 'Complexity'])
summary_table = summary_table.sort_values(by='mean') # Sort by RT to check ordinality

print("\nSummary Table (Sorted by RT):")
print(summary_table[['Format', 'Complexity', 'mean', 'std', 'Accuracy', 'WasEdited']].round(3))
# print number of trials used for RT calculation
print("\nTrials used for RT calculation (correct, cleaned):")
print(rt_stats[['Format', 'Complexity', 'count']])

# ==========================================
# 3. HYPOTHESIS 1: ORDINAL COMPLEXITY
# ==========================================
print("\n" + "="*40)
print("HYPOTHESIS 1: ORDINAL COMPLEXITY CHECK")
print("="*40)

# Extract means for the specific order check
try:
    mu_d_nc = summary_table[(summary_table.Format=='digit') & (summary_table.Complexity=='no')]['mean'].values[0]
    mu_d_c  = summary_table[(summary_table.Format=='digit') & (summary_table.Complexity=='yes')]['mean'].values[0]
    mu_w_nc = summary_table[(summary_table.Format=='word')  & (summary_table.Complexity=='no')]['mean'].values[0]
    mu_w_c  = summary_table[(summary_table.Format=='word')  & (summary_table.Complexity=='yes')]['mean'].values[0]

    print(f"Digit-NoCarry ({mu_d_nc:.0f}) < Digit-Carry ({mu_d_c:.0f}) < Word-NoCarry ({mu_w_nc:.0f}) < Word-Carry ({mu_w_c:.0f})")
    
    if mu_d_nc < mu_d_c < mu_w_nc < mu_w_c:
        print(">> RESULT: Ordinal Prediction CONFIRMED.")
    else:
        print(">> RESULT: Ordinal Prediction NOT perfectly met (check values above).")
except IndexError:
    print("Error: Could not extract means. Check your data labels.")

# ==========================================
# 4. INFERENTIAL STATISTICS (ANOVA)
# ==========================================
print("\n" + "="*40)
print("INFERENTIAL STATISTICS (2x2 ANOVA on RT)")
print("="*40)

# We use the cleaned data (correct trials only)
# Aggregate mean RT per subject per condition for Repeated Measures ANOVA
df_subject_means = df_clean.groupby(['SubjectID', 'Format', 'Complexity'])['RT_Clean'].mean().reset_index()

anova = AnovaRM(data=df_subject_means, depvar='RT_Clean', subject='SubjectID', 
                within=['Format', 'Complexity']).fit()

print(anova)

# ==========================================
# 5. HYPOTHESIS 2: ACTIVATION DECAY (Errors)
# ==========================================
print("\n" + "="*40)
print("HYPOTHESIS 2: ACTIVATION DECAY (Error Analysis)")
print("="*40)

# Isolate incorrect trials
df_errors = df[df['Accuracy'] == 0].copy()

# Calculate Deviation (User Answer - Correct Answer)
df_errors['Deviation'] = df_errors['UserAnswer'] - df_errors['CorrectAnswer']

# Identify "Off-by-10" errors (evidence of carry decay)
# We look for deviations of +10 or -10
df_errors['IsTensError'] = df_errors['Deviation'].apply(lambda x: 1 if abs(x) == 10 else 0)

# Count errors by condition
error_analysis = df_errors.groupby(['Format', 'Complexity']).agg(
    TotalErrors=('UserAnswer', 'count'),
    TensErrors=('IsTensError', 'sum')
).reset_index()

# Calculate what % of the errors in that condition were Tens errors
error_analysis['Pct_TensErrors'] = (error_analysis['TensErrors'] / error_analysis['TotalErrors']) * 100

print("\nError Breakdown by Condition:")
print(error_analysis)

# Check if Word-Carry has the most errors
max_errors_row = error_analysis.loc[error_analysis['TotalErrors'].idxmax()]
print(f"\n>> Condition with most errors: {max_errors_row['Format']} - {max_errors_row['Complexity']} ({max_errors_row['TotalErrors']} errors)")

# Check Editing Rate
max_edit_row = summary_table.loc[summary_table['WasEdited'].idxmax()]
print(f">> Condition with highest editing rate: {max_edit_row['Format']} - {max_edit_row['Complexity']} ({(max_edit_row['WasEdited']*100):.1f}%)")

# ==========================================
# 6. VISUALIZATION (Optional)
# ==========================================
# Generates a plot similar to the ones usually found in CogSci papers
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Response Times
    sns.barplot(data=df_clean, x='Format', y='RT_Clean', hue='Complexity', capsize=.1, ax=ax[0], errorbar='se')
    ax[0].set_title('Response Times by Condition')
    ax[0].set_ylabel('Reaction Time (ms)')
    
    # Plot 2: Error Rates
    # We need to calculate error rate per subject for the bar plot to have error bars
    df_err_subj = df.groupby(['SubjectID', 'Format', 'Complexity'])['Accuracy'].mean().reset_index()
    df_err_subj['ErrorRate'] = 1 - df_err_subj['Accuracy']
    
    sns.barplot(data=df_err_subj, x='Format', y='ErrorRate', hue='Complexity', capsize=.1, ax=ax[1], errorbar='se')
    ax[1].set_title('Error Rates by Condition')
    ax[1].set_ylabel('Error Rate')

    plt.tight_layout()
    plt.savefig('results_plot.png')
    print("\n>> Plots saved as 'results_plot.png'")
except ImportError:
    print("\nSkipping plotting (seaborn/matplotlib not installed).")

print("\nAnalysis Complete.")