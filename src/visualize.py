import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../GiveMeSomeCredit/cs-training.csv', index_col=0)

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    ax = axes[i]
    if col == 'SeriousDlqin2yrs':
        df[col].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
        ax.set_title('Target: SeriousDlqin2yrs')
    else:
        # Plot capped at 99th percentile so outliers don't squash the chart
        cap = df[col].quantile(0.99)
        df[col].clip(upper=cap).plot(kind='hist', bins=50, ax=ax, color='steelblue', alpha=0.7)
        ax.set_title(col)
        ax.axvline(df[col].median(), color='red', linestyle='--', linewidth=1, label='median')
    ax.set_xlabel('')

# Hide unused subplot
axes[-1].set_visible(False)

plt.suptitle('Feature Distributions (capped at 99th percentile)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('./graphs/eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to eda_distributions.png")

# Also check: default rate by age bucket — usually very telling
df['age_bucket'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 120])
default_by_age = df.groupby('age_bucket', observed=True)['SeriousDlqin2yrs'].mean()

plt.figure(figsize=(8, 4))
default_by_age.plot(kind='bar', color='tomato', alpha=0.8)
plt.title('Default Rate by Age Group')
plt.ylabel('Default Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./graphs/default_by_age.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to default_by_age.png")

# Sentinel value check — these columns should max out around 10-15 realistically
sentinel_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse'
]
for col in sentinel_cols:
    high = df[col][df[col] >= 90]
    if len(high) > 0:
        print(f"{col}: {len(high)} rows with value >= 90 → {high.value_counts().to_dict()}")