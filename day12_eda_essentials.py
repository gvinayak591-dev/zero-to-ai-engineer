# day 12 - learning EDA today
# EDA = Exploratory Data Analysis
# basically before making any ML model you gotta understand your data first
# lets gooo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200

# making a fake employee dataset to practice on
mydata = pd.DataFrame({
    'age':        np.random.randint(22, 60, n),
    'salary':     np.random.randint(30000, 150000, n),
    'experience': np.random.randint(0, 30, n),
    'department': np.random.choice(['AI', 'ML', 'Data', 'DevOps'], n),
    'score':      np.random.randint(50, 100, n),
    'promoted':   np.random.choice([0, 1], n, p=[0.7, 0.3])
})

# adding some missing values manually bcz real world data always has this problem
mydata.loc[np.random.choice(n, 15), 'salary']     = np.nan
mydata.loc[np.random.choice(n, 10), 'experience'] = np.nan

# STEP 1 - first look at the data
# always do this when u get any new dataset
print("how many rows and columns do we have")
print(mydata.shape)

print("\nwhat are the column names")
print(mydata.columns.tolist())

print("\nfirst 5 rows")
print(mydata.head())

print("\nwhat datatype is each column")
print(mydata.dtypes)

# STEP 2 - check for missing values
# this is super important, missing values break your model silently
print("\n\nchecking missing values now")
print(mydata.isnull().sum())

# percentage missing
pct_missing = (mydata.isnull().sum() / len(mydata) * 100).round(1)
print("\nmissing percentage per column")
print(pct_missing)

# filling the missing ones
# using median for salary bcz salary data usually has outliers
# median is better than mean when outliers exist
# note: newer pandas doesnt like inplace=True so using = assignment instead
mydata['salary']     = mydata['salary'].fillna(mydata['salary'].median())
mydata['experience'] = mydata['experience'].fillna(mydata['experience'].mean())

print("\nafter filling - any missing left?", mydata.isnull().sum().sum())

# STEP 3 - basic stats
# describe() gives count mean std min max and quartiles all at once
print("\n\nbasic statistics of the data")
print(mydata.describe().round(1))

# STEP 4 - lets plot distributions to see how data looks
# histogram = shows how values are spread out
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('how is the data distributed - vinayak gautam', fontweight='bold')

# salary distribution
axes[0,0].hist(mydata['salary'], bins=25, color='steelblue', edgecolor='black', alpha=0.8)
axes[0,0].axvline(mydata['salary'].mean(), color='red', linestyle='--', label='avg salary')
axes[0,0].set_title('salary distribution')
axes[0,0].legend()

# age distribution
axes[0,1].hist(mydata['age'], bins=20, color='coral', edgecolor='black', alpha=0.8)
axes[0,1].set_title('age distribution')

# score distribution
axes[1,0].hist(mydata['score'], bins=20, color='green', edgecolor='black', alpha=0.8)
axes[1,0].set_title('score distribution')

# experience distribution
axes[1,1].hist(mydata['experience'], bins=20, color='purple', edgecolor='black', alpha=0.8)
axes[1,1].set_title('experience distribution')

plt.tight_layout()
plt.savefig('distributions.png', dpi=150)
plt.show()
print("saved distribution charts!")

# STEP 5 - outlier detection using IQR method
# IQR = difference between 75th and 25th percentile
# anything outside 1.5 * IQR from Q1 or Q3 is an outlier
print("\n\nchecking for outliers")

def find_outliers(col):
    q1  = mydata[col].quantile(0.25)
    q3  = mydata[col].quantile(0.75)
    iqr = q3 - q1
    low  = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    bad_ones = mydata[(mydata[col] < low) | (mydata[col] > high)]
    print(f"{col} has {len(bad_ones)} outliers | safe range: {low:.0f} to {high:.0f}")

for col in ['salary', 'age', 'score', 'experience']:
    find_outliers(col)

# box plot shows outliers visually
# the dots outside the whiskers = outliers
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle('outlier detection using box plots - vinayak gautam', fontweight='bold')

for i, col in enumerate(['salary', 'age', 'score', 'experience']):
    axes[i].boxplot(mydata[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[i].set_title(col)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boxplots.png', dpi=150)
plt.show()
print("saved box plots!")

# STEP 6 - correlation
# correlation tells us how much two columns are related
# +1 = move together, -1 = move opposite, 0 = no relation
print("\n\ncorrelation between all numeric columns")
only_numbers = mydata.select_dtypes(include=[np.number])
corr_table   = only_numbers.corr().round(2)
print(corr_table)

# making a heatmap to visualize the correlation matrix
fig, ax = plt.subplots(figsize=(7, 6))
cols = corr_table.columns.tolist()
vals = corr_table.values

img = ax.imshow(vals, cmap='RdYlGn', vmin=-1, vmax=1)

ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticklabels(cols)

# write the actual number inside each cell
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{vals[i,j]:.2f}', ha='center', va='center', fontsize=9)

plt.colorbar(img)
ax.set_title('correlation heatmap - vinayak gautam')
plt.tight_layout()
plt.savefig('correlation.png', dpi=150)
plt.show()
print("saved heatmap!")

# STEP 7 - categorical stuff
# how many people in each dept and their promotion rates
print("\n\nhow many people per department")
print(mydata['department'].value_counts())

print("\npromotion rate per department (0 = no, 1 = yes)")
print(mydata.groupby('department')['promoted'].mean().round(2))
