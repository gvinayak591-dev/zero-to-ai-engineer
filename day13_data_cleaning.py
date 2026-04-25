# day 13 - data cleaning and feature engineering
# this is probably the most important skill in real ML projects
# 80% of an ML engineer's time is spent here, not on the model itself
# vinayak gautam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 300

# creating a messy real-world style dataset
# deliberately adding problems we need to fix
raw = pd.DataFrame({
    'name':       ['Person_' + str(i) for i in range(n)],
    'age':        np.random.randint(20, 65, n),
    'salary':     np.random.randint(25000, 180000, n),
    'experience': np.random.randint(0, 40, n),
    'department': np.random.choice(['AI', 'ML', 'data', 'Data', 'DEVOPS', 'devops'], n),
    'join_year':  np.random.randint(2010, 2024, n),
    'score':      np.random.randint(40, 100, n),
    'promoted':   np.random.choice([0, 1], n, p=[0.65, 0.35])
})

# ============================================================
# PROBLEM 1 - injecting missing values
# ============================================================
# in real data, certain fields are often blank
raw.loc[np.random.choice(n, 30), 'salary']     = np.nan
raw.loc[np.random.choice(n, 20), 'experience'] = np.nan
raw.loc[np.random.choice(n, 15), 'score']      = np.nan

# ============================================================
# PROBLEM 2 - injecting duplicate rows
# ============================================================
# real datasets often have the same record entered twice
duplicates = raw.sample(10)
raw        = pd.concat([raw, duplicates], ignore_index=True)

# ============================================================
# PROBLEM 3 - injecting outliers
# ============================================================
# some salary values are clearly wrong - typos or errors
raw.loc[5,  'salary'] = 9999999   # someone accidentally added extra zeros
raw.loc[10, 'salary'] = -5000     # negative salary makes no sense
raw.loc[15, 'age']    = 150       # age 150 is impossible

print("=" * 55)
print("RAW DATA PROBLEMS - before cleaning")
print("=" * 55)
print(f"total rows (with duplicates) : {len(raw)}")
print(f"missing values               : {raw.isnull().sum().sum()}")
print(f"duplicate rows               : {raw.duplicated().sum()}")
print(f"\ndepartment unique values (messy):")
print(raw['department'].value_counts())


# ============================================================
# CLEANING STEP 1 - remove duplicate rows
# ============================================================
# .duplicated() returns True for rows that are exact copies
# .drop_duplicates() removes those rows
# keep='first' means keep the first occurrence, remove the rest

print("\n\nSTEP 1 - removing duplicates")
print(f"rows before : {len(raw)}")
cleaned = raw.drop_duplicates(keep='first')
cleaned = cleaned.reset_index(drop=True)
# reset_index because after dropping rows the index has gaps like 0,1,5,6...
# drop=True means dont add the old index as a new column
print(f"rows after  : {len(cleaned)}")
print(f"removed     : {len(raw) - len(cleaned)} rows")


# ============================================================
# CLEANING STEP 2 - fix inconsistent text values
# ============================================================
# 'data', 'Data', 'DEVOPS', 'devops' are all the same thing
# we need to standardize them

print("\n\nSTEP 2 - fixing inconsistent department names")
print("before:", cleaned['department'].unique())

# .str.strip() removes any extra spaces before/after
# .str.title() converts to Title Case - first letter capital
cleaned['department'] = cleaned['department'].str.strip().str.title()

print("after :", cleaned['department'].unique())


# ============================================================
# CLEANING STEP 3 - handle outliers
# ============================================================
# using IQR method we learned in day 12
# but this time instead of removing, we CAP the values
# capping = replacing outlier with the boundary value
# removing outliers loses data, capping keeps the row

print("\n\nSTEP 3 - handling outliers")

def cap_outliers(df, col):
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    low  = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    # count outliers before capping
    outlier_count = ((df[col] < low) | (df[col] > high)).sum()

    # clip replaces anything below low with low, anything above high with high
    df[col] = df[col].clip(lower=low, upper=high)
    print(f"  {col:12} - capped {outlier_count} outliers | range set to [{low:.0f}, {high:.0f}]")
    return df

cleaned = cap_outliers(cleaned, 'salary')
cleaned = cap_outliers(cleaned, 'age')
cleaned = cap_outliers(cleaned, 'experience')


# ============================================================
# CLEANING STEP 4 - handle missing values smartly
# ============================================================
# not all missing values should be handled the same way
# numeric columns - fill with median (safer than mean)
# we already know why - outliers affect mean but not median

print("\n\nSTEP 4 - filling missing values")
print("missing before:")
print(cleaned.isnull().sum()[cleaned.isnull().sum() > 0])

cleaned['salary']     = cleaned['salary'].fillna(cleaned['salary'].median())
cleaned['experience'] = cleaned['experience'].fillna(cleaned['experience'].median())
cleaned['score']      = cleaned['score'].fillna(cleaned['score'].median())

print("\nmissing after:", cleaned.isnull().sum().sum())


# ============================================================
# FEATURE ENGINEERING - creating new useful columns
# ============================================================
# this is where the magic happens
# we create NEW information from existing columns
# that helps the model learn patterns better

print("\n\nFEATURE ENGINEERING - creating new columns")

# feature 1 - salary per year of experience
# this tells us how well someone is paid relative to their experience
# a person with 10 years exp earning 50k is underpaid
# a person with 2 years exp earning 50k is well paid
# raw salary alone doesnt tell you this
cleaned['salary_per_exp'] = cleaned['salary'] / (cleaned['experience'] + 1)
# +1 to avoid dividing by zero when experience = 0

# feature 2 - years since joining
# instead of raw join year, how long have they been here is more useful
cleaned['years_at_company'] = 2024 - cleaned['join_year']

# feature 3 - age group category
# sometimes grouping ages is better than raw age numbers
# pd.cut divides a continuous number into labeled buckets
cleaned['age_group'] = pd.cut(
    cleaned['age'],
    bins=[0, 30, 40, 50, 100],
    labels=['Young', 'Mid', 'Senior', 'Veteran']
)
# bins=[0,30,40,50,100] means:
# 0-30  = Young
# 30-40 = Mid
# 40-50 = Senior
# 50+   = Veteran

# feature 4 - performance category
# converting score number into a category
cleaned['performance'] = pd.cut(
    cleaned['score'],
    bins=[0, 60, 75, 90, 100],
    labels=['Poor', 'Average', 'Good', 'Excellent']
)

# feature 5 - is high earner flag
# simple binary feature - 1 if earning above median, 0 if below
salary_median = cleaned['salary'].median()
cleaned['is_high_earner'] = (cleaned['salary'] > salary_median).astype(int)
# .astype(int) converts True/False to 1/0

print("new columns added:")
print(cleaned[['salary_per_exp', 'years_at_company',
               'age_group', 'performance', 'is_high_earner']].head(8))


# ============================================================
# ENCODING - converting text categories to numbers
# ============================================================
# ML models only understand numbers, not text
# so 'AI', 'ML', 'Data' needs to become 0, 1, 2

print("\n\nENCODING text columns to numbers")

# method 1 - label encoding (manual)
# just assign a number to each category
dept_map = {'Ai': 0, 'Ml': 1, 'Data': 2, 'Devops': 3}
cleaned['dept_encoded'] = cleaned['department'].map(dept_map)
# .map() replaces each value using the dictionary

# method 2 - one hot encoding
# creates a separate 0/1 column for each category
# this is better when categories have no order
# AI   ML   Data   DevOps
#  1    0    0      0      ← this row is AI
#  0    1    0      0      ← this row is ML
dept_dummies = pd.get_dummies(cleaned['department'], prefix='dept')
cleaned = pd.concat([cleaned, dept_dummies], axis=1)
# prefix='dept' means columns will be named dept_Ai, dept_Ml etc
# pd.concat with axis=1 adds columns side by side

print("after encoding:")
print(cleaned[['department', 'dept_encoded',
               'dept_Ai', 'dept_Ml', 'dept_Data', 'dept_Devops']].head(6))


# ============================================================
# NORMALIZATION - scaling numbers to same range
# ============================================================
# salary is 25000-180000
# experience is 0-40
# score is 40-100
# these are VERY different scales
# ML models get confused when one feature has huge values vs tiny values
# normalization brings everything to 0-1 range

print("\n\nNORMALIZATION - scaling features to 0-1 range")
print("before scaling:")
print(f"  salary range     : {cleaned['salary'].min():.0f} to {cleaned['salary'].max():.0f}")
print(f"  experience range : {cleaned['experience'].min():.0f} to {cleaned['experience'].max():.0f}")

def normalize(col):
    # min-max normalization formula
    # (value - minimum) / (maximum - minimum)
    # this squeezes everything between 0 and 1
    return (col - col.min()) / (col.max() - col.min())

cleaned['salary_scaled']     = normalize(cleaned['salary'])
cleaned['experience_scaled'] = normalize(cleaned['experience'])
cleaned['score_scaled']      = normalize(cleaned['score'])

print("\nafter scaling:")
print(f"  salary range     : {cleaned['salary_scaled'].min():.2f} to {cleaned['salary_scaled'].max():.2f}")
print(f"  experience range : {cleaned['experience_scaled'].min():.2f} to {cleaned['experience_scaled'].max():.2f}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n\n" + "=" * 55)
print("CLEANING COMPLETE - final dataset summary")
print("=" * 55)
print(f"total rows      : {len(cleaned)}")
print(f"total columns   : {len(cleaned.columns)}")
print(f"missing values  : {cleaned.isnull().sum().sum()}")
print(f"original columns: 8")
print(f"new columns made: {len(cleaned.columns) - 8}")
print("\nall columns:")
for col in cleaned.columns:
    print(f"  {col}")
