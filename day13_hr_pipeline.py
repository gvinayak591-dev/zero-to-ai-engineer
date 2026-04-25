# day 13 mini project - full data cleaning pipeline
# imagine this is real HR data given to you by a company
# your job: clean it, engineer features, make it ML-ready
# vinayak gautam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500

# ============================================================
# THE RAW MESSY HR DATASET
# ============================================================
hr = pd.DataFrame({
    'emp_id':     range(1000, 1000 + n),
    'name':       ['Emp_' + str(i) for i in range(n)],
    'age':        np.random.randint(21, 62, n),
    'gender':     np.random.choice(['Male', 'Female', 'male', 'FEMALE', 'M', 'F'], n),
    'education':  np.random.choice(['Bachelors', 'Masters', 'PhD', 'bachelor', 'MASTERS'], n),
    'department': np.random.choice(['Tech', 'HR', 'Finance', 'tech', 'hr'], n),
    'salary':     np.random.randint(30000, 200000, n),
    'experience': np.random.randint(0, 35, n),
    'join_year':  np.random.randint(2005, 2024, n),
    'rating':     np.random.randint(1, 6, n),
    'left':       np.random.choice([0, 1], n, p=[0.75, 0.25])
})

# injecting real world problems
hr.loc[np.random.choice(n, 40), 'salary']     = np.nan
hr.loc[np.random.choice(n, 25), 'experience'] = np.nan
hr.loc[np.random.choice(n, 20), 'rating']     = np.nan
duplicates = hr.sample(15)
hr         = pd.concat([hr, duplicates], ignore_index=True)
hr.loc[3,  'salary'] = 5000000
hr.loc[7,  'age']    = 200
hr.loc[12, 'salary'] = -1000

print("=" * 55)
print("HR DATA CLEANING PIPELINE - Vinayak Gautam")
print("=" * 55)
print(f"\nRaw data shape: {hr.shape}")
print(f"Missing values: {hr.isnull().sum().sum()}")
print(f"Duplicates    : {hr.duplicated().sum()}")


# ============================================================
# PIPELINE STEP 1 - remove duplicates
# ============================================================
hr = hr.drop_duplicates(keep='first').reset_index(drop=True)
print(f"\nAfter removing duplicates: {len(hr)} rows")


# ============================================================
# PIPELINE STEP 2 - standardize text columns
# ============================================================
# Male, male, M all mean the same thing - standardize
hr['gender']     = hr['gender'].str.strip().str.title()
hr['education']  = hr['education'].str.strip().str.title()
hr['department'] = hr['department'].str.strip().str.title()

# now manually fix the short forms
hr['gender'] = hr['gender'].replace({'M': 'Male', 'F': 'Female'})

print("\nGender values after cleaning:", hr['gender'].unique())
print("Education values after cleaning:", hr['education'].unique())


# ============================================================
# PIPELINE STEP 3 - cap outliers
# ============================================================
def cap_col(df, col):
    q1   = df[col].quantile(0.25)
    q3   = df[col].quantile(0.75)
    iqr  = q3 - q1
    low  = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    before = ((df[col] < low) | (df[col] > high)).sum()
    df[col] = df[col].clip(low, high)
    print(f"  {col:12}: capped {before} outliers")
    return df

print("\nCapping outliers:")
hr = cap_col(hr, 'salary')
hr = cap_col(hr, 'age')
hr = cap_col(hr, 'experience')


# ============================================================
# PIPELINE STEP 4 - fill missing values
# ============================================================
hr['salary']     = hr['salary'].fillna(hr['salary'].median())
hr['experience'] = hr['experience'].fillna(hr['experience'].median())
hr['rating']     = hr['rating'].fillna(hr['rating'].mode()[0])
# mode()[0] = most common value, good for ratings/categories
# [0] because mode() can return multiple values, we take first one

print(f"\nMissing values after filling: {hr.isnull().sum().sum()}")


# ============================================================
# PIPELINE STEP 5 - feature engineering
# ============================================================
print("\nEngineering new features...")

# how long at company
hr['tenure'] = 2024 - hr['join_year']

# salary efficiency
hr['salary_per_year'] = hr['salary'] / (hr['tenure'] + 1)

# experience to age ratio - how much of their life is work experience
hr['exp_ratio'] = hr['experience'] / hr['age']

# total value score - combining multiple signals into one
# higher salary + higher rating + more experience = more valuable employee
hr['value_score'] = (
    (hr['salary'] / hr['salary'].max()) * 0.4 +    # 40% weight on salary
    (hr['rating'] / 5) * 0.3 +                      # 30% weight on rating
    (hr['experience'] / hr['experience'].max()) * 0.3  # 30% weight on experience
).round(3)

# risk flag - employees likely to leave based on low rating + low salary
avg_salary = hr['salary'].median()
hr['flight_risk'] = ((hr['rating'] <= 2) & (hr['salary'] < avg_salary)).astype(int)

# age bucket
hr['age_bucket'] = pd.cut(hr['age'],
    bins=[0, 28, 38, 50, 100],
    labels=['Junior', 'Mid', 'Senior', 'Executive'])

print("New features created: tenure, salary_per_year, exp_ratio, value_score, flight_risk, age_bucket")


# ============================================================
# PIPELINE STEP 6 - encode categorical columns
# ============================================================
# label encode department and education (ordinal - has order)
edu_order = {'Bachelors': 0, 'Masters': 1, 'Phd': 2}
hr['edu_encoded'] = hr['education'].map(edu_order)

# one hot encode gender (no order between male/female)
gender_dummies = pd.get_dummies(hr['gender'], prefix='gender')
hr = pd.concat([hr, gender_dummies], axis=1)


# ============================================================
# PIPELINE STEP 7 - normalize key numeric columns
# ============================================================
def minmax(col):
    return (col - col.min()) / (col.max() - col.min())

hr['salary_norm']     = minmax(hr['salary'])
hr['experience_norm'] = minmax(hr['experience'])
hr['rating_norm']     = minmax(hr['rating'])


# ============================================================
# VISUALIZATION - before vs after
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('HR Data Cleaning Pipeline Results - Vinayak Gautam',
             fontweight='bold', fontsize=13)

# salary distribution after cleaning
axes[0,0].hist(hr['salary'], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
axes[0,0].set_title('salary distribution (cleaned)')

# experience distribution
axes[0,1].hist(hr['experience'], bins=25, color='coral', edgecolor='black', alpha=0.8)
axes[0,1].set_title('experience distribution')

# value score distribution - our engineered feature
axes[0,2].hist(hr['value_score'], bins=25, color='green', edgecolor='black', alpha=0.8)
axes[0,2].set_title('value score (engineered feature)')

# salary by department
dept_sal = hr.groupby('department')['salary'].mean().sort_values()
axes[1,0].barh(dept_sal.index, dept_sal.values, color='purple', alpha=0.8)
axes[1,0].set_title('avg salary by department')

# flight risk by department
risk = hr.groupby('department')['flight_risk'].mean() * 100
axes[1,1].bar(risk.index, risk.values, color='red', alpha=0.8)
axes[1,1].set_title('flight risk % by dept')

# age bucket distribution
age_counts = hr['age_bucket'].value_counts()
axes[1,2].bar(age_counts.index, age_counts.values, color='teal', alpha=0.8)
axes[1,2].set_title('employees by age bucket')

plt.tight_layout()
plt.savefig('hr_cleaning_pipeline.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 55)
print("PIPELINE COMPLETE - Final Report")
print("=" * 55)
print(f"Total employees    : {len(hr)}")
print(f"Total features     : {len(hr.columns)}")
print(f"Missing values     : {hr.isnull().sum().sum()}")
print(f"Flight risk count  : {hr['flight_risk'].sum()} employees")
print(f"Avg value score    : {hr['value_score'].mean():.3f}")
print(f"\nTop 5 highest value employees:")
print(hr.nlargest(5, 'value_score')[['name','department','salary','rating','value_score']])
print("\ndashboard saved as hr_cleaning_pipeline.png")
