# ============================================================
# Day 10 — Pandas Essentials for AI
# Zero → AI Engineer in 6 Months | Vinayak Gautam
# ============================================================

import pandas as pd
import numpy as np

# ============================================
# 1. CREATING DATAFRAMES
# ============================================
data = {
    'Name':   ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age':    [25, 30, 35, 28, 22],
    'Salary': [50000, 75000, 90000, 65000, 45000],
    'Dept':   ['AI', 'ML', 'AI', 'Data', 'ML'],
    'Score':  [88, 92, 85, 79, 95]
}

df = pd.DataFrame(data)
print("Full DataFrame:")
print(df)
print("\nShape:", df.shape)
print("Columns:", df.columns.tolist())
print("Dtypes:\n", df.dtypes)

# ============================================
# 2. EXPLORING DATA (first thing you do always)
# ============================================
print("\n--- First 3 rows ---")
print(df.head(3))

print("\n--- Stats Summary ---")
print(df.describe())

print("\n--- Missing values ---")
print(df.isnull().sum())

# ============================================
# 3. SELECTING DATA
# ============================================
print("\nNames:", df['Name'].tolist())

print("\nName + Salary:")
print(df[['Name', 'Salary']])

ai_team = df[df['Dept'] == 'AI']
print("\nAI Department only:")
print(ai_team)

high_earners = df[df['Salary'] > 60000]
print("\nHigh Earners (>60k):")
print(high_earners)

top = df[(df['Salary'] > 60000) & (df['Score'] > 85)]
print("\nHigh salary AND high score:")
print(top)

# ============================================
# 4. ADDING & MODIFYING COLUMNS
# ============================================
df['Bonus'] = df['Salary'] * 0.10
df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C')
df['Senior'] = df['Age'] > 28

print("\nDataFrame with new columns:")
print(df)

# ============================================
# 5. GROUPBY
# ============================================
print("\n--- Avg Salary by Department ---")
print(df.groupby('Dept')['Salary'].mean())

print("\n--- Multiple stats by Department ---")
print(df.groupby('Dept').agg({
    'Salary': ['mean', 'max'],
    'Score':  ['mean', 'min']
}))

# ============================================
# 6. SORTING
# ============================================
print("\n--- Top scorers ---")
print(df.sort_values('Score', ascending=False).head(3))

# ============================================
# 7. HANDLING MISSING DATA
# ============================================
df_messy = pd.DataFrame({
    'Name':   ['Alice', 'Bob', None, 'Diana'],
    'Age':    [25, None, 35, 28],
    'Salary': [50000, 75000, 90000, None]
})

print("\nMessy data:")
print(df_messy)
print("\nMissing values:\n", df_messy.isnull().sum())

df_messy['Age'].fillna(df_messy['Age'].mean(), inplace=True)
df_messy['Salary'].fillna(df_messy['Salary'].median(), inplace=True)
df_messy.dropna(inplace=True)

print("\nCleaned data:")
print(df_messy)
