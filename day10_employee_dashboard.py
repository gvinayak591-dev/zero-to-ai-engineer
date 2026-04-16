# ============================================================
# Day 10 — Mini Project: Employee Performance Dashboard
# Zero → AI Engineer in 6 Months | Vinayak Gautam
# ============================================================

import pandas as pd
import numpy as np

np.random.seed(42)

# Generate realistic employee dataset
departments = ['AI', 'ML', 'Data Engineering', 'DevOps']
names = ['Alice','Bob','Charlie','Diana','Eve','Frank',
         'Grace','Henry','Iris','Jack','Karen','Leo']

df = pd.DataFrame({
    'Name':       names,
    'Department': np.random.choice(departments, 12),
    'Age':        np.random.randint(22, 45, 12),
    'Salary':     np.random.randint(40000, 120000, 12),
    'Projects':   np.random.randint(1, 10, 12),
    'Score':      np.random.randint(60, 100, 12),
    'YearsExp':   np.random.randint(1, 15, 12)
})

print("=" * 55)
print("       EMPLOYEE PERFORMANCE DASHBOARD")
print("=" * 55)

# --- Overview ---
print(f"\n📋 Total Employees : {len(df)}")
print(f"💰 Avg Salary      : ${df['Salary'].mean():,.0f}")
print(f"⭐ Avg Score       : {df['Score'].mean():.1f}")
print(f"📁 Avg Projects    : {df['Projects'].mean():.1f}")

# --- Department summary ---
print("\n📊 Department Summary:")
dept_summary = df.groupby('Department').agg(
    Headcount  = ('Name', 'count'),
    Avg_Salary = ('Salary', 'mean'),
    Avg_Score  = ('Score', 'mean')
).round(1)
print(dept_summary)

# --- Performance rating ---
def rate(row):
    if row['Score'] >= 85 and row['Projects'] >= 6:
        return '🌟 Star'
    elif row['Score'] >= 75:
        return '✅ Good'
    else:
        return '⚠️  Needs Improvement'

df['Rating'] = df.apply(rate, axis=1)

# --- Top performers ---
print("\n🏆 Top Performers:")
stars = df[df['Rating'] == '🌟 Star'][['Name', 'Department', 'Score', 'Salary']]
print(stars.to_string(index=False))

# --- Salary insights ---
print("\n💡 Salary Insights:")
print(f"  Highest Paid : {df.loc[df['Salary'].idxmax(), 'Name']} (${df['Salary'].max():,})")
print(f"  Lowest Paid  : {df.loc[df['Salary'].idxmin(), 'Name']} (${df['Salary'].min():,})")

high_exp_low_pay = df[(df['YearsExp'] > 8) & (df['Salary'] < 70000)]
if len(high_exp_low_pay) > 0:
    print(f"\n⚠️  Underpaid veterans (exp>8yrs, salary<70k):")
    print(high_exp_low_pay[['Name', 'YearsExp', 'Salary']].to_string(index=False))

# --- Save report ---
df.to_csv('employee_report.csv', index=False)
print("\n✅ Report saved to employee_report.csv")
