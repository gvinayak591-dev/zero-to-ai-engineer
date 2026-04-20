# day 12 mini project - job market analysis
# i wanted to analyze what factors affect hiring and salary in tech jobs
# used EDA techniques i learned today
# dataset is randomly generated but structure is realistic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 300

# building a job market dataset
jobs = pd.DataFrame({
    'job_title':    np.random.choice(['AI Engineer', 'ML Engineer', 'Data Scientist',
                                      'Data Analyst', 'DevOps Engineer'], n),
    'company_size': np.random.choice(['Startup', 'Mid-size', 'Enterprise'], n),
    'experience':   np.random.randint(0, 20, n),
    'salary':       np.random.randint(40000, 200000, n),
    'remote':       np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'skills':       np.random.randint(2, 15, n),
    'age':          np.random.randint(22, 55, n),
    'hired':        np.random.choice([0, 1], n, p=[0.65, 0.35])
})

# making salary more realistic
jobs.loc[jobs['job_title'] == 'AI Engineer',   'salary'] += 30000
jobs.loc[jobs['job_title'] == 'ML Engineer',   'salary'] += 20000
jobs.loc[jobs['company_size'] == 'Enterprise', 'salary'] += 15000
jobs.loc[jobs['experience'] > 10,              'salary'] += 20000

# inject missing values like real datasets have
jobs.loc[np.random.choice(n, 20), 'salary'] = np.nan
jobs.loc[np.random.choice(n, 15), 'skills'] = np.nan

# quick overview first
print("JOB MARKET ANALYSIS - Vinayak Gautam")
print("-" * 40)
print(f"total records    : {len(jobs)}")
print(f"missing values   : {jobs.isnull().sum().sum()}")
print(f"different roles  : {jobs['job_title'].nunique()}")
print(f"avg salary       : ${jobs['salary'].mean():,.0f}")
print(f"remote jobs      : {jobs['remote'].mean()*100:.0f}%")
print(f"overall hire rate: {jobs['hired'].mean()*100:.0f}%")

# fix missing values before analysis
# using = instead of inplace=True (newer pandas prefers this)
jobs['salary'] = jobs['salary'].fillna(jobs['salary'].median())
jobs['skills'] = jobs['skills'].fillna(jobs['skills'].median())

# big dashboard with 9 charts
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Job Market Analysis Dashboard - Vinayak Gautam',
             fontsize=13, fontweight='bold')

# chart 1 - which job title pays the most
ax1 = fig.add_subplot(3, 3, 1)
avg_by_role = jobs.groupby('job_title')['salary'].mean().sort_values()
ax1.barh(avg_by_role.index, avg_by_role.values, color='steelblue', alpha=0.85)
ax1.set_title('avg salary by role')
ax1.set_xlabel('salary ($)')

# chart 2 - salary distribution overall
ax2 = fig.add_subplot(3, 3, 2)
ax2.hist(jobs['salary'], bins=30, color='coral', edgecolor='black', alpha=0.8)
ax2.axvline(jobs['salary'].mean(), color='red', linestyle='--', label='mean')
ax2.set_title('salary distribution')
ax2.legend(fontsize=8)

# chart 3 - does more experience = more money?
ax3 = fig.add_subplot(3, 3, 3)
color_map = {'Startup': 'coral', 'Mid-size': 'steelblue', 'Enterprise': 'green'}
for size, group in jobs.groupby('company_size'):
    ax3.scatter(group['experience'], group['salary'],
                label=size, alpha=0.5, s=30, color=color_map[size])
ax3.set_title('experience vs salary')
ax3.set_xlabel('years of experience')
ax3.legend(fontsize=7)

# chart 4 - which role has best chance of getting hired
ax4 = fig.add_subplot(3, 3, 4)
hire_by_role = jobs.groupby('job_title')['hired'].mean() * 100
ax4.bar(hire_by_role.index, hire_by_role.values, color='green', alpha=0.8)
ax4.set_title('hire rate by role (%)')
ax4.tick_params(axis='x', rotation=30)

# chart 5 - remote vs onsite split
ax5 = fig.add_subplot(3, 3, 5)
ax5.pie(jobs['remote'].value_counts(),
        labels=['Remote', 'On-site'],
        autopct='%1.0f%%',
        colors=['#4CAF50', '#FF9800'],
        startangle=90)
ax5.set_title('remote vs on-site')

# chart 6 - does company size affect salary?
ax6 = fig.add_subplot(3, 3, 6)
sal_by_size = [jobs[jobs['company_size'] == s]['salary'].values
               for s in ['Startup', 'Mid-size', 'Enterprise']]
ax6.boxplot(sal_by_size, tick_labels=['Startup', 'Mid-size', 'Enterprise'],
            patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.6))
ax6.set_title('salary by company size')
ax6.grid(True, alpha=0.3)

# chart 7 - more skills = better hire rate?
ax7 = fig.add_subplot(3, 3, 7)
skill_level    = pd.cut(jobs['skills'], bins=[0, 5, 10, 15], labels=['Low', 'Mid', 'High'])
hire_by_skills = jobs.groupby(skill_level, observed=True)['hired'].mean() * 100
ax7.bar(hire_by_skills.index, hire_by_skills.values, color='purple', alpha=0.8)
ax7.set_title('hire rate by skill count')
ax7.set_ylabel('%')

# chart 8 - age distribution of applicants
ax8 = fig.add_subplot(3, 3, 8)
ax8.hist(jobs['age'], bins=20, color='teal', edgecolor='black', alpha=0.8)
ax8.set_title('age distribution')

# chart 9 - correlation heatmap
ax9 = fig.add_subplot(3, 3, 9)
cols_to_check = ['salary', 'experience', 'skills', 'age', 'hired', 'remote']
corr          = jobs[cols_to_check].corr()
img           = ax9.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)

ax9.set_xticks(range(len(cols_to_check)))
ax9.set_yticks(range(len(cols_to_check)))
ax9.set_xticklabels(cols_to_check, rotation=45, ha='right', fontsize=7)
ax9.set_yticklabels(cols_to_check, fontsize=7)

for i in range(len(cols_to_check)):
    for j in range(len(cols_to_check)):
        ax9.text(j, i, f'{corr.values[i,j]:.1f}',
                 ha='center', va='center', fontsize=7)

ax9.set_title('correlation heatmap')

plt.tight_layout()
plt.savefig('job_market_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# key findings
print("\nKEY FINDINGS")
print("-" * 40)
print(f"highest paying role  : {jobs.groupby('job_title')['salary'].mean().idxmax()}")
print(f"best company to join : {jobs.groupby('company_size')['salary'].mean().idxmax()}")
print(f"easiest role to get  : {jobs.groupby('job_title')['hired'].mean().idxmax()}")
print(f"salary vs experience : {jobs['salary'].corr(jobs['experience']):.2f} correlation")
print(f"skills vs hiring     : {jobs['skills'].corr(jobs['hired']):.2f} correlation")
print("\ndashboard saved as job_market_dashboard.png")
