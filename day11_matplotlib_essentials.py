# ============================================================
# Day 11 — Matplotlib & Data Visualization Essentials
# Zero → AI Engineer in 6 Months | Vinayak Gautam
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. BASIC LINE PLOT
# ============================================
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, color='blue', linewidth=2, label='sin(x)')
plt.title('Basic Line Plot — Sin Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('day11_lineplot.png')
plt.show()
print("✅ Line plot saved!")

# ============================================
# 2. BAR CHART
# ============================================
departments = ['AI', 'ML', 'Data Eng', 'DevOps', 'Research']
salaries    = [95000, 88000, 82000, 78000, 91000]

plt.figure(figsize=(8, 5))
bars = plt.bar(departments, salaries, color=['#4CAF50','#2196F3','#FF9800','#E91E63','#9C27B0'])
plt.title('Average Salary by Department')
plt.xlabel('Department')
plt.ylabel('Salary ($)')
for bar, sal in zip(bars, salaries):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'${sal:,}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('day11_barchart.png')
plt.show()
print("✅ Bar chart saved!")

# ============================================
# 3. SCATTER PLOT (used in ML to visualize data)
# ============================================
np.random.seed(42)
experience = np.random.randint(1, 15, 50)
salary     = experience * 6000 + np.random.randint(-5000, 5000, 50)

plt.figure(figsize=(8, 5))
plt.scatter(experience, salary, color='coral', edgecolors='black', alpha=0.7, s=80)
plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day11_scatter.png')
plt.show()
print("✅ Scatter plot saved!")

# ============================================
# 4. HISTOGRAM (distribution — critical in ML)
# ============================================
scores = np.random.normal(75, 10, 200)

plt.figure(figsize=(8, 5))
plt.hist(scores, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
plt.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.1f}')
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('day11_histogram.png')
plt.show()
print("✅ Histogram saved!")

# ============================================
# 5. SUBPLOTS (multiple charts in one figure)
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Data Visualization Dashboard — Vinayak Gautam', fontsize=14, fontweight='bold')

# Plot 1 — Line
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), 'b-', label='sin')
axes[0, 0].plot(x, np.cos(x), 'r-', label='cos')
axes[0, 0].set_title('Sin & Cos Waves')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2 — Bar
axes[0, 1].bar(departments, salaries, color='steelblue')
axes[0, 1].set_title('Salary by Department')
axes[0, 1].tick_params(axis='x', rotation=15)

# Plot 3 — Scatter
axes[1, 0].scatter(experience, salary, color='coral', alpha=0.6)
axes[1, 0].set_title('Experience vs Salary')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4 — Histogram
axes[1, 1].hist(scores, bins=20, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Score Distribution')

plt.tight_layout()
plt.savefig('day11_dashboard.png')
plt.show()
print("✅ Dashboard saved!")
