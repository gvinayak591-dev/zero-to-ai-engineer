# ============================================================
# Day 11 — Mini Project: Sales Performance Visual Dashboard
# Zero → AI Engineer in 6 Months | Vinayak Gautam
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

# ============================================
# DATA
# ============================================
months    = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sales     = [52000, 58000, 61000, 55000, 67000, 72000,
             69000, 74000, 78000, 82000, 88000, 95000]
target    = [60000] * 12
products  = ['Laptop', 'Phone', 'Tablet', 'Watch', 'Earbuds']
units     = [340, 520, 210, 430, 680]
regions   = ['North', 'South', 'East', 'West']
rev_share = [30, 25, 20, 25]

np.random.seed(42)
ad_spend  = np.random.randint(5000, 20000, 12)
revenue   = np.array(sales)

# ============================================
# DASHBOARD — 2x2 grid
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('📊 Annual Sales Performance Dashboard\nVinayak Gautam | Zero → AI Engineer',
             fontsize=13, fontweight='bold', y=1.01)

# --- Chart 1: Monthly Sales vs Target (Line) ---
ax1 = axes[0, 0]
ax1.plot(months, sales,  'o-', color='#2196F3', linewidth=2.5, markersize=6, label='Actual Sales')
ax1.plot(months, target, '--', color='#F44336', linewidth=2,   label='Target ($60k)')
ax1.fill_between(months, sales, target,
                 where=[s > t for s, t in zip(sales, target)],
                 alpha=0.15, color='green', label='Above Target')
ax1.set_title('Monthly Sales vs Target')
ax1.set_ylabel('Revenue ($)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# --- Chart 2: Product Units Sold (Bar) ---
ax2 = axes[0, 1]
colors = ['#4CAF50','#2196F3','#FF9800','#E91E63','#9C27B0']
bars   = ax2.bar(products, units, color=colors, edgecolor='black', alpha=0.85)
ax2.set_title('Units Sold by Product')
ax2.set_ylabel('Units')
for bar, unit in zip(bars, units):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(unit), ha='center', fontsize=9, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)

# --- Chart 3: Revenue by Region (Pie) ---
ax3 = axes[1, 0]
explode = (0.05, 0.05, 0.05, 0.05)
ax3.pie(rev_share, labels=regions, autopct='%1.1f%%',
        colors=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'],
        explode=explode, startangle=90)
ax3.set_title('Revenue Share by Region')

# --- Chart 4: Ad Spend vs Revenue (Scatter) ---
ax4 = axes[1, 1]
sc = ax4.scatter(ad_spend, revenue, c=range(12), cmap='RdYlGn',
                 s=100, edgecolors='black', alpha=0.8)
for i, month in enumerate(months):
    ax4.annotate(month, (ad_spend[i], revenue[i]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)
ax4.set_title('Ad Spend vs Revenue')
ax4.set_xlabel('Ad Spend ($)')
ax4.set_ylabel('Revenue ($)')
ax4.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax4, label='Month (1=Jan)')

plt.tight_layout()
plt.savefig('day11_sales_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# SUMMARY STATS
# ============================================
print("=" * 45)
print("   SALES DASHBOARD SUMMARY — Vinayak Gautam")
print("=" * 45)
print(f"💰 Total Revenue    : ${sum(sales):,}")
print(f"📈 Best Month       : {months[sales.index(max(sales))]} (${max(sales):,})")
print(f"📉 Worst Month      : {months[sales.index(min(sales))]} (${min(sales):,})")
print(f"🎯 Months Hit Target: {sum(s >= 60000 for s in sales)}/12")
print(f"📦 Top Product      : {products[units.index(max(units))]} ({max(units)} units)")
print(f"🌍 Top Region       : {regions[rev_share.index(max(rev_share))]}")
print("\n✅ Dashboard saved as day11_sales_dashboard.png")
