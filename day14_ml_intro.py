# day 14 - introduction to machine learning
# today we build our FIRST actual ML model
# linear regression - the simplest and most important ML algorithm
# vinayak gautam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# WHAT IS LINEAR REGRESSION?
# ============================================================
# it tries to find a straight line that best fits your data
# that line can then PREDICT new values
#
# example: given years of experience → predict salary
# the model finds: salary = (experience * some_number) + some_base
# that formula is what it learns from data


# ============================================================
# PART 1 - understanding the concept with simple data
# ============================================================

# lets create a simple dataset
# salary increases with experience but not perfectly (real life has noise)
experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
salary     = np.array([35000, 42000, 50000, 55000, 62000,
                       70000, 75000, 82000, 90000, 98000])

print("our simple dataset:")
for exp, sal in zip(experience, salary):
    print(f"  {exp} years experience → ${sal:,} salary")


# ============================================================
# THE MATH BEHIND LINEAR REGRESSION
# ============================================================
# the line formula is: y = mx + b
# y = salary (what we want to predict)
# x = experience (what we know)
# m = slope (how much salary increases per year of experience)
# b = intercept (base salary at 0 experience)
#
# we need to find m and b that makes the line fit best
# "best fit" = minimizes the error between predicted and actual values

# calculating m and b manually (so you understand what sklearn does internally)
n        = len(experience)
mean_exp = experience.mean()
mean_sal = salary.mean()

# slope formula: m = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
numerator   = np.sum((experience - mean_exp) * (salary - mean_sal))
denominator = np.sum((experience - mean_exp) ** 2)
slope       = numerator / denominator

# intercept formula: b = mean_y - m * mean_x
intercept = mean_sal - slope * mean_exp

print(f"\nwhat the model learned:")
print(f"  slope (m)     = {slope:.2f}")
print(f"  intercept (b) = {intercept:.2f}")
print(f"  formula: salary = {slope:.0f} * experience + {intercept:.0f}")
print(f"\nmeaning: for every 1 year of experience, salary increases by ${slope:.0f}")


# ============================================================
# MAKING PREDICTIONS
# ============================================================
# now use the formula to predict salary for any experience value

def predict_salary(exp):
    return slope * exp + intercept

print("\npredictions using our model:")
for exp in [3, 7, 12, 15]:
    predicted = predict_salary(exp)
    print(f"  {exp} years experience → predicted salary: ${predicted:,.0f}")


# ============================================================
# VISUALIZING THE MODEL
# ============================================================
# plotting actual data points AND the line our model found

x_line = np.linspace(0, 12, 100)
y_line = slope * x_line + intercept

plt.figure(figsize=(9, 5))
plt.scatter(experience, salary, color='coral', s=100,
            edgecolors='black', zorder=5, label='actual data')
plt.plot(x_line, y_line, color='steelblue', linewidth=2.5,
         label=f'model line: salary = {slope:.0f}x + {intercept:.0f}')

# show predictions for specific points
for exp in [3, 7, 11]:
    pred = predict_salary(exp)
    plt.scatter(exp, pred, color='green', s=120,
                marker='*', zorder=6)
    plt.annotate(f'predict\n${pred:,.0f}',
                 xy=(exp, pred), xytext=(exp+0.3, pred-5000), fontsize=8)

plt.title('linear regression - experience vs salary\nvinayak gautam')
plt.xlabel('years of experience')
plt.ylabel('salary ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day14_linear_regression.png', dpi=150)
plt.show()
print("\nplot saved!")


# ============================================================
# PART 2 - measuring how good our model is
# ============================================================
# after building a model, we need to measure its performance
# we cant just "look" at it and say its good

predictions = predict_salary(experience)

# ERROR = difference between what model predicted vs actual value
errors = salary - predictions

print("\n\nmodel performance check:")
print("actual vs predicted:")
for i in range(len(experience)):
    print(f"  actual: ${salary[i]:,}  |  predicted: ${predictions[i]:,.0f}  |  error: ${errors[i]:,.0f}")


# ============================================================
# METRIC 1 - MAE (Mean Absolute Error)
# ============================================================
# average of all errors (ignoring negative/positive direction)
# easy to understand - "on average my model is off by $X"

mae = np.mean(np.abs(errors))
print(f"\nMAE (Mean Absolute Error)  = ${mae:,.0f}")
print(f"meaning: on average the model is wrong by ${mae:,.0f}")


# ============================================================
# METRIC 2 - MSE (Mean Squared Error)
# ============================================================
# squares the errors before averaging
# big errors are punished MORE because squaring amplifies them
# so model is forced to avoid large mistakes

mse = np.mean(errors ** 2)
print(f"\nMSE (Mean Squared Error)   = {mse:,.0f}")


# ============================================================
# METRIC 3 - RMSE (Root Mean Squared Error)
# ============================================================
# square root of MSE
# brings the error back to original units (dollars)
# more interpretable than MSE

rmse = np.sqrt(mse)
print(f"RMSE (Root Mean Sq Error)  = ${rmse:,.0f}")
print(f"meaning: typical error is around ${rmse:,.0f}")


# ============================================================
# METRIC 4 - R² Score (R-squared)
# ============================================================
# tells you HOW MUCH of the variation in y is explained by x
# range: 0 to 1
# 1.0 = perfect model, predicts everything correctly
# 0.0 = model is as bad as just predicting the average every time
# 0.9 = model explains 90% of variation - very good

ss_total   = np.sum((salary - salary.mean()) ** 2)
ss_residual = np.sum(errors ** 2)
r2          = 1 - (ss_residual / ss_total)

print(f"R² Score                   = {r2:.4f}")
print(f"meaning: model explains {r2*100:.1f}% of salary variation")

if r2 > 0.9:
    print("verdict: excellent model!")
elif r2 > 0.7:
    print("verdict: good model")
else:
    print("verdict: needs improvement")
