# day 14 mini project - salary prediction model
# first time using scikit-learn (sklearn)
# sklearn is the go-to library for ML in python
# it has all major ML algorithms ready to use
# vinayak gautam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
n = 200

# ============================================================
# STEP 1 - create a realistic dataset
# ============================================================
# salary depends on experience, age, skills and has some randomness

experience = np.random.randint(0, 25, n)
age        = experience + np.random.randint(22, 30, n)
skills     = np.random.randint(3, 15, n)

# realistic salary formula with some noise
salary = (
    experience * 3500 +
    skills * 1200 +
    np.random.randint(-8000, 8000, n) +
    30000
)

df = pd.DataFrame({
    'experience': experience,
    'age':        age,
    'skills':     skills,
    'salary':     salary
})

print("SALARY PREDICTION MODEL - Vinayak Gautam")
print("-" * 45)
print(f"dataset size : {len(df)} employees")
print(f"avg salary   : ${df['salary'].mean():,.0f}")
print(f"salary range : ${df['salary'].min():,} to ${df['salary'].max():,}")
print("\nfirst 5 rows:")
print(df.head())


# ============================================================
# STEP 2 - split data into training and testing sets
# ============================================================
# this is CRUCIAL and something beginners often skip
#
# imagine studying for exam with 100 questions
# you practice on 80 questions (training set)
# then test yourself on 20 unseen questions (test set)
# if you only test on same questions you practiced - its cheating!
# same logic applies to ML models
#
# train_test_split does this automatically

X = df[['experience', 'age', 'skills']]  # features (input)
y = df['salary']                          # target (what we want to predict)

# X = features, y = target
# test_size=0.2 means 20% goes to testing, 80% for training
# random_state=42 makes the split same every time we run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\ndata split:")
print(f"  training set : {len(X_train)} rows (model learns from these)")
print(f"  testing set  : {len(X_test)} rows (model tested on these - never seen before)")


# ============================================================
# STEP 3 - train the model
# ============================================================
# sklearn makes this incredibly simple
# .fit() = train the model on training data
# internally it finds the best m and b values (like we did manually above)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nmodel trained!")
print(f"coefficients learned:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:12} : ${coef:,.0f} per unit")
print(f"  base salary  : ${model.intercept_:,.0f}")
print(f"\nmeaning:")
print(f"  every 1 yr of experience adds ${model.coef_[0]:,.0f} to salary")
print(f"  every 1 skill adds ${model.coef_[2]:,.0f} to salary")


# ============================================================
# STEP 4 - make predictions on test set
# ============================================================
# .predict() uses what model learned to predict on new unseen data

y_predicted = model.predict(X_test)

print(f"\nfirst 8 predictions vs actual:")
print(f"{'actual':>10}  {'predicted':>10}  {'error':>10}")
print("-" * 35)
for actual, pred in zip(y_test[:8], y_predicted[:8]):
    error = actual - pred
    print(f"${actual:>9,.0f}  ${pred:>9,.0f}  ${error:>+9,.0f}")


# ============================================================
# STEP 5 - evaluate the model
# ============================================================
# always evaluate on TEST SET (data model never saw during training)

mae  = mean_absolute_error(y_test, y_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
r2   = r2_score(y_test, y_predicted)

print(f"\nmodel performance on test set:")
print(f"  MAE  : ${mae:,.0f}  (avg error per prediction)")
print(f"  RMSE : ${rmse:,.0f}  (typical error size)")
print(f"  R²   : {r2:.3f}  (model explains {r2*100:.1f}% of salary variation)")


# ============================================================
# STEP 6 - visualize predictions vs actual
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Salary Prediction Model - Vinayak Gautam', fontweight='bold')

# chart 1 - actual vs predicted scatter
# perfect model = all points on the diagonal line
axes[0].scatter(y_test, y_predicted, alpha=0.6,
                color='steelblue', edgecolors='black', s=60)
min_val = min(y_test.min(), y_predicted.min())
max_val = max(y_test.max(), y_predicted.max())
axes[0].plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='perfect prediction line')
axes[0].set_xlabel('actual salary')
axes[0].set_ylabel('predicted salary')
axes[0].set_title('actual vs predicted\n(closer to red line = better)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# chart 2 - error distribution
# good model = errors centered around 0, no big outliers
residuals = y_test - y_predicted
axes[1].hist(residuals, bins=25, color='coral',
             edgecolor='black', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--',
                linewidth=2, label='zero error line')
axes[1].axvline(residuals.mean(), color='blue', linestyle='-',
                linewidth=2, label=f'avg error: ${residuals.mean():,.0f}')
axes[1].set_xlabel('prediction error ($)')
axes[1].set_ylabel('frequency')
axes[1].set_title('error distribution\n(good model = centered at 0)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('day14_salary_predictor.png', dpi=150)
plt.show()


# ============================================================
# STEP 7 - predict for a NEW person
# ============================================================
# the whole point of building a model - predict for new unknown data

print("\n" + "=" * 45)
print("PREDICTING FOR NEW PEOPLE")
print("=" * 45)

new_people = pd.DataFrame({
    'experience': [2,  8,  15],
    'age':        [25, 32, 42],
    'skills':     [4,  8,  12]
})

new_predictions = model.predict(new_people)

for i, (_, person) in enumerate(new_people.iterrows()):
    print(f"\nperson {i+1}:")
    print(f"  experience : {person['experience']} years")
    print(f"  age        : {person['age']}")
    print(f"  skills     : {person['skills']}")
    print(f"  predicted salary: ${new_predictions[i]:,.0f}")

print("\nmodel saved and dashboard exported as day14_salary_predictor.png")
