# day 15 - logistic regression and classification
# yesterday we predicted numbers (salary)
# today we predict categories (hired or not hired)
# this is called classification
# vinayak gautam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

np.random.seed(42)


# ============================================================
# PART 1 - why cant we use linear regression for this?
# ============================================================
# imagine predicting hired (1) or not hired (0)
# linear regression can give you 0.3, 1.5, -0.2
# those numbers make no sense for a yes/no question
# we need something that outputs between 0 and 1 only
# that's what logistic regression does


# ============================================================
# PART 2 - the sigmoid function (the magic of logistic regression)
# ============================================================
# logistic regression uses a special S-shaped curve called sigmoid
# sigmoid takes ANY number and squeezes it between 0 and 1
# formula: sigmoid(x) = 1 / (1 + e^(-x))
# output close to 1 = high probability = predict class 1
# output close to 0 = low probability  = predict class 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# lets see what sigmoid does to different numbers
test_values = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
print("how sigmoid squeezes any number between 0 and 1:")
print(f"{'input':>8}  {'sigmoid output':>15}  {'meaning':>20}")
print("-" * 50)
for val in test_values:
    sig = sigmoid(val)
    meaning = "very likely YES" if sig > 0.8 else \
              "likely YES"      if sig > 0.6 else \
              "uncertain"       if sig > 0.4 else \
              "likely NO"       if sig > 0.2 else "very likely NO"
    print(f"{val:>8}  {sig:>15.4f}  {meaning:>20}")

# plot the sigmoid curve
x = np.linspace(-10, 10, 200)
plt.figure(figsize=(9, 4))
plt.plot(x, sigmoid(x), color='steelblue', linewidth=3)
plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5,
            label='decision boundary (0.5)')
plt.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
plt.axhline(1, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
plt.fill_between(x, sigmoid(x), 0.5,
                 where=(sigmoid(x) > 0.5), alpha=0.15,
                 color='green', label='predict YES (1)')
plt.fill_between(x, sigmoid(x), 0.5,
                 where=(sigmoid(x) < 0.5), alpha=0.15,
                 color='red', label='predict NO (0)')
plt.title('sigmoid function - the heart of logistic regression\nvinayak gautam')
plt.xlabel('any input value')
plt.ylabel('probability (0 to 1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day15_sigmoid.png', dpi=150)
plt.show()
print("\nsigmoid plot saved!")


# ============================================================
# PART 3 - simple classification example
# ============================================================
# predicting if a student passes or fails based on study hours

n = 100
study_hours = np.random.uniform(0, 10, n)
# probability of passing increases with study hours
pass_prob   = sigmoid((study_hours - 5) * 1.2)
passed      = (np.random.rand(n) < pass_prob).astype(int)
# students who study more than 5 hours tend to pass

students = pd.DataFrame({
    'study_hours': study_hours.round(1),
    'passed':      passed
})

print("\n\nsimple example - predicting pass/fail from study hours")
print(f"total students : {len(students)}")
print(f"students passed: {students['passed'].sum()}")
print(f"students failed: {(students['passed'] == 0).sum()}")
print(f"pass rate      : {students['passed'].mean()*100:.1f}%")

# split and train
X_s = students[['study_hours']]
y_s = students['passed']

X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.2, random_state=42)

simple_model = LogisticRegression()
simple_model.fit(X_train, y_train)

# predict
y_pred = simple_model.predict(X_test)

# get probabilities not just 0/1
y_prob = simple_model.predict_proba(X_test)
# predict_proba gives [prob of 0, prob of 1] for each student

print("\nfirst 8 predictions with probabilities:")
print(f"{'hours':>7}  {'actual':>8}  {'predicted':>10}  {'prob pass':>10}")
print("-" * 45)
for i in range(8):
    hours  = X_test.iloc[i]['study_hours']
    actual = y_test.iloc[i]
    pred   = y_pred[i]
    prob   = y_prob[i][1]  # probability of passing (class 1)
    print(f"{hours:>7.1f}  {actual:>8}  {pred:>10}  {prob:>9.1%}")

print(f"\naccuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")


# ============================================================
# PART 4 - evaluation metrics for classification
# ============================================================
# accuracy alone is not enough - we need confusion matrix

print("\n\nCONFUSION MATRIX explained:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("""
what each number means:
        predicted NO    predicted YES
actual NO  [TN]           [FP]    ← model said YES but was actually NO (False Positive)
actual YES [FN]           [TP]    ← model said NO  but was actually YES (False Negative)

TN = True Negative  = correctly predicted NO
TP = True Positive  = correctly predicted YES
FP = False Positive = wrongly predicted YES (Type 1 error)
FN = False Negative = wrongly predicted NO  (Type 2 error)
""")

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives  (correctly said FAIL): {tn}")
print(f"True Positives  (correctly said PASS): {tp}")
print(f"False Positives (said PASS wrongly)  : {fp}")
print(f"False Negatives (said FAIL wrongly)  : {fn}")

print("\nfull classification report:")
print(classification_report(y_test, y_pred,
      target_names=['Failed', 'Passed']))
