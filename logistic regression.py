# day 15 mini project - job hiring prediction model
# predict whether a candidate will get hired or not
# based on their skills, experience, score and other factors
# this is a real world classification problem
# vinayak gautam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 500

# ============================================================
# STEP 1 - create a realistic hiring dataset
# ============================================================
experience = np.random.randint(0, 15, n)
skills     = np.random.randint(2, 15, n)
score      = np.random.randint(40, 100, n)
age        = experience + np.random.randint(22, 28, n)
projects   = np.random.randint(0, 10, n)

# hiring probability based on realistic logic
# more skills, higher score, more projects = better chance
hire_score = (
    experience * 0.15 +
    skills     * 0.25 +
    score      * 0.05 +
    projects   * 0.20 +
    np.random.randn(n) * 0.5  # some randomness
)

# convert to binary hired/not hired using sigmoid
hire_prob = 1 / (1 + np.exp(-hire_score + 3))
hired     = (np.random.rand(n) < hire_prob).astype(int)

df = pd.DataFrame({
    'experience': experience,
    'skills':     skills,
    'score':      score,
    'age':        age,
    'projects':   projects,
    'hired':      hired
})

print("HIRING PREDICTION MODEL - Vinayak Gautam")
print("-" * 45)
print(f"total candidates : {len(df)}")
print(f"hired            : {df['hired'].sum()} ({df['hired'].mean()*100:.1f}%)")
print(f"not hired        : {(df['hired']==0).sum()} ({(df['hired']==0).mean()*100:.1f}%)")
print("\nfirst 5 rows:")
print(df.head())


# ============================================================
# STEP 2 - prepare features and target
# ============================================================
X = df[['experience', 'skills', 'score', 'age', 'projects']]
y = df['hired']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\ntraining set : {len(X_train)} candidates")
print(f"testing set  : {len(X_test)} candidates")


# ============================================================
# STEP 3 - scale features
# ============================================================
# logistic regression works better when all features
# are on similar scale
# StandardScaler makes mean=0 and std=1 for each column
# this is different from min-max we did before

scaler   = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# important: fit on training data ONLY
# then transform both train and test
# never fit on test data - that would be data leakage!


# ============================================================
# STEP 4 - train the model
# ============================================================
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
print("\nmodel trained!")


# ============================================================
# STEP 5 - predictions and probabilities
# ============================================================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
# [:, 1] = take only the probability of being hired (class 1)
# model gives [prob_not_hired, prob_hired] for each person


# ============================================================
# STEP 6 - evaluate
# ============================================================
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nMODEL PERFORMANCE")
print("-" * 35)
print(f"accuracy         : {acc*100:.1f}%")
print(f"true positives   : {tp}  (correctly predicted hired)")
print(f"true negatives   : {tn}  (correctly predicted not hired)")
print(f"false positives  : {fp}  (said hired but actually not)")
print(f"false negatives  : {fn}  (said not hired but actually hired)")

# precision = of all people model said HIRED, how many actually got hired?
precision = tp / (tp + fp)
# recall = of all people who actually got HIRED, how many did model catch?
recall    = tp / (tp + fn)
# f1 = balance between precision and recall
f1        = 2 * (precision * recall) / (precision + recall)

print(f"\nprecision : {precision:.3f} (when model says hired, it's right {precision*100:.1f}% of the time)")
print(f"recall    : {recall:.3f}    (model catches {recall*100:.1f}% of all hired candidates)")
print(f"f1 score  : {f1:.3f}       (balance between precision and recall)")

print("\nfull report:")
print(classification_report(y_test, y_pred,
      target_names=['Not Hired', 'Hired']))


# ============================================================
# STEP 7 - visualizations
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Hiring Prediction Model - Vinayak Gautam',
             fontweight='bold', fontsize=13)

# chart 1 - confusion matrix heatmap
cm_display = axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Not Hired', 'Hired'])
axes[0].set_yticklabels(['Not Hired', 'Hired'])
axes[0].set_xlabel('predicted')
axes[0].set_ylabel('actual')
axes[0].set_title('confusion matrix')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i,j], ha='center',
                    va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')

# chart 2 - probability distribution
# good model = hired candidates cluster near 1.0
#              not hired candidates cluster near 0.0
hired_probs     = y_prob[y_test == 1]
not_hired_probs = y_prob[y_test == 0]
axes[1].hist(not_hired_probs, bins=20, alpha=0.7,
             color='red', label='not hired', edgecolor='black')
axes[1].hist(hired_probs, bins=20, alpha=0.7,
             color='green', label='hired', edgecolor='black')
axes[1].axvline(0.5, color='black', linestyle='--',
                linewidth=2, label='decision boundary')
axes[1].set_title('predicted probability distribution')
axes[1].set_xlabel('probability of being hired')
axes[1].set_ylabel('count')
axes[1].legend()

# chart 3 - ROC curve
# shows model performance at different thresholds
# AUC = area under curve, higher = better (max 1.0)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
axes[2].plot(fpr, tpr, color='steelblue', linewidth=2.5,
             label=f'model (AUC = {roc_auc:.2f})')
axes[2].plot([0,1], [0,1], 'r--', linewidth=1.5,
             label='random guessing (AUC = 0.50)')
axes[2].set_xlabel('false positive rate')
axes[2].set_ylabel('true positive rate')
axes[2].set_title('ROC curve')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day15_hiring_predictor.png', dpi=150)
plt.show()


# ============================================================
# STEP 8 - predict for new candidates
# ============================================================
print("\n" + "=" * 45)
print("PREDICTING FOR NEW CANDIDATES")
print("=" * 45)

new_candidates = pd.DataFrame({
    'experience': [1,  6,  12],
    'skills':     [3,  8,  13],
    'score':      [55, 75, 92],
    'age':        [23, 30, 38],
    'projects':   [1,  5,  9]
})

new_scaled   = scaler.transform(new_candidates)
new_pred     = model.predict(new_scaled)
new_prob     = model.predict_proba(new_scaled)[:, 1]

labels = ['Fresh Graduate', 'Mid Level', 'Senior Expert']
for i, label in enumerate(labels):
    result = "HIRED ✅" if new_pred[i] == 1 else "NOT HIRED ❌"
    print(f"\n{label}:")
    print(f"  experience: {new_candidates.iloc[i]['experience']} yrs | "
          f"skills: {new_candidates.iloc[i]['skills']} | "
          f"score: {new_candidates.iloc[i]['score']}")
    print(f"  prediction : {result}")
    print(f"  confidence : {new_prob[i]*100:.1f}% chance of being hired")
