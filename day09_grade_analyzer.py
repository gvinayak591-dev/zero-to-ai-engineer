# ============================================================
# Day 9 — Mini Project: Student Grade Analyzer
# Zero → AI Engineer in 6 Months
# ============================================================

import numpy as np

# Simulated student data: 10 students, 5 subjects
np.random.seed(42)
grades = np.random.randint(50, 100, size=(10, 5))

subjects = ["Math", "Physics", "CS", "English", "Stats"]
students = [f"Student_{i+1}" for i in range(10)]

print("=" * 50)
print("       STUDENT GRADE ANALYZER")
print("=" * 50)
print(f"\nGrades Matrix (10 students x 5 subjects):\n{grades}")

# --- Analysis ---
student_avg = grades.mean(axis=1)
subject_avg = grades.mean(axis=0)

print("\n📊 Student Averages:")
for i, avg in enumerate(student_avg):
    grade = "A" if avg >= 85 else "B" if avg >= 70 else "C"
    print(f"  {students[i]}: {avg:.1f} → Grade {grade}")

print("\n📚 Subject Averages:")
for i, avg in enumerate(subject_avg):
    print(f"  {subjects[i]}: {avg:.1f}")

# --- Top & Bottom performers ---
top_student     = students[np.argmax(student_avg)]
weak_student    = students[np.argmin(student_avg)]
hardest_subject = subjects[np.argmin(subject_avg)]
easiest_subject = subjects[np.argmax(subject_avg)]

print(f"\n🏆 Top Performer: {top_student} ({np.max(student_avg):.1f})")
print(f"⚠️  Needs Help:    {weak_student} ({np.min(student_avg):.1f})")
print(f"😅 Hardest Subject: {hardest_subject}")
print(f"😊 Easiest Subject: {easiest_subject}")

# --- Pass/Fail (threshold = 60) ---
pass_fail  = grades >= 60
fail_count = (~pass_fail).sum(axis=1)

print("\n❌ Students with failures per subject:")
for i, fails in enumerate(fail_count):
    if fails > 0:
        print(f"  {students[i]}: {fails} subject(s) failed")

print("\n✅ Overall pass rate:", f"{pass_fail.mean()*100:.1f}%")
