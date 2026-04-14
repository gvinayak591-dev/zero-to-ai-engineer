# ============================================================
# Day 9 — NumPy Essentials for AI
# Zero → AI Engineer in 6 Months
# ============================================================

import numpy as np

# ============================================
# 1. CREATING ARRAYS (the ML way)
# ============================================
a = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("Array:", a)
print("Matrix:\n", matrix)
print("Shape:", matrix.shape)
print("Dimensions:", matrix.ndim)
print("Data type:", matrix.dtype)

# ============================================
# 2. ARRAYS YOU'LL USE IN ML EVERY DAY
# ============================================
zeros  = np.zeros((3, 3))
ones   = np.ones((2, 4))
rand   = np.random.rand(3, 3)
randn  = np.random.randn(3, 3)
eye    = np.eye(3)

print("\nRandom Normal:\n", randn)

# ============================================
# 3. INDEXING & SLICING
# ============================================
data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

print("\nFirst row:", data[0])
print("Element [1][2]:", data[1, 2])
print("First 2 rows:\n", data[:2])
print("Last column:", data[:, -1])

# ============================================
# 4. MATH OPERATIONS (this IS neural networks)
# ============================================
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("\nAdd:", a + b)
print("Multiply:", a * b)
print("Dot product:", np.dot(a, b))
print("Matrix multiply:\n", matrix @ matrix)

# ============================================
# 5. AGGREGATIONS (used in model evaluation)
# ============================================
scores = np.array([85, 92, 78, 95, 88, 76, 91])

print("\nMean:", np.mean(scores))
print("Std Dev:", np.std(scores))
print("Max:", np.max(scores))
print("Min:", np.min(scores))
print("Argmax (index of max):", np.argmax(scores))

# ============================================
# 6. BROADCASTING (numpy's superpower)
# ============================================
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("\nMatrix + 10:\n", matrix + 10)

mean       = matrix.mean(axis=0)
std        = matrix.std(axis=0)
normalized = (matrix - mean) / std
print("\nNormalized:\n", normalized)

# ============================================
# 7. RESHAPING (critical for deep learning)
# ============================================
flat      = np.array([1, 2, 3, 4, 5, 6])
reshaped  = flat.reshape(2, 3)
print("\nReshaped:\n", reshaped)

img       = np.random.rand(28, 28)
flattened = img.flatten()
print("Image shape:", img.shape, "→ Flattened:", flattened.shape)
