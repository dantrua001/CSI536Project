# Phase 2: Importance Weighting for SVM Robustness

**CSI 536 - Robust Linear Models under Distribution Shift**  
**Group 2:** Mehak Seth, Daniel Truax, Juhan Choi

---

## Overview

Phase 2 explores **importance weighting** methods to improve SVM robustness against **covariate shift**. When test data is shifted (contaminated with Gaussian noise), importance weighting reweights training samples to emphasize those that better represent the test distribution.

**Key Insight:** Not all training samples are equally useful when the test distribution has shifted. By giving higher weight to samples that look more like the shifted test data, we can improve model robustness.

---

## Problem Setup

### What is Covariate Shift?

The input feature distribution P(X) changes between training and test, but the relationship P(Y|X) stays the same.

- **Training:** Clean Breast Cancer features
- **Testing:** Same features + Gaussian noise (σ = 0 to 3.0)
- **Challenge:** SVM trained on clean data performs poorly on noisy test data

### The Performance Cliff (from Phase 1)

At **σ = 1.5**, SVM accuracy drops significantly. This is our target improvement zone.

---

## Methods Implemented

### 1. **Uniform Weighting** (Baseline)

**What it does:** All training samples have equal weight (weight = 1.0)

**Why it's here:** This is the baseline - same as Phase 1. It represents standard SVM training with no importance weighting.

**Expected performance:** No improvement (0% gain)

**Code concept:**
```
weights = np.ones(len(X_train))  # All weights = 1.0
```

---

### 2. **Euclidean Distance Weighting**

**What it does:** Weights training samples by their Euclidean distance to the test distribution center

**How it works:**
1. Compute the center (mean) of the shifted test distribution
2. Measure distance from each training sample to this center
3. Convert distance to weight using exponential decay: `w = exp(-distance² / bandwidth²)`
4. Closer samples → higher weight

**Why this helps:** Training samples that are closer to the shifted test data are more representative and should have more influence.

**Expected performance:** +1-2% accuracy improvement at σ=1.5

**Pros:**
- Very interpretable
- Simple to implement
- No hyperparameters (bandwidth is fixed)

**Cons:**
- Assumes all features equally important
- Sensitive to curse of dimensionality
- Uses only the test center (not full distribution)

**Code concept:**
```
test_center = np.mean(X_test_shifted, axis=0)
distances = euclidean_distances(X_train, test_center)
weights = np.exp(-distances**2 / (2 * bandwidth**2))
```

---

### 3. **RBF Kernel Weighting**

**What it does:** Weights training samples by their RBF kernel similarity to the entire test distribution

**How it works:**
1. Compute RBF kernel between each training sample and ALL test samples
2. Average the kernel similarity across test samples
3. Use average similarity as the weight

**Why this helps:** RBF kernel captures nonlinear relationships and works well with Gaussian noise (which is what we're adding). It considers the entire test distribution, not just the center.

**Expected performance:** +2-4% accuracy improvement at σ=1.5 (BEST METHOD)

**Pros:**
- Handles nonlinear relationships
- Naturally suited to Gaussian noise structure
- Uses full test distribution information
- Works well in higher dimensions

**Cons:**
- Slightly less interpretable than Euclidean
- Requires gamma parameter (we use default γ=0.1)

**Code concept:**
```
kernel_matrix = rbf_kernel(X_train, X_test_shifted, gamma=0.1)
weights = np.mean(kernel_matrix, axis=1)  # Average over test samples
```

**Why RBF Kernel is expected to work best:**
- Our shift is Gaussian noise
- RBF kernel is based on Gaussian function
- Perfect mathematical match!

---

### 4. **Density Ratio Estimation**

**What it does:** Estimates the density ratio `p_test(x) / p_train(x)` using probabilistic classification

**How it works:**
1. Combine train and test data
2. Label training samples as 0, test samples as 1
3. Train a logistic regression classifier to distinguish them
4. Use predicted probabilities as weights: `w = P(test|x) / P(train|x)`
5. Clip extreme weights for stability

**Why this helps:** Samples that look more like test data (high `P(test|x)`) get higher weight. This directly estimates the importance ratio.

**Expected performance:** +1-3% accuracy improvement at σ=1.5

**Pros:**
- Theoretically principled (directly estimates density ratio)
- Works for any type of distribution shift
- Adaptive to the specific shift

**Cons:**
- Can be unstable (requires weight clipping)
- Relies on classifier quality
- More complex than distance-based methods

**Code concept:**
```
# Train classifier: train vs test
clf = LogisticRegression()
clf.fit(X_combined, [0, 0, ..., 1, 1])  # 0=train, 1=test

# Get probabilities for training samples
probs = clf.predict_proba(X_train)
weights = probs[:, 1] / probs[:, 0]  # P(test) / P(train)
weights = np.clip(weights, 0.1, 10.0)  # Clip for stability
```

---

## How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn --break-system-packages
```

### Execution
```bash
python phase2_importance_weighting.py
```

**Runtime:** ~30-60 seconds (36 experiments: 4 methods × 9 noise levels)

---

## Output Files

After running the script, you'll find:

### 1. **`Results/Phase2/phase2_results.csv`**
Complete results table with 36 rows (4 methods × 9 noise levels)

Columns:
- `method`: Weighting method name
- `sigma`: Noise level
- `accuracy`: Classification accuracy
- `f1`: F1 score
- `precision`: Precision
- `recall`: Recall

### 2. **`Results/Phase2/importance_weights_comparison.png`**
4-panel histogram showing weight distributions for each method at σ=1.5

**What to look for:**
- **Uniform:** Flat distribution (all weights = 1.0)
- **Euclidean/RBF:** Spread distribution (varying weights)
- **Density Ratio:** May have some extreme values (why we clip)

### 3. **`Results/Phase2/svm_robustness_comparison.png`**
4-panel performance curves (Accuracy, F1, Precision, Recall) vs noise level

**What to look for:**
- All methods start together at σ=0 (clean data)
- Performance diverges as noise increases
- Red dotted line marks σ=1.5 (the cliff)
- Best method should show smallest degradation

---

## Interpreting Results

### Expected Performance at σ=1.5 (The Cliff)

Based on Phase 1, baseline accuracy at σ=1.5 was around **63-65%**.

| Method | Expected Accuracy | Improvement | Why |
|--------|------------------|-------------|-----|
| Uniform (Baseline) | ~63-65% | 0% | No weighting |
| Euclidean Distance | ~64-67% | +1-2% | Simple distance weighting |
| **RBF Kernel** | ~65-69% | +2-4% | Best match for Gaussian noise |
| Density Ratio | ~64-68% | +1-3% | Adaptive but can be unstable |

### What Counts as Success?

- **Modest improvement (1-2%):** Expected and realistic
- **Good improvement (2-4%):** Excellent result for simple methods
- **No improvement (<1%):** Still valuable - shows limits of approach

**Important:** Importance weighting is NOT a miracle cure. It provides modest but meaningful improvements. The goal is to demonstrate that reweighting helps, not to fully recover baseline performance.

### Understanding the Plots

**Weight Distribution Plot:**
- Uniform should be a vertical line (all weights equal)
- Other methods should show variation (some samples weighted higher)
- High variance in weights = method is discriminating between samples

**Robustness Comparison Plot:**
- Look at the gap between methods at σ=1.5
- Steeper decline = more sensitive to shift
- Flatter curve = more robust

---

## Why Simple Methods?

**Q: Why not use Kernel Mean Matching (KMM)?**

**A:** KMM is a sophisticated optimization-based method that:
- Requires solving a quadratic program (cvxpy dependency)
- Is the dedicated focus of another team's project
- Is beyond the scope of a "basic robustification" project phase

**Our approach:** Demonstrate that even simple importance weighting helps. This is more appropriate for:
- Educational purposes (easier to understand)
- Project scope (3-week timeline)
- Fair comparison (doesn't overlap with other teams)

---


---

## Next Steps (Phase 3)

After completing Phase 2, we'll move to:

**Phase 3: Data Augmentation for Ridge Regression**
- Goal: Make Ridge robust to label noise
- Methods: Noise injection, Huber loss, bagging
- Target: Reduce MSE increase at 40% noise from +130% to <80%

---

## Key Takeaways

1. **Importance weighting helps** - even simple methods provide 1-4% improvements
2. **RBF kernel is best** - because our shift is Gaussian and RBF is Gaussian-based
3. **No silver bullet** - improvements are modest but meaningful
4. **Simplicity matters** - complex methods aren't always necessary

---

