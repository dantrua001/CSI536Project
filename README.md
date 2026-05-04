# Robust Linear Models under Distribution Shift

**CSI 536 - Machine Learning | Project Code: R**  
**Team Members:** Mehak, Daniel Truax, Juhan Choi

---

##  Project Overview

This project evaluates how two fundamental machine learning models perform when test data differs from training data—a common real-world challenge called **distribution shift**. We implement SVM (classification) and Ridge Regression (regression), apply four types of synthetic shifts at varying intensities, and measure performance degradation.

**Key Question:** How robust are these models when deployment data doesn't match training data?

---

## Methods & Datasets

| Method | Model | Dataset | Task | Samples | Features | Baseline Metric |
|--------|-------|---------|------|---------|----------|-----------------|
| **1. SVM** | Support Vector Machine (RBF) | Breast Cancer | Binary Classification | 569 | 30 | Accuracy: ~96.5% |
| **2. Ridge** | Ridge Regression (α=1.0) | Diabetes | Regression | 442 | 10 | R²: ~0.45, MSE: ~2900 |

Both datasets are scikit-learn built-ins—no downloads required.

---

##  Distribution Shifts Tested

We test **4 shift types** at **7 intensity levels** (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):

| Shift Type | Description | What It Simulates | Severity |
|------------|-------------|-------------------|----------|
| **Covariate Shift** | Add Gaussian noise to features | Sensor drift, measurement errors | Moderate |
| **Label Noise** | Flip labels (SVM) / Add noise to targets (Ridge) | Annotation errors, measurement noise | High |
| **Subset Shift** | Remove samples from feature range | Missing subpopulations | Variable |
| **Feature Corruption** | Corrupt 3 specific features | Feature-specific degradation | Low-Moderate |

**Critical Design:** Training data remains clean; shifts only applied to test data. This simulates real deployment where models encounter data different from training.

---

## Parameter Selection & Methodology

### Why These Models?

**SVM with RBF Kernel:**
- **Chosen because:** RBF kernel can capture non-linear decision boundaries, which is important for complex medical data like Breast Cancer features
- **Parameter C=1.0 (default):** Balanced regularization—not too strict (underfitting) or too loose (overfitting)
- **Why not linear kernel?** Breast Cancer data has complex feature interactions that benefit from non-linear mapping

**Ridge Regression with α=1.0:**
- **Chosen because:** L2 regularization prevents overfitting on small datasets (442 samples)
- **Parameter α=1.0 (default):** Standard regularization strength, proven effective for general-purpose regression
- **Why not higher α?** Would shrink coefficients too much, potentially underfitting
- **Why not lower α?** Would reduce regularization benefit, approaching ordinary least squares

### Train/Test Split: 80/20

**Why 80/20?**
- **80% training (SVM: 455 samples, Ridge: 353 samples):** Sufficient data for model to learn patterns
- **20% testing (SVM: 114 samples, Ridge: 89 samples):** Large enough to measure performance reliably
- **Random state=42:** Ensures reproducibility—same split every time

### Shift Intensity Levels: 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0

**Why these 7 levels?**
- **Intensity 0:** Baseline (no shift) to measure ideal performance
- **0.5-1.0:** Mild shifts—realistic deployment scenarios
- **1.5-2.0:** Moderate shifts—stress testing
- **2.5-3.0:** Severe shifts—worst-case scenarios
- **Why not higher?** Beyond 3.0, shifts become unrealistic (e.g., >30% label flipping)

---

##  Detailed Shift Implementations

### 1. Covariate Shift: Gaussian Noise

#### How It Works
```python
def apply_covariate_shift(X_test, intensity):
    noise = np.random.randn(*X_test.shape)  # Standard normal distribution
    X_shifted = X_test + noise * intensity
    return X_shifted
```

#### Parameter Explanation
- **`np.random.randn()`:** Generates noise from standard normal distribution (mean=0, std=1)
- **Why Gaussian?** Most common noise distribution in real-world measurements
- **Intensity as multiplier:** Controls noise magnitude
  - Intensity 0.5: Small perturbations (±0.5 standard deviations)
  - Intensity 1.0: Moderate noise (±1.0 standard deviations)
  - Intensity 3.0: Severe noise (±3.0 standard deviations)

#### What This Means in Practice
- **For SVM (Breast Cancer features):** If a feature like "mean radius" normally has value 14.5, with intensity 1.0, it might become 13.8 or 15.2 (±0.7 units of noise)
- **For Ridge (Diabetes features):** If BMI is 0.05 (normalized), with intensity 1.0, it might become -0.02 or 0.12

**Why this approach?**
- Simulates sensor drift, measurement errors, equipment calibration issues
- Uniform across all features—worst case where all sensors degrade simultaneously

---

### 2. Label Noise

#### For SVM (Classification): Label Flipping

```python
def apply_label_noise(y_test, intensity):
    flip_probability = intensity * 0.1  # 10% per intensity unit
    mask = np.random.rand(len(y_test)) < flip_probability
    y_shifted = y_test.copy()
    y_shifted[mask] = 1 - y_shifted[mask]  # Flip binary labels
    return y_shifted
```

#### What the Percentages Mean (SVM)

| Intensity | Flip Probability | Meaning | Example (114 test samples) |
|-----------|------------------|---------|----------------------------|
| 0.5 | 5% | **5% of labels flipped** | ~6 labels wrong |
| 1.0 | 10% | **10% of labels flipped** | ~11 labels wrong |
| 1.5 | 15% | **15% of labels flipped** | ~17 labels wrong |
| 2.0 | 20% | **20% of labels flipped** | ~23 labels wrong |
| 2.5 | 25% | **25% of labels flipped** | ~29 labels wrong |
| 3.0 | 30% | **30% of labels flipped** | ~34 labels wrong |

**Example:** At intensity 1.0, if there are 114 test samples:
- 10% × 114 ≈ 11 samples have their labels flipped
- A tumor labeled "malignant" (1) becomes "benign" (0)
- A tumor labeled "benign" (0) becomes "malignant" (1)

**Why 10% per unit?**
- Intensity 1.0 → 10% is a realistic annotation error rate in medical datasets
- Intensity 3.0 → 30% is severe but still plausible with poor annotation quality

---

#### For Ridge (Regression): Target Noise

```python
def apply_label_noise(y_test, intensity):
    noise = np.random.randn(len(y_test))  # Standard normal noise
    noise_scaled = noise * intensity * np.std(y_test)  # Scale by target std
    y_shifted = y_test + noise_scaled
    return y_shifted
```

#### What the Percentages Mean (Ridge)

For Ridge, we don't use "flipping" because targets are continuous (not binary). Instead, we add **noise proportional to the target's standard deviation**.

**Parameter Explanation:**
- **`np.std(y_test)`:** Standard deviation of disease progression values (≈ 59.4)
- **Scaling:** `intensity × std(y_test)` ensures noise magnitude is meaningful relative to target range

| Intensity | Noise Magnitude | What This Means |
|-----------|-----------------|-----------------|
| 0.5 | 0.5 × 59.4 ≈ **30 units** | Adds ±30 to disease progression value |
| 1.0 | 1.0 × 59.4 ≈ **59 units** | Adds ±59 (noise equals one std) |
| 1.5 | 1.5 × 59.4 ≈ **89 units** | Adds ±89 |
| 2.0 | 2.0 × 59.4 ≈ **119 units** | Adds ±119 (noise equals two stds) |
| 2.5 | 2.5 × 59.4 ≈ **149 units** | Adds ±149 |
| 3.0 | 3.0 × 59.4 ≈ **178 units** | Adds ±178 (noise equals three stds) |

**Example:** 
- True disease progression value: 150
- At intensity 1.0: Corrupted value could be 150 + 59 = 209 or 150 - 59 = 91
- At intensity 2.0: Corrupted value could be 150 + 119 = 269 or 150 - 119 = 31

**What "40% corruption" or "50% corruption" means:**
- This phrasing can be misleading—we're not corrupting 40% of samples
- Instead, at intensity 2.0, noise magnitude is **2 standard deviations**, which can shift values by ~40% of the target range (25 to 346)
- At intensity 2.5, noise is ~2.5 stds, which can shift values by ~50% of the target range

**More accurate interpretation:**

| Intensity | Noise (in stds) | Approximate % of Range | Effect |
|-----------|-----------------|------------------------|--------|
| 0.5 | 0.5 std | ~10% of range | Mild measurement error |
| 1.0 | 1.0 std | ~20% of range | Moderate measurement error |
| 2.0 | 2.0 std | ~40% of range | Severe measurement error |
| 2.5 | 2.5 std | ~50% of range | Extreme measurement error |
| 3.0 | 3.0 std | ~60% of range | Catastrophic measurement error |


**Why scale by std(y_test)?**
- Makes noise magnitude relative to the actual target distribution
- Ensures consistent interpretation across different datasets
- Intensity 1.0 always means "noise equal to one standard deviation"

---

### 3. Subset Shift: Sample Removal

```python
def apply_subset_shift(X_test, y_test, intensity):
    feature_idx = 0  # First feature (Age for Diabetes, Mean Radius for Breast Cancer)
    mean = X_test[:, feature_idx].mean()
    std = X_test[:, feature_idx].std()
    
    threshold = mean - intensity * std
    mask = X_test[:, feature_idx] >= threshold
    
    return X_test[mask], y_test[mask]
```

#### Parameter Explanation

**Why first feature?**
- **For SVM (Breast Cancer):** First feature is "mean radius"—removing smaller tumors
- **For Ridge (Diabetes):** First feature is "age"—removing younger patients

**How threshold works:**
- **`mean - intensity × std`:** Progressively removes samples from lower end
- Intensity 0.5: Removes samples below (mean - 0.5×std) ≈ bottom 30%
- Intensity 1.0: Removes samples below (mean - 1.0×std) ≈ bottom 16%
- Intensity 2.0: Removes samples below (mean - 2.0×std) ≈ bottom 2.5%
- Intensity 3.0: Removes samples below (mean - 3.0×std) ≈ bottom 0.1%

**Number of samples remaining:**

For Ridge (89 test samples):

| Intensity | Threshold | Samples Removed | Samples Remaining | % Remaining |
|-----------|-----------|-----------------|-------------------|-------------|
| 0.0 | -∞ | 0 | 89 | 100% |
| 0.5 | mean - 0.5σ | ~13 | ~76 | 85% |
| 1.0 | mean - 1.0σ | ~26 | ~63 | 71% |
| 2.0 | mean - 2.0σ | ~51 | ~38 | 43% |
| 3.0 | mean - 3.0σ | ~68 | ~21 | 24% |

**Why this matters:**

**Why this matters:**
- Simulates missing subpopulations (e.g., young patients not in deployment data)
- Tests model's ability to extrapolate beyond training distribution
- Real-world scenario: Model trained on all ages, deployed only on older patients

---

### 4. Feature Corruption

```python
def apply_feature_corruption(X_test, intensity):
    X_corrupted = X_test.copy()
    n_features = X_test.shape[1]
    
    # Randomly select 3 features to corrupt
    features_to_corrupt = np.random.choice(n_features, size=3, replace=False)
    
    for feature_idx in features_to_corrupt:
        noise = np.random.randn(len(X_test)) * intensity
        X_corrupted[:, feature_idx] += noise
    
    return X_corrupted
```

#### Parameter Explanation

**Why 3 features?**
- **For SVM (30 features):** 3 features = 10% of features corrupted
- **For Ridge (10 features):** 3 features = 30% of features corrupted
- Represents realistic scenario where only some sensors/measurements degrade

**How corruption works:**
- Each selected feature gets independent Gaussian noise
- Noise magnitude = intensity (same as covariate shift)
- Other features remain clean

**Example for Ridge (10 features):**
If features [2, 5, 7] (BMI, S2, S4) are randomly selected:
- Feature 2 (BMI): Gets noise ±intensity
- Feature 5 (S2): Gets noise ±intensity  
- Feature 7 (S4): Gets noise ±intensity
- Features [0, 1, 3, 4, 6, 8, 9]: Remain unchanged

**Why this approach?**
- More realistic than corrupting all features
- Simulates partial sensor failure or specific measurement errors
- Tests model's ability to rely on clean features when some are corrupted

---

## Project Structure

```
distribution-shift-project/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── svm_experiments.py                # SVM implementation (Method 1)
├── ridge_experiments.py              # Ridge implementation (Method 2)
│
└── results/                          # All output plots
    ├── svm/                          # 5 SVM plots
    │   ├── covariate_shift_results.png
    │   ├── label_noise_results.png
    │   ├── subset_shift_results.png
    │   ├── feature_corruption_results.png
    │   └── comparison_plot.png
    │
    └── ridge/                        # 5 Ridge plots
        ├── covariate_shift_results.png
        ├── label_noise_results.png
        ├── subset_shift_results.png
        ├── feature_corruption_results.png
        └── comparison_plot.png
```

---

## How to Run

### Prerequisites

```bash
# Requires Python 3.8+
python --version

# Install dependencies
pip install numpy scikit-learn matplotlib seaborn pandas
```

### Run All Experiments (Recommended)

```bash
# Run both SVM and Ridge experiments
python svm_experiments.py
python ridge_experiments.py

# Expected runtime: ~40-90 seconds total
# Output: 10 plots in results/ directory
```

### Run Individual Methods

```bash
# Option 1: SVM only (~30-60 seconds)
python svm_experiments.py

# Option 2: Ridge only (~10-30 seconds)
python ridge_experiments.py
```

### What Gets Generated

After running both scripts:
- **10 total plots** (5 per method)
- **Console output** with performance metrics at each intensity
- **Results folders** automatically created if needed

---

##  Results & Interpretation

### Output Files

Each method generates **5 plots**:

1. **Individual Shift Plots (4 files):**
   - `covariate_shift_results.png` - Gaussian noise impact
   - `label_noise_results.png` - Label/target corruption impact
   - `subset_shift_results.png` - Sample removal impact
   - `feature_corruption_results.png` - Feature-specific noise impact

2. **Comparison Plot (1 file):**
   - `comparison_plot.png` - All 4 shifts on one graph for direct comparison

### How to Read SVM Results

**Plot Structure:**
- X-axis: Shift intensity (0 to 3.0)
- Y-axis: Accuracy (0 to 1.0, higher = better)

**Typical Performance:**
```
Baseline (Intensity 0):         0.965 accuracy (96.5% correct)
Moderate Shift (Intensity 1.0): 0.930 accuracy (93.0% correct)
Severe Shift (Intensity 3.0):   0.860 accuracy (86.0% correct)
```

**Label Noise Impact (SVM):**
At intensity 1.0 (10% labels flipped):
- Accuracy drops from 96.5% → 88.6%
- This means ~8% absolute drop
- In clinical terms: 8 more misdiagnoses per 100 cases

**Key Observation:** Label noise typically causes steepest degradation; subset shift shows most robustness.

### How to Read Ridge Results

**Plot Structure:**
- X-axis: Shift intensity (0 to 3.0)
- Left Y-axis (Blue): MSE (lower = better)
- Right Y-axis (Red): R² (higher = better)

**Typical Performance:**

```
Shift Type          | Baseline  | Moderate   | Severe
                    | (Int. 0)  | (Int. 1.0) | (Int. 3.0)
--------------------|-----------|------------|------------
Covariate Shift:
  MSE:              | 2900      | 3587       | 7826
  R²:               | 0.45      | 0.32       | -0.48

Label Noise:
  MSE:              | 2900      | 5874       | 46789
  R²:               | 0.45      | 0.11       | -7.85 
```

**Label Noise Impact (Ridge):**
At intensity 1.0 (noise = 1 std ≈ 59 units):
- MSE increases from 2900 → 5874 (2× worse)
- R² drops from 0.45 → 0.11 (loses most predictive power)
- Predictions off by ±77 units on average (vs. ±54 at baseline)

At intensity 2.5 (noise = 2.5 stds ≈ 149 units):
- MSE explodes to 32,457 (11× baseline)
- R² becomes -5.12 (model worse than predicting mean)
- Predictions essentially random at this point

**Important:** R² can go negative when model performs worse than predicting the mean—this is expected at severe shifts!

---

## Implementation Details

### Method 1: SVM Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM with RBF kernel
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Apply shifts to X_test only, measure accuracy degradation
```

**Key Parameters:**
- Kernel: RBF (captures non-linear patterns)
- C: 1.0 (default regularization)
- Train/Test: 455/114 samples

---

### Method 2: Ridge Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Load dataset
data = load_diabetes()
X, y = data.data, data.target

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Ridge with L2 regularization
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)

# Apply shifts to X_test or y_test, measure MSE/R² degradation
```

**Key Parameters:**
- Alpha: 1.0 (regularization strength)
- Train/Test: 353/89 samples







## Key Findings & Analysis

### Robustness Comparison

Based on our experiments:

**SVM (Classification):**
- **Most robust to:** Subset Shift (sometimes even improves), Feature Corruption
- **Least robust to:** Label Noise (30% flip → 72% accuracy)
- **Degradation pattern:** Gradual decline, relatively stable
- **Critical insight:** Can maintain >85% accuracy up to intensity 3.0 for most shifts
- **Why?** SVM's margin-based approach provides inherent noise tolerance; subset shift may remove outliers that hurt performance

**Ridge Regression:**
- **Most robust to:** Feature Corruption, Subset Shift
- **Least robust to:** Label Noise (R² from +0.45 to -7.85—catastrophic)
- **Degradation pattern:** Label noise causes exponential MSE growth, others moderate
- **Critical insight:** Target noise destroys predictive power completely
- **Why?** Ridge optimizes squared error directly; corrupting targets destroys the entire optimization objective

### Cross-Method Comparison

| Aspect | SVM | Ridge |
|--------|-----|-------|
| **Overall Robustness** | More robust | Less robust |
| **Worst-case Degradation** | ~28% accuracy drop | R² from +0.45 to -7.85 |
| **Label Noise Impact** | Severe but recoverable | Catastrophic |
| **Covariate Shift Impact** | Moderate (11% drop at int. 3.0) | Moderate (R² from 0.45 to -0.48) |
| **Best Case Shift** | Subset shift (sometimes improves!) | Feature corruption (R² stays positive) |
| **Practical Usability** | Still functional at high shifts | Unusable at intensity >2.0 for label noise |

### Why These Differences?

**Classification vs. Regression:**
- **Discrete decisions** (SVM) more tolerant to noise than **continuous predictions** (Ridge)
- Flipping a label still leaves binary structure; adding noise to continuous target introduces unbounded error

**Model Architecture:**
- **SVM:** Decision boundary based on support vectors (few samples); less affected by global noise
- **Ridge:** Uses all training data; sensitive to changes in target distribution

**Optimization Objective:**
- **SVM:** Maximizes margin; some noise tolerance built in
- **Ridge:** Minimizes squared error directly; target corruption directly destroys objective

### Practical Implications

1. **For Classification Tasks:** 
   - SVM shows good robustness across shifts
   - Suitable for deployment with monitoring
   - Critical: Validate annotation quality (label noise most harmful)

2. **For Regression Tasks:** 
   - Ridge extremely vulnerable to target noise
   - Requires careful data validation before training
   - Consider robust regression methods (Huber, RANSAC) for noisy targets

3. **General Insights:** 
   - Both models more robust to feature noise than label/target noise
   - Feature corruption less harmful than expected (clean features compensate)
   - Subset shift can surprisingly improve performance (removes outliers)

4. **Monitoring Strategy:** 
   - Track performance on holdout set
   - Alert if drops >10% from baseline
   - For regression: Watch for R² going negative (complete failure indicator)

5. **Deployment Recommendations:**
   - **SVM:** Deploy with confidence up to moderate shifts (intensity ~1.5)
   - **Ridge:** Only deploy if target quality guaranteed; very sensitive to measurement errors
   - Both: Implement data quality checks before inference

---

## References

1. **SVM Theory:**
   - Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.

2. **Ridge Regression:**
   - Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55-67.

3. **Distribution Shift:**
   - Quionero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (2009). Dataset shift in machine learning. MIT Press.

4. **Noise Robustness:**
   - Frénay, B., & Verleysen, M. (2014). Classification in the presence of label noise: A survey. IEEE transactions on neural networks and learning systems, 25(5), 845-869.

5. **Scikit-learn Documentation:**
   - https://scikit-learn.org/stable/

---



