
# Robust Linear Models under Distribution Shift
## CSI 536 - Group 2: Mehak Seth, Daniel Truax, Juhan Choi

---

## Project Overview

**Research Question:**
How do linear models perform when training and test distributions differ?

**Approach:**
1. ✅ Implement baseline models (SVM, Ridge)
2. ✅ Design synthetic distribution shifts
3. ✅ Evaluate performance degradation
4. ✅ Explore robustification strategies

**Models:** SVM (classification) + Ridge Regression (regression)
**Datasets:** Breast Cancer + Diabetes (sklearn)

---

## Phase 1: Baseline Degradation

### SVM - Covariate Shift
- Added Gaussian noise to test features
- **Key Finding:** Performance cliff at σ=1.5
  - Accuracy: 98.3% → 64.0% (-34.3%)
  - Margin collapse at σ=2.0

### Ridge - Label Noise
- Random corruption of test labels
- **Key Finding:** MSE explosion at 40% noise
  - MSE increase: +68.0% over baseline
  - R² score drops significantly

**Takeaway:** Both models degrade under distribution shift

---

## Phase 2: Importance Weighting (SVM)

### Methods Tested:
1. Uniform (baseline)
2. Euclidean Distance
3. **RBF Kernel** ⭐
4. Density Ratio

### Results:
- **RBF Kernel: +20.2% accuracy at σ=1.5**
- Baseline: 64.04% → RBF: 84.21%
- Why? Gaussian kernel matches Gaussian noise structure

**Takeaway:** Simple importance weighting significantly improves robustness

---

## Phase 3: Data Augmentation (Ridge)

### Methods Tested:
1. Baseline Ridge
2. Noise Injection
3. **Huber Regression** ⭐
4. Bagging

### Results:
- **Huber: +2.0% improvement at 40% noise**
- Baseline: +68.0% MSE increase → Huber: +66.0%
- Robust loss function handles label noise better

**Takeaway:** Modest but consistent improvement from robust methods

---

## Key Lessons Learned

### 1. Match Method to Shift Type
- Covariate shift → Importance weighting (RBF: +20%)
- Label noise → Robust loss (Huber: +2%)

### 2. Simple Methods Work
- No complex optimization needed
- sklearn implementations sufficient
- Practical and scalable

### 3. Different Shift Magnitudes
- Training-time solutions more effective (+20%)
- Test-time noise harder to mitigate (+2%)
- Prevention better than cure

---

## Practical Recommendations

### For Production Systems:
1. **Monitor distribution shift** in deployment
2. **Use RBF weighting** if covariate shift expected
3. **Use Huber loss** for noisy label scenarios
4. **Keep methods simple** for scalability

### For Future Work:
- Test on real-world datasets
- Combine multiple robustification strategies
- Investigate other shift types (concept drift)

---

## Conclusions

✅ **Successfully demonstrated** distribution shift impact
✅ **Implemented effective** robustification strategies  
✅ **Achieved significant improvements** (up to +20%)
✅ **Validated simple methods** work in practice

**Final Insight:** Understanding shift type is crucial for choosing the right robustification strategy.

---

## Questions?

**Group 2:**
- Mehak Seth
- Daniel Truax
- Juhan Choi

**Course:** CSI 536 - Robust Linear Models under Distribution Shift

Thank you!
