
# Final Project Report
## Robust Linear Models under Distribution Shift

**Course:** CSI 536  
**Group 2:** Mehak Seth, Daniel Truax, Juhan Choi  
**Date:** April 2026

---

## Executive Summary

This project investigates how linear classifiers (SVM) and regressors (Ridge Regression) perform when training and test data are drawn from different distributions. We implemented synthetic distribution shifts—covariate shift and label noise—evaluated baseline performance degradation, and explored simple robustification strategies including importance weighting and robust loss functions.

**Key Findings:**
- SVM accuracy drops 34% under moderate covariate shift (σ=1.5)
- Ridge MSE increases 68% under 40% label noise
- RBF kernel importance weighting recovers +20% accuracy for SVM
- Huber regression provides +2% improvement for Ridge under label noise

---

## 1. Introduction

### 1.1 Motivation

Standard supervised learning assumes training and test data come from the same distribution (i.i.d. assumption). In practice, this assumption often breaks:
- Medical models deployed across different hospitals
- Recommendation systems with evolving user preferences
- Autonomous systems operating in varying environments

When distributions shift, model performance degrades. This project explores robustification strategies to mitigate this degradation.

### 1.2 Research Questions

1. How do linear models degrade under distribution shift?
2. Can simple robustification methods improve performance?
3. Which methods work best for different shift types?

### 1.3 Scope

**Models:** SVM (classification), Ridge Regression (regression)  
**Datasets:** Breast Cancer (569 samples), Diabetes (442 samples)  
**Shift Types:** Covariate shift (feature noise), Label noise (label corruption)  
**Methods:** Importance weighting, Data augmentation, Robust loss functions

---

## 2. Background

### 2.1 Distribution Shift

**Covariate Shift:** P(X) changes, P(Y|X) stays constant
- Example: Different sensor calibration between train/test
- Our implementation: Gaussian noise on test features

**Label Noise:** Labels corrupted with random errors
- Example: Human annotation mistakes
- Our implementation: Gaussian noise on test labels

### 2.2 Robustification Strategies

**Importance Weighting:**
- Reweight training samples to match test distribution
- Applicable to covariate shift
- Implementation: Distance-based, kernel-based, density ratio

**Robust Loss Functions:**
- Use losses less sensitive to outliers
- Applicable to label noise
- Implementation: Huber loss (hybrid L1/L2)

---

## 3. Methodology

### 3.1 Experimental Setup

**Phase 1: Baseline Degradation**
- Train models on clean data
- Test on shifted distributions
- Measure performance degradation

**Phase 2: SVM Robustification (Importance Weighting)**
- Implement 4 weighting methods
- Evaluate on covariate shift
- Compare against baseline

**Phase 3: Ridge Robustification (Robust Methods)**
- Implement 4 robust methods
- Evaluate on label noise
- Compare against baseline

### 3.2 Datasets & Preprocessing

**Breast Cancer (SVM):**
- 569 samples, 30 features
- Binary classification (malignant/benign)
- 80/20 train/test split
- StandardScaler normalization

**Diabetes (Ridge):**
- 442 samples, 10 features
- Regression (disease progression)
- 80/20 train/test split
- StandardScaler normalization

### 3.3 Distribution Shift Implementation

**Covariate Shift (SVM):**
```python
X_test_shifted = X_test + np.random.normal(0, σ, X_test.shape)
σ ∈ [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
```

**Label Noise (Ridge):**
```python
y_test_noisy = y_test + np.random.normal(0, noise_rate * np.std(y_test), len(y_test))
noise_rate ∈ [0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%]
```

---

## 4. Results

### 4.1 Phase 1: Baseline Degradation

**SVM (Covariate Shift):**
| Noise (σ) | Accuracy | F1 Score | Change |
|-----------|----------|----------|--------|
| 0.0 | 98.25% | 98.61% | - |
| 1.0 | 90.35% | 91.97% | -7.9% |
| 1.5 | 64.04% | 61.68% | -34.2% |
| 2.0 | 40.35% | 8.11% | -57.9% |

**Finding:** Performance cliff at σ=1.5

**Ridge (Label Noise):**
| Noise | MSE | R² | Increase |
|-------|-----|-----|----------|
| 0% | 2892 | 0.454 | - |
| 20% | 3362 | 0.410 | +16.2% |
| 40% | 4859 | 0.287 | +68.0% |
| 50% | 5447 | 0.250 | +88.3% |

**Finding:** MSE explosion at high noise levels

### 4.2 Phase 2: Importance Weighting (SVM)

**Performance at σ=1.5:**
| Method | Accuracy | Improvement |
|--------|----------|-------------|
| Baseline | 64.04% | - |
| Euclidean | 63.16% | -0.9% |
| **RBF Kernel** | **84.21%** | **+20.2%** |
| Density Ratio | 66.67% | +2.6% |

**Winner:** RBF Kernel Weighting

**Why RBF wins:**
- Gaussian kernel matches Gaussian noise structure
- Uses full test distribution information
- Handles high-dimensional features well

### 4.3 Phase 3: Data Augmentation (Ridge)

**Performance at 40% noise:**
| Method | MSE | MSE Increase | Improvement |
|--------|-----|--------------|-------------|
| Baseline | 4859 | +68.0% | - |
| Noise Injection | 4875 | +68.6% | -0.6% |
| **Huber** | **4799** | **+66.0%** | **+2.0%** |
| Bagging | 4913 | +69.9% | -1.9% |

**Winner:** Huber Regression

**Why Huber wins:**
- Robust loss designed for outliers
- Better generalization properties
- Consistent across all noise levels

---

## 5. Discussion

### 5.1 Method Effectiveness

**RBF Kernel Importance Weighting:**
- ✅ Highly effective for covariate shift (+20%)
- ✅ Simple sklearn implementation
- ✅ Matches Gaussian noise structure
- ⚠️ Requires access to test distribution

**Huber Robust Loss:**
- ✅ Consistent improvement across noise levels (+2%)
- ✅ Theoretically principled
- ✅ No hyperparameter tuning needed
- ⚠️ Modest improvement magnitude

### 5.2 Why Different Improvement Magnitudes?

**Large improvement (Phase 2: +20%):**
- Training-time intervention
- Can reweight samples to match test distribution
- Direct correction of distribution mismatch

**Small improvement (Phase 3: +2%):**
- Test-time label corruption
- Can't fix corrupted ground truth
- Only better generalization helps

### 5.3 Practical Implications

**For Deployment:**
1. Monitor for distribution shift in production
2. Use RBF weighting if covariate shift detected
3. Use Huber loss for noisy label scenarios
4. Simple methods scale better than complex ones

**Trade-offs:**
- Computational cost: Bagging (10x slower) vs Huber (1.5x slower)
- Improvement: RBF (+20%) vs Huber (+2%)
- Scalability: All methods practical for sklearn datasets

---

## 6. Limitations & Future Work

### 6.1 Limitations

1. **Synthetic shifts only:** Real-world shifts may differ
2. **Small datasets:** sklearn datasets (442-569 samples)
3. **Limited shift types:** Only covariate & label noise
4. **No combination strategies:** Didn't test multiple methods together

### 6.2 Future Directions

1. **Real-world evaluation:** Test on actual distribution shifts
2. **Combined methods:** RBF weighting + Huber loss
3. **Other shift types:** Concept drift, prior shift
4. **Deep learning:** Extend to neural networks
5. **Online adaptation:** Update models as shift occurs

---

## 7. Conclusions

### 7.1 Summary of Findings

✅ **Distribution shift significantly degrades performance**
- SVM: -34% accuracy under moderate covariate shift
- Ridge: +68% MSE under moderate label noise

✅ **Simple robustification methods work**
- RBF kernel weighting: +20% accuracy improvement
- Huber regression: +2% MSE reduction

✅ **Method choice depends on shift type**
- Covariate shift → Importance weighting
- Label noise → Robust loss functions

### 7.2 Key Takeaways

1. **Understand your shift type** before choosing methods
2. **Simple methods are effective** - no complex optimization needed
3. **Training-time solutions** more effective than test-time fixes
4. **Prevention is better than cure** - data quality matters most

### 7.3 Contribution

This project demonstrates that:
- Distribution shift is a real problem for linear models
- Simple, practical robustification strategies exist
- Significant improvements are achievable with standard tools
- Understanding shift mechanisms guides method selection

---

## 8. References

1. Shimodaira, H. (2000). "Improving predictive inference under covariate shift"
2. Sugiyama, M., et al. (2008). "Direct importance estimation with model selection"
3. Huber, P. J. (1964). "Robust Estimation of a Location Parameter"
4. Breiman, L. (1996). "Bagging Predictors"
5. Gretton, A., et al. (2009). "Covariate shift by kernel mean matching"

---

## 9. Appendices

### Appendix A: Hyperparameters

**SVM:**
- C = 1.0
- kernel = 'rbf'
- gamma = 'scale'

**Ridge:**
- alpha = 1.0

**Huber:**
- epsilon = 1.35
- alpha = 1.0

**Bagging:**
- n_estimators = 10
- max_samples = 0.8

### Appendix B: Code Repository

All code available at project repository with:
- phase1_baseline.py
- phase2_importance_weighting.py
- phase3_data_augmentation.py
- phase4_final_analysis.py

### Appendix C: Full Results

See CSV files in Results/ directory for complete numerical results.

---

**End of Report**
