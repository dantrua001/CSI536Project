"""
================================================================================
PHASE 1: BASELINE PERFORMANCE & DEGRADATION ANALYSIS
================================================================================
Course: CSI 536 - Robust Linear Models under Distribution Shift
Group ID: 2
Members: Mehak Seth, Daniel Truax & Juhan Choi

Purpose:
    This script establishes baseline performance for SVM and Ridge Regression
    models, then systematically tests their degradation under distribution shift.

    Experiment 1: SVM + Covariate Shift (Breast Cancer Dataset)
    Experiment 2: Ridge + Label Noise (Diabetes Dataset)

Output:
    All results stored in ./Results/ folder:
    - svm_degradation.png: Performance curves for SVM
    - ridge_degradation.png: Performance curves for Ridge
    - svm_degradation_results.csv: Numerical results for SVM
    - ridge_degradation_results.csv: Numerical results for Ridge
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


# ============================================================================
# RESULTS FOLDER SETUP
# ============================================================================

def setup_results_folder():
    """
    Create Results folder if it doesn't exist.
    Returns the path to the Results folder.
    """
    results_dir = Path("Results")
    results_dir.mkdir(exist_ok=True)
    print(f"\nResults folder ready: {results_dir.absolute()}")
    return results_dir


# ============================================================================
# EXPERIMENT 1: SVM WITH COVARIATE SHIFT
# ============================================================================

def svm_covariate_shift_experiment():
    """
    Test SVM performance across multiple covariate shift intensities.

    Process:
        1. Load Breast Cancer dataset (569 samples, 30 features)
        2. Train SVM on clean training data
        3. Add Gaussian noise to test features (X_test) at different levels
        4. Measure how accuracy degrades as noise increases

    Returns:
        df_results: DataFrame with metrics for each shift level
        svm: Trained SVM model
        Additional data for visualization
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SVM with Covariate Shift (Breast Cancer Dataset)")
    print("=" * 70)

    # ========== STEP 1: Load and Prepare Data ==========
    print("\n[1/5] Loading and preparing data...")
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Scale features (critical for SVM - ensures all features contribute equally)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data (80/20, stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: Malignant ({np.sum(y == 0)}), Benign ({np.sum(y == 1)})")

    # ========== STEP 2: Train Baseline SVM ==========
    print("\n[2/5] Training baseline SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # Test on clean data (no shift)
    y_pred_clean = svm.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred_clean)
    print(f"   Baseline Accuracy (no shift): {baseline_accuracy:.4f}")

    # ========== STEP 3: Define Shift Levels ==========
    print("\n[3/5] Defining shift intensity levels...")
    shift_levels = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"   Testing {len(shift_levels)} shift levels: {shift_levels}")

    # ========== STEP 4: Test Across Shift Levels ==========
    print("\n[4/5] Testing SVM across shift intensities...")

    # Storage for results
    results = {
        'shift_sigma': [],
        'accuracy': [],
        'f1_score': [],
        'precision': [],
        'recall': []
    }

    # Random number generator (fixed seed for reproducibility)
    rng = np.random.default_rng(seed=0)

    # Loop through each shift intensity
    for sigma in shift_levels:
        print(f"   → Testing σ = {sigma:4.2f}...", end=" ")

        # Apply covariate shift to test data
        if sigma == 0:
            X_test_shifted = X_test.copy()
        else:
            # Add Gaussian noise to features
            # This simulates: sensor degradation, measurement errors, domain shift
            X_test_shifted = X_test + rng.normal(loc=0, scale=sigma, size=X_test.shape)

        # Predict on shifted test data (using FIXED trained model)
        y_pred = svm.predict(X_test_shifted)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # Store results
        results['shift_sigma'].append(sigma)
        results['accuracy'].append(acc)
        results['f1_score'].append(f1)
        results['precision'].append(prec)
        results['recall'].append(rec)

        print(f"Accuracy: {acc:.4f}")

    # ========== STEP 5: Convert to DataFrame ==========
    print("\n[5/5] Storing results...")
    df_results = pd.DataFrame(results)

    print("\n--- SVM Performance Across Covariate Shifts ---")
    print(df_results.to_string(index=False))

    # Identify performance cliff (where accuracy drops below 70% of baseline)
    cliff_threshold = 0.7 * baseline_accuracy
    cliff_idx = np.where(df_results['accuracy'] < cliff_threshold)[0]
    if len(cliff_idx) > 0:
        cliff_sigma = df_results['shift_sigma'].iloc[cliff_idx[0]]
        print(f"\nPerformance Cliff Detected at σ = {cliff_sigma}")
        print(f"   Accuracy dropped below 70% of baseline ({cliff_threshold:.4f})")
    else:
        print("\nNo severe performance cliff detected")

    return df_results, svm, X_train, X_test, y_train, y_test, X_scaled, y


# ============================================================================
# EXPERIMENT 2: RIDGE WITH LABEL NOISE
# ============================================================================

def ridge_label_noise_experiment():
    """
    Test Ridge Regression performance with label noise.

    Process:
        1. Load Diabetes dataset (442 samples, 10 features)
        2. Train Ridge on clean training data
        3. Corrupt test labels (y_test) by adding noise to random subset
        4. Measure how MSE/R² degrade as label corruption increases

    Returns:
        df_results: DataFrame with metrics for each noise level
        model: Trained Ridge model
        Additional data for visualization
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Ridge with Label Noise (Diabetes Dataset)")
    print("=" * 70)

    # ========== STEP 1: Load and Prepare Data ==========
    print("\n[1/5] Loading and preparing data...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target range: [{y.min():.1f}, {y.max():.1f}]")

    # ========== STEP 2: Find Best Alpha via Cross-Validation ==========
    print("\n[2/5] Finding best regularization parameter (alpha)...")
    alphas = np.logspace(-3, 4, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha = ridge_cv.alpha_
    print(f"   Best alpha: {best_alpha:.4f}")

    # ========== STEP 3: Train Baseline Ridge Model ==========
    print("\n[3/5] Training baseline Ridge model...")
    model = Ridge(alpha=best_alpha, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Test on clean data (no noise)
    y_pred_clean = model.predict(X_test_scaled)
    baseline_mse = mean_squared_error(y_test, y_pred_clean)
    baseline_r2 = r2_score(y_test, y_pred_clean)
    print(f"   Baseline MSE (no noise): {baseline_mse:.2f}")
    print(f"   Baseline R² (no noise): {baseline_r2:.4f}")

    # ========== STEP 4: Define Noise Levels ==========
    print("\n[4/5] Defining label noise levels...")
    noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(f"   Testing {len(noise_levels)} noise levels: {[f'{n:.0%}' for n in noise_levels]}")

    # ========== STEP 5: Test Across Noise Levels ==========
    print("\n[5/5] Testing Ridge across label noise levels...")

    # Storage for results
    results = {
        'noise_ratio': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2_score': []
    }

    # Random number generator (fixed seed for reproducibility)
    rng = np.random.default_rng(seed=0)

    # Loop through each noise level
    for noise_ratio in noise_levels:
        print(f"   → Testing {noise_ratio:4.0%} noise...", end=" ")

        # Create corrupted labels
        y_test_noisy = y_test.copy()

        if noise_ratio > 0:
            # Select random subset of labels to corrupt
            n_samples = len(y_test)
            n_corrupt = int(n_samples * noise_ratio)
            corrupt_idx = rng.choice(n_samples, size=n_corrupt, replace=False)

            # Add Gaussian noise scaled by label standard deviation
            # This simulates: annotation errors, measurement mistakes, data entry errors
            noise = rng.normal(loc=0, scale=y_test.std(), size=n_corrupt)
            y_test_noisy[corrupt_idx] += noise

        # Predict on test data (using FIXED trained model)
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics against NOISY labels
        # This measures: "How does label corruption affect perceived model performance?"
        mse = mean_squared_error(y_test_noisy, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_noisy, y_pred)
        r2 = r2_score(y_test_noisy, y_pred)

        # Store results
        results['noise_ratio'].append(noise_ratio)
        results['mse'].append(mse)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['r2_score'].append(r2)

        print(f"MSE: {mse:.2f}, R²: {r2:.4f}")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    print("\n--- Ridge Performance Across Label Noise ---")
    print(df_results.to_string(index=False))

    # Check for negative R² (model worse than predicting mean)
    negative_r2_idx = np.where(df_results['r2_score'] < 0)[0]
    if len(negative_r2_idx) > 0:
        first_negative = df_results['noise_ratio'].iloc[negative_r2_idx[0]]
        print(f"\nR² became negative at {first_negative:.0%} noise")
        print(f"   (Model worse than predicting mean)")

    return df_results, model, X_train_scaled, X_test_scaled, y_train, y_test


# ============================================================================
# VISUALIZATION: SVM DEGRADATION CURVES
# ============================================================================

def plot_svm_degradation(df_results, results_dir, save_filename='svm_degradation.png'):
    """
    Plot SVM performance degradation across shift intensities.

    Creates 4 subplots:
        - Accuracy vs Shift
        - F1 Score vs Shift
        - Precision vs Shift
        - Recall vs Shift

    Each plot includes:
        - Baseline reference line (green dashed)
        - 70% threshold line (red dotted)
        - Actual performance curve (colored solid)
    """
    print("\n[Visualization] Creating SVM degradation plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SVM Performance Degradation Under Covariate Shift\n(Breast Cancer Dataset)',
                 fontsize=16, fontweight='bold', y=0.995)

    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    titles = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
        # Plot metric vs shift
        ax.plot(df_results['shift_sigma'], df_results[metric],
                marker='o', linewidth=2.5, markersize=8, color=color, label=title)

        # Baseline reference line
        baseline = df_results[metric].iloc[0]
        ax.axhline(y=baseline, color='green', linestyle='--',
                   linewidth=1.5, alpha=0.6, label=f'Baseline: {baseline:.3f}')

        # 70% threshold line
        threshold = 0.7 * baseline
        ax.axhline(y=threshold, color='red', linestyle=':',
                   linewidth=1.5, alpha=0.6, label=f'70% Baseline: {threshold:.3f}')

        # Styling
        ax.set_xlabel('Shift Intensity (σ)', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} vs Covariate Shift', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim([max(0, df_results[metric].min() - 0.1),
                     min(1.05, df_results[metric].max() + 0.05)])

    plt.tight_layout()

    # Save to Results folder
    save_path = results_dir / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.show()


# ============================================================================
# VISUALIZATION: RIDGE DEGRADATION CURVES
# ============================================================================

def plot_ridge_degradation(df_results, results_dir, save_filename='ridge_degradation.png'):
    """
    Plot Ridge performance degradation with label noise.

    Creates 4 subplots:
        - MSE vs Label Noise
        - RMSE vs Label Noise
        - R² Score vs Label Noise (with R²=0 reference)
        - MAE vs Label Noise

    Each plot includes:
        - Baseline reference line (green dashed)
        - Actual performance curve (colored solid)
    """
    print("\n[Visualization] Creating Ridge degradation plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ridge Regression Performance Degradation Under Label Noise\n(Diabetes Dataset)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: MSE
    axes[0, 0].plot(df_results['noise_ratio'] * 100, df_results['mse'],
                    marker='s', linewidth=2.5, markersize=8, color='#e74c3c', label='MSE')
    axes[0, 0].axhline(y=df_results['mse'].iloc[0], color='green',
                       linestyle='--', linewidth=1.5, alpha=0.6,
                       label=f'Baseline: {df_results["mse"].iloc[0]:.2f}')
    axes[0, 0].set_xlabel('Label Noise (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('MSE vs Label Noise', fontsize=12, pad=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].legend(loc='best', fontsize=9)

    # Plot 2: RMSE
    axes[0, 1].plot(df_results['noise_ratio'] * 100, df_results['rmse'],
                    marker='s', linewidth=2.5, markersize=8, color='#e67e22', label='RMSE')
    axes[0, 1].axhline(y=df_results['rmse'].iloc[0], color='green',
                       linestyle='--', linewidth=1.5, alpha=0.6,
                       label=f'Baseline: {df_results["rmse"].iloc[0]:.2f}')
    axes[0, 1].set_xlabel('Label Noise (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Root Mean Squared Error', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('RMSE vs Label Noise', fontsize=12, pad=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes[0, 1].legend(loc='best', fontsize=9)

    # Plot 3: R² Score
    axes[1, 0].plot(df_results['noise_ratio'] * 100, df_results['r2_score'],
                    marker='s', linewidth=2.5, markersize=8, color='#9b59b6', label='R² Score')
    axes[1, 0].axhline(y=df_results['r2_score'].iloc[0], color='green',
                       linestyle='--', linewidth=1.5, alpha=0.6,
                       label=f'Baseline: {df_results["r2_score"].iloc[0]:.3f}')
    axes[1, 0].axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.6,
                       label='R²=0 (worse than mean)')
    axes[1, 0].set_xlabel('Label Noise (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('R² Score vs Label Noise', fontsize=12, pad=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].legend(loc='best', fontsize=9)

    # Plot 4: MAE
    axes[1, 1].plot(df_results['noise_ratio'] * 100, df_results['mae'],
                    marker='s', linewidth=2.5, markersize=8, color='#16a085', label='MAE')
    axes[1, 1].axhline(y=df_results['mae'].iloc[0], color='green',
                       linestyle='--', linewidth=1.5, alpha=0.6,
                       label=f'Baseline: {df_results["mae"].iloc[0]:.2f}')
    axes[1, 1].set_xlabel('Label Noise (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('MAE vs Label Noise', fontsize=12, pad=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes[1, 1].legend(loc='best', fontsize=9)

    plt.tight_layout()

    # Save to Results folder
    save_path = results_dir / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for Phase 1: Baseline Performance Analysis

    This function orchestrates the complete Phase 1 workflow:
        1. Setup Results folder
        2. Run SVM covariate shift experiment
        3. Visualize and save SVM results
        4. Run Ridge label noise experiment
        5. Visualize and save Ridge results
        6. Print summary statistics
    """
    print("\n" + "=" * 70)
    print("        PHASE 1: BASELINE DEGRADATION ANALYSIS")
    print("     CSI 536 - Robust Linear Models under Distribution Shift")
    print("=" * 70)

    # ========== Setup Results Folder ==========
    results_dir = setup_results_folder()

    # ========== Experiment 1: SVM ==========
    svm_results, svm_model, X_train_svm, X_test_svm, y_train_svm, y_test_svm, X_scaled, y = \
        svm_covariate_shift_experiment()

    # Visualize SVM results
    plot_svm_degradation(svm_results, results_dir)

    # Save SVM results to Results folder
    svm_csv_path = results_dir / 'svm_degradation_results.csv'
    svm_results.to_csv(svm_csv_path, index=False)
    print(f"\nSVM results saved to: {svm_csv_path}")

    # ========== Experiment 2: Ridge ==========
    ridge_results, ridge_model, X_train_ridge, X_test_ridge, y_train_ridge, y_test_ridge = \
        ridge_label_noise_experiment()

    # Visualize Ridge results
    plot_ridge_degradation(ridge_results, results_dir)

    # Save Ridge results to Results folder
    ridge_csv_path = results_dir / 'ridge_degradation_results.csv'
    ridge_results.to_csv(ridge_csv_path, index=False)
    print(f"\nRidge results saved to: {ridge_csv_path}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("                    PHASE 1 COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {results_dir.absolute()}")
    print("\nGenerated Files:")
    print(f"   1. {results_dir}/svm_degradation.png - SVM performance curves")
    print(f"   2. {results_dir}/ridge_degradation.png - Ridge performance curves")
    print(f"   3. {results_dir}/svm_degradation_results.csv - SVM numerical results")
    print(f"   4. {results_dir}/ridge_degradation_results.csv - Ridge numerical results")

    print("\nKey Findings:")
    print(f"\n   SVM (Covariate Shift):")
    print(f"   - Baseline Accuracy: {svm_results['accuracy'].iloc[0]:.4f}")
    print(f"   - Worst Accuracy (σ={svm_results['shift_sigma'].iloc[-1]}): {svm_results['accuracy'].iloc[-1]:.4f}")
    print(
        f"   - Performance Drop: {(1 - svm_results['accuracy'].iloc[-1] / svm_results['accuracy'].iloc[0]) * 100:.1f}%")

    print(f"\n   Ridge (Label Noise):")
    print(f"   - Baseline MSE: {ridge_results['mse'].iloc[0]:.2f}")
    print(f"   - Worst MSE ({ridge_results['noise_ratio'].iloc[-1]:.0%} noise): {ridge_results['mse'].iloc[-1]:.2f}")
    print(f"   - MSE Increase: {(ridge_results['mse'].iloc[-1] / ridge_results['mse'].iloc[0] - 1) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  → Analyze degradation curves in Results folder")
    print("  → Identify performance cliffs")
    print("  → Move to Phase 2: Implement robustification techniques")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()