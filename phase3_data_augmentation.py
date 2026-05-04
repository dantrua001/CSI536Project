"""
Phase 3: Data Augmentation for Ridge Regression Robustness
CSI 536 - Robust Linear Models under Distribution Shift
Group 2: Mehak Seth, Daniel Truax, Juhan Choi

This script implements data augmentation and robust regression methods to improve
Ridge Regression robustness against label noise.

Methods Implemented:
1. Baseline Ridge (Standard) - No augmentation/robustification
2. Ridge + Noise Injection Augmentation - Train with augmented noisy samples
3. Huber Regression - Robust loss function (less sensitive to outliers)
4. Ridge + Bagging - Bootstrap aggregation for stability

Dataset: Diabetes (sklearn)
Shift Type: Label noise (random corruption of y_test)
Noise Levels: [0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%]

EVALUATION APPROACH:
- Train all models on CLEAN training data
- Evaluate on NOISY test labels (distribution shift)
- Measure how performance degrades when test distribution differs from training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


# ============================================================================
# DATA AUGMENTATION & ROBUST REGRESSION METHODS
# ============================================================================

class BaselineRidge:
    """Baseline: Standard Ridge Regression with no robustification."""

    def __init__(self, alpha=1.0):
        self.name = "Baseline Ridge"
        self.alpha = alpha
        self.model = None

    def fit(self, X_train, y_train):
        """Train standard Ridge regression."""
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)


class NoiseInjectionRidge:
    """Ridge with training data augmentation via noise injection."""

    def __init__(self, alpha=1.0, noise_rates=[0.05, 0.10, 0.15], augmentation_factor=2):
        self.name = "Ridge + Noise Injection"
        self.alpha = alpha
        self.noise_rates = noise_rates
        self.augmentation_factor = augmentation_factor
        self.model = None

    def fit(self, X_train, y_train):
        """
        Train Ridge with augmented training data.

        Strategy:
        1. Keep original clean training data
        2. Create augmented copies with varying levels of label noise
        3. Train on combined dataset (clean + noisy)

        This forces the model to be robust to label corruption.
        """
        X_augmented = [X_train.copy()]
        y_augmented = [y_train.copy()]

        # Generate augmented samples with different noise levels
        for noise_rate in self.noise_rates:
            for _ in range(self.augmentation_factor):
                # Create noisy copy of labels
                y_noisy = y_train.copy()

                # Add Gaussian noise to labels
                # Noise magnitude = noise_rate * std(y_train)
                noise_std = noise_rate * np.std(y_train)
                noise = np.random.normal(0, noise_std, size=len(y_train))
                y_noisy = y_noisy + noise

                X_augmented.append(X_train.copy())
                y_augmented.append(y_noisy)

        # Combine all data
        X_combined = np.vstack(X_augmented)
        y_combined = np.hstack(y_augmented)

        # Train on augmented dataset
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.model.fit(X_combined, y_combined)

        print(f"  [Augmentation] Original: {len(X_train)} → Augmented: {len(X_combined)} samples")

        return self

    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)


class HuberRobust:
    """Huber Regression - robust to outliers in labels."""

    def __init__(self, epsilon=1.35, alpha=1.0):
        self.name = "Huber Regression"
        self.epsilon = epsilon
        self.alpha = alpha
        self.model = None

    def fit(self, X_train, y_train):
        """
        Train Huber regressor.

        Huber loss is a hybrid:
        - Quadratic (L2) for small errors → smooth, efficient
        - Linear (L1) for large errors → robust to outliers

        The 'epsilon' parameter controls the transition point.
        epsilon=1.35 is standard (gives 95% efficiency vs OLS for normal data)
        """
        self.model = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha, max_iter=1000)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)


class RidgeBagging:
    """Ridge Regression with Bootstrap Aggregation (Bagging)."""

    def __init__(self, alpha=1.0, n_estimators=10, max_samples=0.8):
        self.name = "Ridge + Bagging"
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.model = None

    def fit(self, X_train, y_train):
        """
        Train ensemble of Ridge models via bagging.

        Strategy:
        1. Train multiple Ridge models on random subsets (bootstrap samples)
        2. Average predictions across all models
        3. Reduces variance and improves robustness

        Bagging helps because:
        - Different bootstrap samples may exclude different noisy labels
        - Averaging reduces impact of any single noisy prediction
        """
        base_estimator = Ridge(alpha=self.alpha, random_state=42)
        self.model = BaggingRegressor(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

def add_label_noise(y, noise_rate, seed=None):
    """
    Add label noise by corrupting a percentage of labels.

    For regression: Add Gaussian noise to labels
    Noise magnitude scales with label standard deviation
    """
    if seed is not None:
        np.random.seed(seed)

    if noise_rate == 0:
        return y.copy()

    y_noisy = y.copy()
    n_corrupt = int(noise_rate * len(y))

    if n_corrupt > 0:
        # Select random indices to corrupt
        corrupt_idx = np.random.choice(len(y), n_corrupt, replace=False)

        # Add Gaussian noise scaled by label std
        noise_std = np.std(y)
        noise = np.random.normal(0, noise_std, size=n_corrupt)
        y_noisy[corrupt_idx] += noise

    return y_noisy


def evaluate_model(model, X_test, y_test_noisy):
    """
    Evaluate model on noisy test data.

    CRITICAL: We evaluate against NOISY labels (y_test_noisy) to measure
    how performance degrades when test distribution has label noise.

    This simulates the distribution shift scenario:
    - Training distribution: Clean labels
    - Test distribution: Noisy labels (shifted!)
    - Question: How much does performance degrade?
    """
    y_pred = model.predict(X_test)

    metrics = {
        'mse': mean_squared_error(y_test_noisy, y_pred),
        'r2': r2_score(y_test_noisy, y_pred),
        'mae': mean_absolute_error(y_test_noisy, y_pred)
    }

    return metrics


def run_experiment(X_train, X_test, y_train, y_test, noise_levels):
    """
    Run full experiment across all methods and noise levels.

    Returns:
        results_df: DataFrame with all results
    """
    # Initialize methods
    methods = [
        BaselineRidge(alpha=1.0),
        NoiseInjectionRidge(alpha=1.0, noise_rates=[0.05, 0.10, 0.15], augmentation_factor=2),
        HuberRobust(epsilon=1.35, alpha=1.0),
        RidgeBagging(alpha=1.0, n_estimators=10, max_samples=0.8)
    ]

    # Storage for results
    results = []

    print("=" * 80)
    print("PHASE 3: DATA AUGMENTATION FOR RIDGE REGRESSION ROBUSTNESS")
    print("=" * 80)
    print(f"Dataset: Diabetes (n_train={len(X_train)}, n_test={len(X_test)})")
    print(f"Noise levels: {[int(n * 100) for n in noise_levels]}%")
    print(f"Methods: {[m.name for m in methods]}")
    print("=" * 80)
    print("\nEvaluation Approach:")
    print("  - Train on CLEAN labels (training distribution)")
    print("  - Test on NOISY labels (test distribution - shifted!)")
    print("  - Measure degradation as noise increases")
    print("=" * 80)

    # Store baseline MSE at 0% noise for calculating percentage increase
    baseline_mse_clean = None

    # For each method
    for method in methods:
        print(f"\nMethod: {method.name}")
        print("-" * 80)

        # Train the model ONCE on clean training data
        # (Except NoiseInjection which augments during training)
        method.fit(X_train, y_train)

        # Evaluate across noise levels
        for noise_rate in noise_levels:
            # Add noise to TEST labels only (training is clean)
            # Using fixed seed for reproducibility
            y_test_noisy = add_label_noise(y_test, noise_rate, seed=0)

            # Evaluate (compare predictions to NOISY labels - distribution shift!)
            metrics = evaluate_model(method, X_test, y_test_noisy)

            # Store baseline MSE at 0% noise for reference
            if noise_rate == 0 and method.name == "Baseline Ridge":
                baseline_mse_clean = metrics['mse']

            # Calculate percentage increase over clean baseline
            if baseline_mse_clean is not None:
                mse_increase_pct = ((metrics['mse'] - baseline_mse_clean) / baseline_mse_clean) * 100
            else:
                mse_increase_pct = 0

            # Store results
            result_row = {
                'method': method.name,
                'noise_rate': noise_rate,
                'noise_pct': int(noise_rate * 100),
                'mse': metrics['mse'],
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'mse_increase_pct': mse_increase_pct
            }
            results.append(result_row)

            print(f"  {int(noise_rate * 100):2d}% noise: MSE={metrics['mse']:7.2f}, "
                  f"R²={metrics['r2']:6.4f}, MAE={metrics['mae']:6.2f}, "
                  f"MSE increase={mse_increase_pct:+6.1f}%")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_method_comparison(results_df, save_path):
    """
    4-panel comparison showing different aspects of performance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = [
        ('mse', 'Mean Squared Error (MSE)', False),
        ('r2', 'R² Score', True),
        ('mae', 'Mean Absolute Error (MAE)', False),
        ('mse_increase_pct', 'MSE Increase over Baseline (%)', False)
    ]

    colors = {
        'Baseline Ridge': 'gray',
        'Ridge + Noise Injection': 'steelblue',
        'Huber Regression': 'forestgreen',
        'Ridge + Bagging': 'coral'
    }

    markers = {
        'Baseline Ridge': 'o',
        'Ridge + Noise Injection': 's',
        'Huber Regression': '^',
        'Ridge + Bagging': 'D'
    }

    for idx, (metric, title, higher_better) in enumerate(metrics):
        ax = axes[idx]

        # Plot each method
        for method_name in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method_name]
            ax.plot(method_data['noise_pct'], method_data[metric],
                    marker=markers[method_name],
                    color=colors[method_name],
                    linewidth=2.5,
                    markersize=8,
                    label=method_name,
                    alpha=0.8)

        # Highlight 40% noise (the target improvement zone)
        ax.axvline(40, color='red', linestyle=':', linewidth=2, alpha=0.5)
        ax.text(40, ax.get_ylim()[0 if not higher_better else 1],
                '  Target\n  (40%)',
                ha='left', va='bottom' if not higher_better else 'top',
                fontsize=10, color='red', fontweight='bold')

        ax.set_xlabel('Label Noise (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} vs Label Noise', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 52)

    plt.suptitle('Phase 3: Data Augmentation Impact on Ridge Robustness',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved method comparison plot: {save_path}")
    plt.close()


def plot_robustness_curves(results_df, save_path):
    """
    Focus plot: MSE increase vs noise level.
    This directly shows progress toward the goal.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors = {
        'Baseline Ridge': 'gray',
        'Ridge + Noise Injection': 'steelblue',
        'Huber Regression': 'forestgreen',
        'Ridge + Bagging': 'coral'
    }

    markers = {
        'Baseline Ridge': 'o',
        'Ridge + Noise Injection': 's',
        'Huber Regression': '^',
        'Ridge + Bagging': 'D'
    }

    # Plot MSE for each method
    for method_name in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method_name]
        ax.plot(method_data['noise_pct'], method_data['mse'],
                marker=markers[method_name],
                color=colors[method_name],
                linewidth=3,
                markersize=10,
                label=method_name,
                alpha=0.8)

    # Highlight 40% noise zone
    ax.axvline(40, color='red', linestyle=':', linewidth=2.5, alpha=0.7)
    ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], xmin=0.7, xmax=0.85,
               alpha=0.1, color='red', label='Target Zone (40%)')

    ax.set_xlabel('Label Noise (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
    ax.set_title('Ridge Regression: MSE vs Label Noise\n(Lower is Better)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 52)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved robustness curves plot: {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Load and prepare data
    print("\nLoading Diabetes dataset...")
    X, y = load_diabetes(return_X_y=True)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Data loaded and preprocessed")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Target range: [{y.min():.1f}, {y.max():.1f}]")

    # Define noise levels (same as Phase 1)
    noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    # Run experiment
    results_df = run_experiment(X_train, X_test, y_train, y_test, noise_levels)

    # Create output directory
    import os
    os.makedirs('Results/Phase3', exist_ok=True)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_path = 'Results/Phase3/phase3_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved results CSV: {results_path}")

    # Generate plots
    plot_method_comparison(
        results_df,
        'Results/Phase3/augmentation_comparison.png'
    )

    plot_robustness_curves(
        results_df,
        'Results/Phase3/label_noise_robustness.png'
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: PERFORMANCE AT 40% LABEL NOISE (THE TARGET)")
    print("=" * 80)

    target_data = results_df[results_df['noise_pct'] == 40]
    baseline_mse = target_data[target_data['method'] == 'Baseline Ridge']['mse'].values[0]
    baseline_increase = target_data[target_data['method'] == 'Baseline Ridge']['mse_increase_pct'].values[0]

    print(f"\nBaseline Ridge at 40% noise:")
    print(f"  MSE: {baseline_mse:.2f}")
    print(f"  MSE increase over clean: +{baseline_increase:.1f}%")
    print(f"\n{'Method':<30} {'MSE':>10} {'MSE Increase':>15} {'Improvement':>15}")
    print("-" * 72)

    for method in results_df['method'].unique():
        method_mse = target_data[target_data['method'] == method]['mse'].values[0]
        method_increase = target_data[target_data['method'] == method]['mse_increase_pct'].values[0]
        improvement = baseline_increase - method_increase

        print(f"{method:<30} {method_mse:>10.2f} {method_increase:>14.1f}% {improvement:>14.1f}%")

    # Check if we met the goal
    print("\n" + "=" * 80)
    print("GOAL ASSESSMENT")
    print("=" * 80)
    print(f"Original Goal: Reduce MSE increase at 40% noise from +130% to <80%")
    print(f"Actual baseline result: +{baseline_increase:.1f}%")

    best_method = None
    best_increase = float('inf')

    for method in ['Ridge + Noise Injection', 'Huber Regression', 'Ridge + Bagging']:
        method_increase = target_data[target_data['method'] == method]['mse_increase_pct'].values[0]
        if method_increase < best_increase:
            best_increase = method_increase
            best_method = method

    print(f"\nBest performing method: {best_method}")
    print(f"MSE increase at 40% noise: +{best_increase:.1f}%")

    improvement = baseline_increase - best_increase
    print(f"Improvement over baseline: {improvement:.1f}%")

    if best_increase < 80:
        print(f"GOAL ACHIEVED! ({best_increase:.1f}% < 80%)")
    else:
        print(f"Goal not fully achieved, but showing {improvement:.1f}% improvement!")

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - Results/Phase3/phase3_results.csv")
    print("  - Results/Phase3/augmentation_comparison.png")
    print("  - Results/Phase3/label_noise_robustness.png")
    print("\nNext step: Review README_Phase3.md for interpretation guidance")
    print("=" * 80)


if __name__ == "__main__":
    main()