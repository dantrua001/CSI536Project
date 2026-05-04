"""
Phase 2: Importance Weighting for SVM Robustness
CSI 536 - Robust Linear Models under Distribution Shift
Group 2: Mehak Seth, Daniel Truax, Juhan Choi

This script implements importance weighting methods to improve SVM robustness
against covariate shift (Gaussian noise added to test features).

Methods Implemented:
1. Uniform Weighting (Baseline) - No reweighting
2. Euclidean Distance Weighting - Weight by distance to test center
3. RBF Kernel Weighting - Weight by kernel similarity to test distribution
4. Density Ratio Estimation - Weight by train/test classification probability

Dataset: Breast Cancer (sklearn)
Shift Type: Covariate shift via Gaussian noise
Noise Levels (σ): [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


# ============================================================================
# IMPORTANCE WEIGHTING METHODS
# ============================================================================

class ImportanceWeighter:
    """Base class for importance weighting methods."""

    def __init__(self, name):
        self.name = name
        self.weights = None

    def fit(self, X_train, X_test_shifted):
        """Compute importance weights for training samples."""
        raise NotImplementedError

    def get_weights(self):
        """Return computed weights."""
        return self.weights


class UniformWeighting(ImportanceWeighter):
    """Baseline: All samples have equal weight."""

    def __init__(self):
        super().__init__("Uniform")

    def fit(self, X_train, X_test_shifted):
        """All weights = 1.0"""
        self.weights = np.ones(len(X_train))
        return self


class EuclideanWeighting(ImportanceWeighter):
    """Distance-based weighting using Euclidean distance to test center."""

    def __init__(self, bandwidth=1.0):
        super().__init__("Euclidean Distance")
        self.bandwidth = bandwidth

    def fit(self, X_train, X_test_shifted):
        """
        Weight samples by their distance to test distribution center.
        Closer samples get higher weight.

        Formula: w_i = exp(-distance_i^2 / (2 * bandwidth^2))
        """
        # Compute test distribution center
        test_center = np.mean(X_test_shifted, axis=0, keepdims=True)

        # Compute distances from each training sample to test center
        distances = euclidean_distances(X_train, test_center).flatten()

        # Convert distances to weights (exponential decay)
        self.weights = np.exp(-distances ** 2 / (2 * self.bandwidth ** 2))

        # Normalize weights to sum to N (preserves effective sample size)
        self.weights = self.weights * len(X_train) / np.sum(self.weights)

        return self


class RBFKernelWeighting(ImportanceWeighter):
    """Kernel-based weighting using RBF similarity to test distribution."""

    def __init__(self, gamma=0.1):
        super().__init__("RBF Kernel")
        self.gamma = gamma

    def fit(self, X_train, X_test_shifted):
        """
        Weight samples by RBF kernel similarity to test distribution.
        Works better than Euclidean for Gaussian noise shifts.

        Formula: w_i = mean(K(x_i, x_test_j)) for all test samples
        where K is the RBF kernel
        """
        # Compute RBF kernel between train and test samples
        # Shape: (n_train, n_test)
        kernel_matrix = rbf_kernel(X_train, X_test_shifted, gamma=self.gamma)

        # Average kernel similarity across all test samples
        # Each training sample gets weight = average similarity to test set
        self.weights = np.mean(kernel_matrix, axis=1)

        # Normalize weights
        self.weights = self.weights * len(X_train) / np.sum(self.weights)

        return self


class DensityRatioWeighting(ImportanceWeighter):
    """Density ratio estimation via probabilistic classification."""

    def __init__(self):
        super().__init__("Density Ratio")

    def fit(self, X_train, X_test_shifted):
        """
        Estimate density ratio p_test(x) / p_train(x) using classification.

        Method:
        1. Train classifier to distinguish train (label=0) vs test (label=1)
        2. Use predicted probabilities: w_i = P(test|x_i) / P(train|x_i)
        3. Clip extreme weights for stability
        """
        # Combine train and test data
        X_combined = np.vstack([X_train, X_test_shifted])
        y_combined = np.hstack([
            np.zeros(len(X_train)),  # Train samples = 0
            np.ones(len(X_test_shifted))  # Test samples = 1
        ])

        # Train logistic regression classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_combined, y_combined)

        # Get probabilities for training samples
        probs = clf.predict_proba(X_train)  # Shape: (n_train, 2)
        p_test = probs[:, 1]  # P(test|x)
        p_train = probs[:, 0]  # P(train|x)

        # Compute density ratio: w = p_test / p_train
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        self.weights = p_test / (p_train + epsilon)

        # Clip extreme weights (prevents instability)
        # Keep weights in [0.1, 10] range
        self.weights = np.clip(self.weights, 0.1, 10.0)

        # Normalize weights
        self.weights = self.weights * len(X_train) / np.sum(self.weights)

        return self


# ============================================================================
# WEIGHTED SVM TRAINING
# ============================================================================

def train_weighted_svm(X_train, y_train, weights, C=1.0):
    """
    Train SVM with importance weights.

    sklearn's SVC accepts sample_weight parameter to give higher
    importance to specific training samples.
    """
    model = SVC(C=C, kernel='rbf', gamma='scale', random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    return model


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

def evaluate_model(model, X_test_shifted, y_test):
    """Compute all metrics for a given model and test set."""
    y_pred = model.predict(X_test_shifted)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0)
    }

    return metrics


def run_experiment(X_train, X_test, y_train, y_test, noise_levels):
    """
    Run full experiment across all weighting methods and noise levels.

    Returns:
        results_df: DataFrame with all results
        all_weights: Dictionary of weight distributions for each method
    """
    # Initialize weighting methods
    methods = [
        UniformWeighting(),
        EuclideanWeighting(bandwidth=1.0),
        RBFKernelWeighting(gamma=0.1),
        DensityRatioWeighting()
    ]

    # Storage for results
    results = []
    all_weights = {}

    print("=" * 80)
    print("PHASE 2: IMPORTANCE WEIGHTING FOR SVM ROBUSTNESS")
    print("=" * 80)
    print(f"Dataset: Breast Cancer (n_train={len(X_train)}, n_test={len(X_test)})")
    print(f"Noise levels (σ): {noise_levels}")
    print(f"Methods: {[m.name for m in methods]}")
    print("=" * 80)

    # For each weighting method
    for method in methods:
        print(f"\nMethod: {method.name}")
        print("-" * 80)

        # Store weights at σ=1.5 for visualization (the cliff point)
        weights_at_cliff = None

        # For each noise level
        for sigma in noise_levels:
            # Generate noisy test data
            np.random.seed(0)  # Fixed seed for reproducibility
            noise = np.random.normal(0, sigma, X_test.shape)
            X_test_shifted = X_test + noise

            # Compute importance weights based on shifted test data
            method.fit(X_train, X_test_shifted)
            weights = method.get_weights()

            # Store weights at σ=1.5 for visualization
            if sigma == 1.5:
                weights_at_cliff = weights.copy()

            # Train weighted SVM
            model = train_weighted_svm(X_train, y_train, weights)

            # Evaluate
            metrics = evaluate_model(model, X_test_shifted, y_test)

            # Store results
            result_row = {
                'method': method.name,
                'sigma': sigma,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }
            results.append(result_row)

            print(f"  σ={sigma:.2f}: Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, "
                  f"Rec={metrics['recall']:.4f}")

        # Store weights for visualization
        if weights_at_cliff is not None:
            all_weights[method.name] = weights_at_cliff

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, all_weights


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_weight_distributions(all_weights, save_path):
    """
    Plot histogram of weight distributions for each method.
    Shows how different methods assign importance to training samples.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    method_names = ['Uniform', 'Euclidean Distance', 'RBF Kernel', 'Density Ratio']
    colors = ['gray', 'steelblue', 'forestgreen', 'coral']

    for idx, method_name in enumerate(method_names):
        ax = axes[idx]
        weights = all_weights[method_name]

        # Histogram
        ax.hist(weights, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')
        ax.axvline(np.mean(weights), color='red', linestyle='--',
                   linewidth=2, label=f'Mean = {np.mean(weights):.2f}')
        ax.axvline(np.median(weights), color='orange', linestyle='--',
                   linewidth=2, label=f'Median = {np.median(weights):.2f}')

        ax.set_title(f'{method_name} Weighting', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Weight', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Importance Weight Distributions at σ=1.5 (Performance Cliff)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved weight distributions plot: {save_path}")
    plt.close()


def plot_robustness_comparison(results_df, save_path):
    """
    Plot 4-panel comparison of all metrics across noise levels.
    Shows how each weighting method affects SVM robustness.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = ['accuracy', 'f1', 'precision', 'recall']
    titles = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    colors = {'Uniform': 'gray',
              'Euclidean Distance': 'steelblue',
              'RBF Kernel': 'forestgreen',
              'Density Ratio': 'coral'}
    markers = {'Uniform': 'o',
               'Euclidean Distance': 's',
               'RBF Kernel': '^',
               'Density Ratio': 'D'}

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        # Plot each method
        for method_name in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method_name]
            ax.plot(method_data['sigma'], method_data[metric],
                    marker=markers[method_name],
                    color=colors[method_name],
                    linewidth=2.5,
                    markersize=8,
                    label=method_name,
                    alpha=0.8)

        # Highlight the cliff point (σ=1.5)
        ax.axvline(1.5, color='red', linestyle=':', linewidth=2, alpha=0.5)
        ax.text(1.5, ax.get_ylim()[0] + 0.02, 'Cliff\n(σ=1.5)',
                ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')

        ax.set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'SVM {title} vs Covariate Shift', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 3.1)

    plt.suptitle('Phase 2: Importance Weighting Impact on SVM Robustness',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved robustness comparison plot: {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Load and prepare data
    print("\nLoading Breast Cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features (critical for SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Data loaded and preprocessed")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")

    # Define noise levels (same as Phase 1)
    noise_levels = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Run experiment
    results_df, all_weights = run_experiment(
        X_train, X_test, y_train, y_test, noise_levels
    )

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_path = 'Results/Phase2/phase2_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved results CSV: {results_path}")

    # Generate plots
    plot_weight_distributions(
        all_weights,
        'Results/Phase2/importance_weights_comparison.png'
    )

    plot_robustness_comparison(
        results_df,
        'Results/Phase2/svm_robustness_comparison.png'
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: IMPROVEMENT AT σ=1.5 (THE CLIFF)")
    print("=" * 80)

    cliff_data = results_df[results_df['sigma'] == 1.5]
    baseline_acc = cliff_data[cliff_data['method'] == 'Uniform']['accuracy'].values[0]

    print(f"\nBaseline (Uniform) Accuracy at σ=1.5: {baseline_acc:.4f}")
    print("\nImprovements over baseline:")

    for method in ['Euclidean Distance', 'RBF Kernel', 'Density Ratio']:
        method_acc = cliff_data[cliff_data['method'] == method]['accuracy'].values[0]
        improvement = (method_acc - baseline_acc) * 100
        print(f"  {method:20s}: {method_acc:.4f} ({improvement:+.2f}% improvement)")

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - Results/Phase2/phase2_results.csv")
    print("  - Results/Phase2/importance_weights_comparison.png")
    print("  - Results/Phase2/svm_robustness_comparison.png")
    print("\nNext step: Review README_Phase2.md for interpretation guidance")
    print("=" * 80)


if __name__ == "__main__":
    main()