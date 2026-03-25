from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def SVMBreast():
    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Introduce Gaussian Shift
    shift_strength = 0.5 
    rng = np.random.default_rng(seed=0)  # reproducible
    X_test_shifted = X_test + rng.normal(loc=shift_strength, scale=0.5, size=X_test.shape)

    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test_shifted)
    #y_pred_modified = svm.predict(X_test_shifted)

    # Reduce to 2D with PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_pca.fit(X_train_pca, y_train)

    # Decision boundary mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes:
        ax.set_facecolor('#1a1d2e')



    # Plot 1: Decision boundary
    axes[0].contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm')
    axes[0].contour(xx, yy, Z, colors='white', linewidths=0.8, alpha=0.5)
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                            cmap='coolwarm', edgecolors='white', linewidths=0.3, s=30, alpha=0.85)
    axes[0].set_title('SVM Decision Boundary (PCA)', color='white', fontsize=13, pad=10)
    axes[0].set_xlabel('Principal Component 1', color='#aaaacc')
    axes[0].set_ylabel('Principal Component 2', color='#aaaacc')
    axes[0].tick_params(colors='#888899')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#333355')
    legend = axes[0].legend(handles=[
        mpatches.Patch(color='#3b82f6', label='Malignant'),
        mpatches.Patch(color='#ef4444', label='Benign')
    ], facecolor='#1a1d2e', edgecolor='#333355', labelcolor='white')

    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    im = axes[1].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[1].set_title('Confusion Matrix', color='white', fontsize=13, pad=10)
    axes[1].set_xlabel('Predicted Label', color='#aaaacc')
    axes[1].set_ylabel('True Label', color='#aaaacc')
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Malignant', 'Benign'], color='#aaaacc')
    axes[1].set_yticklabels(['Malignant', 'Benign'], color='#aaaacc', rotation=90, va='center')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#333355')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else '#333355', fontsize=18, fontweight='bold')

    acc = (y_pred == y_test).mean()
    fig.text(0.5, 0.01, f'Test Accuracy: {acc:.1%}', ha='center', color='#60a5fa', fontsize=12, fontweight='bold')



    plt.tight_layout()
    plt.show()

def diaRR():
    # Loads and splits the data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    shift_strength = 1.5
    rng = np.random.default_rng(seed=0)
    X_test_shifted = X_test_scaled + rng.normal(loc=shift_strength, scale=0.5, size=X_test_scaled.shape)

    # Find best alpha value
    alphas = np.logspace(-3, 4, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha = ridge_cv.alpha_

    # Fit final model
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_shifted = model.predict(X_test_shifted)  # shifted
    rmse_orig   = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_new = np.sqrt(mean_squared_error(y_test, y_pred_shifted))
    r2_orig     = r2_score(y_test, y_pred)
    r2_new = r2_score(y_test, y_pred_shifted)


    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ridge Regression — Diabetes Dataset", fontsize=14, fontweight="bold")

    # (a) Actual vs Predicted
    for ax, preds, label, color in zip(
        axes[:2],
        [y_pred, y_pred_shifted],
        [f"Original  R²={r2_orig:.3f}  RMSE={rmse_orig:.1f}",
        f"Shifted   R²={r2_new:.3f}  RMSE={rmse_new:.1f}"],
        ["steelblue", "tomato"]
    ):
        ax.scatter(y_test, preds, alpha=0.6, color=color, edgecolors="k", linewidths=0.4)
        lims = [min(y_test.min(), preds.min()) - 10, max(y_test.max(), preds.max()) + 10]
        ax.plot(lims, lims, "r--", lw=1.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(label)

    #ax = axes[2]
    #rmses = []
    #for a in alphas:
        #m = Ridge(alpha=a).fit(X_train_scaled, y_train)
        #rmses.append(np.sqrt(mean_squared_error(y_test, m.predict(X_test_scaled))))
    #ax.semilogx(alphas, rmses, color="steelblue")
    #ax.axvline(best_alpha, color="red", linestyle="--", label=f"Best α={best_alpha:.3f}")
    #ax.set_xlabel("Alpha (log scale)"); ax.set_ylabel("Test RMSE")
    #ax.set_title("Regularisation Path"); ax.legend()

    plt.show()



def main():
    #SVMBreast()
    diaRR()

if __name__ == "__main__":
    main()
