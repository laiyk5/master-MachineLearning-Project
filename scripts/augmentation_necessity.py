#!/usr/bin/env python3
"""Demonstrate the necessity of data augmentation by visualizing samples.

Trains two models (without and with augmentation) on the same data,
identifies samples that are misclassified without augmentation but
correctly classified with it, and visualizes them to show why
data augmentation is necessary.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_loader import get_train_test_split, load_digits_data, normalize_features
from src.classifiers import OneVsAllClassifier, PCATransform, get_classifier
from src.augmentation import augment_training_data
from src.utils import resolve_run_dir, setup_run_logging, get_script_output_dir


BASE_CONFIG = {
    'classifier_name': 'svm_rbf',
    'classifier_params': {'C': 4, 'gamma': 'scale'},
    'normalize': 'scale255',
    'pca_components': 35,
    'use_one_vs_all': True,
}


def _fit_predict(X_train, y_train, X_test, augment=0, augment_strength='mild'):
    """Fit model and return predictions."""
    X_tr = X_train.copy()
    y_tr = y_train.copy()

    if augment > 0:
        X_tr, y_tr = augment_training_data(X_tr, y_tr, n_augment=augment, strength=augment_strength)

    if BASE_CONFIG['normalize'] != 'none':
        X_tr, X_te = normalize_features(X_tr, X_test, method=BASE_CONFIG['normalize'])
    else:
        X_te = X_test.copy()

    if BASE_CONFIG['pca_components'] is not None:
        pca = PCATransform(n_components=BASE_CONFIG['pca_components'])
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

    if BASE_CONFIG['use_one_vs_all']:
        base_clf = lambda **kwargs: get_classifier(BASE_CONFIG['classifier_name'], **kwargs)
        clf = OneVsAllClassifier(base_clf, n_classes=10)
        clf.fit(X_tr, y_tr, **BASE_CONFIG['classifier_params'])
    else:
        clf = get_classifier(BASE_CONFIG['classifier_name'], **BASE_CONFIG['classifier_params'])
        clf.fit(X_tr, y_tr)

    return clf.predict(X_te)


def visualize_samples(images, labels, pred_no_aug, pred_aug, title, save_path):
    """Visualize a grid of samples with true label and predictions."""
    n = min(len(images), 20)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        ax = axes[i]
        img = images[i].reshape((28, 28), order='F')
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)

        true = labels[i]
        noaug = pred_no_aug[i]
        aug = pred_aug[i]

        color = 'green' if aug == true else 'red'
        ax.set_title(f"True: {true}\nNo aug: {noaug} | Aug: {aug}", fontsize=9, color=color)
        ax.axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main(output_dir: Path = None):
    print("=" * 80)
    print("Demonstrating Necessity of Data Augmentation")
    print("=" * 80)

    data_path = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"
    data = load_digits_data(data_path)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_fixed_samples = []

    for trial in [0, 1]:
        X_train, X_test, y_train, y_test = get_train_test_split(data, trial)

        # Get test indices to retrieve original images
        test_idx = data['testset'][trial, :] - 1
        # digits_vec is (784, 4000), transpose to (4000, 784)
        all_features = data['digits_vec'].T
        test_images = all_features[test_idx]

        print(f"\n--- Trial {trial + 1} ---")

        # Predictions without augmentation
        pred_no_aug = _fit_predict(X_train, y_train, X_test, augment=0)
        # Predictions with augmentation
        pred_aug = _fit_predict(X_train, y_train, X_test, augment=3, augment_strength='mild')

        # Categories
        no_aug_errors = pred_no_aug != y_test
        aug_correct = pred_aug == y_test
        fixed_by_aug = no_aug_errors & aug_correct
        still_wrong = no_aug_errors & (pred_aug != y_test)

        n_fixed = int(np.sum(fixed_by_aug))
        n_still = int(np.sum(still_wrong))
        n_total_errors = int(np.sum(no_aug_errors))

        print(f"  Total errors (no aug): {n_total_errors}")
        print(f"  Fixed by augmentation: {n_fixed} ({100*n_fixed/n_total_errors:.1f}%)")
        print(f"  Still wrong (both):    {n_still} ({100*n_still/n_total_errors:.1f}%)")

        # Collect fixed samples for combined visualization
        if n_fixed > 0:
            fixed_idx = np.where(fixed_by_aug)[0]
            for idx in fixed_idx:
                all_fixed_samples.append({
                    'image': test_images[idx],
                    'true': y_test[idx],
                    'pred_no_aug': pred_no_aug[idx],
                    'pred_aug': pred_aug[idx],
                    'trial': trial + 1,
                })

        # Visualize fixed samples per trial
        if n_fixed > 0:
            visualize_samples(
                test_images[fixed_by_aug],
                y_test[fixed_by_aug],
                pred_no_aug[fixed_by_aug],
                pred_aug[fixed_by_aug],
                title=f"Trial {trial + 1}: Samples Fixed by Data Augmentation",
                save_path=figures_dir / f"augmentation_fixed_trial{trial+1}.png",
            )

        # Visualize still-wrong samples per trial
        if n_still > 0:
            visualize_samples(
                test_images[still_wrong],
                y_test[still_wrong],
                pred_no_aug[still_wrong],
                pred_aug[still_wrong],
                title=f"Trial {trial + 1}: Still Misclassified (Even with Augmentation)",
                save_path=figures_dir / f"augmentation_still_wrong_trial{trial+1}.png",
            )

    # Combined visualization of all fixed samples
    if all_fixed_samples:
        n = min(len(all_fixed_samples), 20)
        cols = 5
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
        axes = axes.flatten() if n > 1 else [axes]

        for i in range(n):
            ax = axes[i]
            s = all_fixed_samples[i]
            img = s['image'].reshape((28, 28), order='F')
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(
                f"Trial {s['trial']}\nTrue: {s['true']} | No aug: {s['pred_no_aug']} | Aug: {s['pred_aug']}",
                fontsize=8,
                color='green',
            )
            ax.axis('off')

        for i in range(n, len(axes)):
            axes[i].axis('off')

        fig.suptitle(
            "Data Augmentation Fixes These Samples\n"
            "(Misclassified without augmentation, correctly classified with augmentation)",
            fontsize=13,
            fontweight='bold',
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = figures_dir / "augmentation_necessity_combined.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved combined figure: {save_path}")

        # Write summary report
        report_path = figures_dir / "augmentation_necessity_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("DEMONSTRATION: NECESSITY OF DATA AUGMENTATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model: SVM-RBF + PCA35 + C=4\n")
            f.write(f"Augmentation: mild geometric (rotation +/-5 deg, translation +/-1px, scale 0.95-1.05) x3\n\n")
            f.write(f"Total samples fixed by augmentation: {len(all_fixed_samples)}\n\n")
            for i, s in enumerate(all_fixed_samples[:30], 1):
                f.write(f"  {i}. Trial {s['trial']}: true={s['true']}, no_aug_pred={s['pred_no_aug']}, aug_pred={s['pred_aug']}\n")
        print(f"Saved report: {report_path}")

    print(f"\n{'=' * 80}")
    print("Analysis complete. Figures saved to:", figures_dir)
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    base_dir = resolve_run_dir(args.run_dir)
    output_dir = get_script_output_dir(base_dir, Path(__file__).stem)
    with setup_run_logging(output_dir):
        main(output_dir)
