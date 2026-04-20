"""Analyze class distribution in the MNIST dataset."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_digits_data, get_train_test_split
from src.utils import resolve_run_dir, setup_run_logging, get_script_output_dir


def analyze_class_distribution(data_path: str = None, save_path: str = None):
    """Analyze and visualize class distribution in train/test sets."""
    
    data = load_digits_data(data_path)
    
    # Get all labels
    all_labels = data['digits_labels'].flatten()  # (4000,)
    
    # Analyze both trials
    results = []
    
    for trial in [0, 1]:
        X_train, X_test, y_train, y_test = get_train_test_split(data, trial)
        
        # Count per class
        train_counts = np.bincount(y_train, minlength=10)
        test_counts = np.bincount(y_test, minlength=10)
        
        results.append({
            'trial': trial,
            'train_counts': train_counts,
            'test_counts': test_counts,
            'train_total': len(y_train),
            'test_total': len(y_test)
        })
    
    # Overall dataset stats
    overall_counts = np.bincount(all_labels, minlength=10)
    
    # Print report
    lines = []
    lines.append("=" * 70)
    lines.append("MNIST SUBSET - CLASS DISTRIBUTION ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total dataset size: {len(all_labels)} samples")
    lines.append("")
    lines.append("Overall class distribution:")
    for digit in range(10):
        count = overall_counts[digit]
        pct = 100 * count / len(all_labels)
        lines.append(f"  Digit {digit}: {count:4d} samples ({pct:5.2f}%)")
    lines.append("")
    
    # Per-trial breakdown
    for r in results:
        lines.append(f"--- Trial {r['trial'] + 1} ---")
        lines.append(f"Training set: {r['train_total']} samples")
        lines.append(f"Test set:     {r['test_total']} samples")
        lines.append("")
        lines.append("  Digit | Train | Test | Train% | Test%")
        lines.append("  ------|-------|------|--------|-------")
        for digit in range(10):
            train_c = r['train_counts'][digit]
            test_c = r['test_counts'][digit]
            train_pct = 100 * train_c / r['train_total']
            test_pct = 100 * test_c / r['test_total']
            lines.append(f"    {digit}   |  {train_c:3d}  |  {test_c:3d} | {train_pct:5.1f}% | {test_pct:5.1f}%")
        lines.append("")
    
    report = "\n".join(lines)
    print(report)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Overall distribution
    ax = axes[0, 0]
    digits = range(10)
    ax.bar(digits, overall_counts, color='steelblue', edgecolor='black')
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Overall Dataset Distribution (4000 samples)', fontsize=14)
    ax.set_xticks(digits)
    for i, v in enumerate(overall_counts):
        ax.text(i, v + 5, str(v), ha='center', fontsize=10)
    ax.set_ylim(0, max(overall_counts) * 1.1)
    
    # Plot 2: Per-trial train/test comparison (Trial 1)
    ax = axes[0, 1]
    x = np.arange(10)
    width = 0.35
    r = results[0]
    bars1 = ax.bar(x - width/2, r['train_counts'], width, label='Train', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, r['test_counts'], width, label='Test', color='lightcoral', edgecolor='black')
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Trial 1: Train vs Test Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.legend()
    
    # Plot 3: Per-trial train/test comparison (Trial 2)
    ax = axes[1, 0]
    r = results[1]
    bars1 = ax.bar(x - width/2, r['train_counts'], width, label='Train', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, r['test_counts'], width, label='Test', color='lightcoral', edgecolor='black')
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Trial 2: Train vs Test Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.legend()
    
    # Plot 4: Side-by-side comparison of both trials
    ax = axes[1, 1]
    r1_train_norm = results[0]['train_counts'] / results[0]['train_total'] * 100
    r1_test_norm = results[0]['test_counts'] / results[0]['test_total'] * 100
    r2_train_norm = results[1]['train_counts'] / results[1]['train_total'] * 100
    r2_test_norm = results[1]['test_counts'] / results[1]['test_total'] * 100
    
    ax.plot(digits, r1_train_norm, 'o-', label='Trial 1 Train', color='blue', linewidth=2)
    ax.plot(digits, r1_test_norm, 's--', label='Trial 1 Test', color='blue', alpha=0.7, linewidth=2)
    ax.plot(digits, r2_train_norm, 'o-', label='Trial 2 Train', color='red', linewidth=2)
    ax.plot(digits, r2_test_norm, 's--', label='Trial 2 Test', color='red', alpha=0.7, linewidth=2)
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Class Distribution Comparison (Normalized)', fontsize=14)
    ax.set_xticks(digits)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 15])
    
    plt.suptitle('MNIST Subset - Class Distribution Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization: {save_path}")
    
    # Save report
    if save_path:
        report_path = Path(save_path).parent / "class_distribution_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Saved report: {report_path}")
    
    return results, overall_counts


def main(output_dir: Path = None):
    data_path = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "figures"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    analyze_class_distribution(
        data_path,
        save_path=figures_dir / "class_distribution.png"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Result directory (auto-generated if omitted)")
    args = parser.parse_args()
    base_dir = resolve_run_dir(args.run_dir)
    output_dir = get_script_output_dir(base_dir, Path(__file__).stem)
    with setup_run_logging(output_dir):
        main(output_dir)
