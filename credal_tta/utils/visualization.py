"""
Visualization Utilities - Generate Paper Figures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_synthetic_comparison(
    time_series,
    shift_point,
    standard_preds,
    credal_preds,
    diagnostics,
    save_path="figures/synthetic_comparison.pdf"
):
    """
    Figure 3: Context inertia elimination on SinFreq dataset
    
    Three-panel plot:
    - Top: Ground truth with shift annotation
    - Middle: Predictions comparison
    - Bottom: Credal set diameter evolution
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    t = np.arange(len(time_series))
    
    # Panel 1: Ground truth
    axes[0].plot(t, time_series, 'k-', alpha=0.7, label='Ground Truth')
    axes[0].axvline(shift_point, color='red', linestyle='--', 
                   linewidth=2, label='Regime Shift')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].set_title('SinFreq Dataset: Frequency Shift', fontsize=14, fontweight='bold')
    
    # Panel 2: Predictions
    axes[1].plot(t[1:], standard_preds, color='orange', alpha=0.7, 
                linewidth=2, label='Standard Chronos')
    axes[1].plot(t[1:], credal_preds, color='blue', alpha=0.8, 
                linewidth=2, label='Credal-Chronos')
    
    # Detect reset point from diagnostics
    reset_points = [d['t'] for d in diagnostics if d.get('reset_occurred', False)]
    if reset_points:
        axes[1].scatter(reset_points, [credal_preds[r-1] for r in reset_points if r < len(credal_preds)],
                       marker='^', s=150, color='green', 
                       label='Reset Triggered', zorder=5)
    
    axes[1].axvline(shift_point, color='red', linestyle='--', 
                   linewidth=2, alpha=0.5)
    axes[1].set_ylabel('Prediction', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].set_title('Prediction Comparison', fontsize=14, fontweight='bold')
    
    # Panel 3: Credal set diameter
    diameters = [d['diameter'] for d in diagnostics]
    axes[2].plot(t[:len(diameters)], diameters, color='purple', 
                linewidth=2, label='Credal Set Diameter')
    axes[2].axvline(shift_point, color='red', linestyle='--', 
                   linewidth=2, alpha=0.5)
    axes[2].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Diameter', fontsize=12)
    axes[2].set_xlabel('Time Step', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].set_title('Epistemic Uncertainty (Hausdorff Diameter)', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_recovery_time_comparison(
    results_dict,
    save_path="figures/recovery_time.pdf"
):
    """
    Figure 4: Architecture generalization (recovery time bars)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results_dict.keys())
    architectures = ['Chronos', 'Moirai', 'PatchTST']
    
    x = np.arange(len(architectures))
    width = 0.25
    
    # Assume results_dict structure: {method: {arch: RT_value}}
    for i, method in enumerate(['Standard', 'Credal-TTA']):
        values = [results_dict.get(method, {}).get(arch, 0) for arch in architectures]
        color = 'gray' if method == 'Standard' else 'blue'
        ax.bar(x + i * width, values, width, 
               label=method, color=color, alpha=0.7)
    
    ax.set_xlabel('Architecture', fontsize=12)
    ax.set_ylabel('Recovery Time (steps)', fontsize=12)
    ax.set_title('Credal-TTA Generalization Across Architectures', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(architectures)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_ablation_heatmap(
    results_matrix,
    row_labels,
    col_labels,
    save_path="figures/ablation_heatmap.pdf"
):
    """
    Heatmap for ablation results
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(results_matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{results_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title("Ablation Study Results", fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='RMSE')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from credal_tta.utils.data_loader import generate_sin_freq_shift
    from credal_tta.credal_tta import CredalTTA
    from credal_tta.models.wrappers import NaiveBaseline
    
    # Generate synthetic data
    data, shift = generate_sin_freq_shift(T=1000, shift_point=500)
    
    # Standard predictions
    model = NaiveBaseline()
    standard = []
    context = []
    for x in data:
        context.append(x)
        if len(context) > 512:
            context = context[-512:]
        standard.append(model.predict(np.array(context)))
    
    # Credal predictions
    adapter = CredalTTA(model=model, K=3, lambda_reset=1.2)
    credal, diag = adapter.predict_sequence(data, return_diagnostics=True)
    
    # Plot
    plot_synthetic_comparison(
        time_series=data,
        shift_point=shift,
        standard_preds=np.array(standard)[:-1],
        credal_preds=credal[:-1],
        diagnostics=diag,
        save_path="test_figure.pdf"
    )
