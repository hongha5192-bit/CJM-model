"""
Lambda scan for small lambda values (1, 2, 4) with URSI features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations
from tqdm import tqdm
from jumpmodels.jump import JumpModel
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def best_balanced_accuracy_K3(s_pred, s_true):
    """Compute best balanced accuracy over all 6 permutations for K=3"""
    K = 3
    best_bac = 0

    for perm in permutations(range(K)):
        mapping = {old: new for old, new in enumerate(perm)}
        s_pred_mapped = pd.Series(s_pred).map(mapping).values
        bac = balanced_accuracy_score(s_true, s_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac

def fit_cjm_single_sim(sim_data, lambda_val, K=3):
    """Fit CJM on a single simulation with given lambda"""
    try:
        # Extract all 8 features including URSI breadth indicators
        feature_names = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP', 'URSI_0_50', 'URSI_0_20']
        X_raw = sim_data[feature_names].values

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Get returns and true states
        ret_ser = sim_data['returns'].values
        s_true = sim_data['s_true'].values

        # Fit CJM
        model = JumpModel(
            n_components=K,
            jump_penalty=lambda_val,
            cont=True,
            mode_loss=True,
            grid_size=0.05,
            n_init=3,
            max_iter=500,
            tol=1e-5,
            random_state=123,
            verbose=0
        )

        model.fit(X, ret_ser=ret_ser, sort_by='cumret')
        s_pred = model.predict(X)

        # Calculate best BAC
        bac = best_balanced_accuracy_K3(s_pred, s_true)
        return bac

    except Exception as e:
        logger.error(f"Error fitting simulation: {e}")
        return np.nan

def main():
    logger.info("="*80)
    logger.info("LAMBDA SCAN FOR EXTREMELY SMALL VALUES (0.01, 0.05)")
    logger.info("="*80)

    # Extremely small lambda values to test
    lambda_values = [0.01, 0.05]
    n_simulations = 20  # Use 20 simulations for quicker testing

    logger.info(f"Testing lambda values: {lambda_values}")
    logger.info(f"Using {n_simulations} simulations")
    logger.info("Features: ADX, DMI_Plus, DMI_Minus, URSI, BB_PCTB, BBWP, URSI_0_50, URSI_0_20")

    # Load simulations
    sim_dir = Path('outputs/step2_sim/K3_with_ursi_breadth')
    sim_files = sorted(sim_dir.glob('sim_*.parquet'))[:n_simulations]

    if not sim_files:
        raise ValueError(f"No simulation files found in {sim_dir}")

    logger.info(f"\nLoading {len(sim_files)} simulations...")
    simulations = []
    for f in sim_files:
        df = pd.read_parquet(f)
        simulations.append(df)

    # Run lambda scan
    results = []
    logger.info("\nStarting lambda scan...")

    for lambda_val in lambda_values:
        logger.info(f"\nProcessing λ={lambda_val}...")
        bac_scores = []

        for i, sim_data in enumerate(tqdm(simulations, desc=f"λ={lambda_val}")):
            bac = fit_cjm_single_sim(sim_data, lambda_val)
            bac_scores.append(bac)

        # Calculate statistics
        valid_bacs = [b for b in bac_scores if not np.isnan(b)]
        if valid_bacs:
            mean_bac = np.mean(valid_bacs)
            std_bac = np.std(valid_bacs)
            logger.info(f"  Result: Mean BAC = {mean_bac:.4f} ± {std_bac:.4f} ({len(valid_bacs)}/{len(bac_scores)} valid)")
        else:
            mean_bac = std_bac = np.nan
            logger.info(f"  Result: All simulations failed")

        results.append({
            'lambda': lambda_val,
            'mean_bac': mean_bac,
            'std_bac': std_bac,
            'n_valid': len(valid_bacs),
            'n_total': len(bac_scores)
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Compare with previous lambda=5 result
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH PREVIOUS RESULTS")
    logger.info("="*80)

    print("\nExtremely Small Lambda Values (0.01, 0.05):")
    print(results_df[['lambda', 'mean_bac', 'std_bac']].to_string(index=False))

    print("\nPrevious best results:")
    print("  λ=0.5, BAC=0.5879 ± 0.0621 (20 simulations)")
    print("  λ=2, BAC=0.5736 ± 0.0643 (20 simulations)")
    print("  λ=5, BAC=0.5605 ± 0.0625 (50 simulations)")

    # Find best lambda
    if results_df['mean_bac'].notna().any():
        best_idx = results_df['mean_bac'].idxmax()
        best_lambda = results_df.loc[best_idx, 'lambda']
        best_bac = results_df.loc[best_idx, 'mean_bac']
        best_std = results_df.loc[best_idx, 'std_bac']

        logger.info(f"\nBest lambda from small values: {best_lambda}")
        logger.info(f"Best BAC: {best_bac:.4f} ± {best_std:.4f}")

        # Save results
        output_dir = Path('outputs/step3_lambda')
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_dir / 'lambda_scores_small_values.csv', index=False)
        logger.info(f"\nResults saved to: {output_dir / 'lambda_scores_small_values.csv'}")

        # Create comparison plot
        import matplotlib.pyplot as plt

        # Combine with previous results for plotting
        all_lambdas = [0.01, 0.05, 0.1, 0.5, 0.8, 1, 2, 4, 5, 10, 15, 20, 50, 100]
        # New results + all previous results
        all_bacs = list(results_df['mean_bac'].values) + [0.5815, 0.5879, 0.5803, 0.5722, 0.5736, 0.5698, 0.5605, 0.5460, 0.5391, 0.5302, 0.5069, 0.4769]
        all_stds = list(results_df['std_bac'].values) + [0.0599, 0.0621, 0.0652, 0.0663, 0.0643, 0.0661, 0.0625, 0.0646, 0.0607, 0.0619, 0.0598, 0.0512]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot all points
        ax.errorbar(all_lambdas[:2], all_bacs[:2], yerr=all_stds[:2],
                   marker='s', markersize=12, linewidth=2, capsize=5,
                   color='green', label='New extremely small λ', alpha=0.9)

        ax.errorbar(all_lambdas[2:], all_bacs[2:], yerr=all_stds[2:],
                   marker='o', markersize=10, linewidth=2, capsize=5,
                   color='blue', label='Previous λ values', alpha=0.7)

        # Highlight best point overall
        best_overall_idx = np.argmax(all_bacs)
        ax.scatter(all_lambdas[best_overall_idx], all_bacs[best_overall_idx],
                  s=300, marker='*', color='gold', edgecolor='darkred', linewidth=2,
                  zorder=5, label=f'Best: λ={all_lambdas[best_overall_idx]}')

        # Add value labels
        for lam, bac in zip(all_lambdas[:2], all_bacs[:2]):
            ax.annotate(f'{bac:.3f}', xy=(lam, bac), xytext=(0, 8),
                       textcoords='offset points', ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

        ax.axhline(y=0.333, linestyle='--', color='red', alpha=0.3, label='Random')
        ax.axhline(y=0.5, linestyle=':', color='gray', alpha=0.3)

        ax.set_xlabel('Lambda (λ)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Balanced Accuracy (BAC)', fontsize=12, fontweight='bold')
        ax.set_title('Extended Lambda Scan: Including Small Values (1, 2, 4)', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(0.8, 120)
        ax.set_ylim(0.45, 0.65)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig('outputs/lambda_scan_extended.png', dpi=120, bbox_inches='tight')
        logger.info(f"Plot saved to: outputs/lambda_scan_extended.png")

    logger.info("\n" + "="*80)

if __name__ == "__main__":
    main()