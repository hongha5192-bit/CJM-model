#!/usr/bin/env python
"""
Step 3: Quick Empirical Lambda Scan for K=3
Uses 20 simulations and 7 lambda points for rapid completion
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from jumpmodels.jump import JumpModel
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations
import warnings
import time
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def best_balanced_accuracy_K3(y_true, y_pred):
    """Compute best balanced accuracy for K=3 with permutation fix"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    best_bac = 0.0
    for perm in permutations([0, 1, 2]):
        mapping = {0: perm[0], 1: perm[1], 2: perm[2]}
        y_pred_mapped = np.array([mapping[label] for label in y_pred])
        bac = balanced_accuracy_score(y_true, y_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac


def fit_cjm_to_simulation(sim_data, lam, feature_names):
    """Fit CJM to a single simulation"""
    try:
        # Extract features and true labels
        X = sim_data[feature_names].values
        s_true = sim_data['s_true'].values

        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit CJM with mode loss - ULTRA FAST settings
        cjm = JumpModel(
            n_components=3,
            jump_penalty=lam,
            cont=True,
            mode_loss=True,
            grid_size=0.05,  # K=3 optimized
            n_init=1,  # Minimal for speed
            max_iter=100,  # Minimal for speed
            tol=1e-4,  # Relaxed for speed
            random_state=42,
            verbose=0
        )

        cjm.fit(X_scaled)
        s_pred = cjm.predict(X_scaled)
        bac = best_balanced_accuracy_K3(s_true, s_pred)

        return bac

    except Exception as e:
        logger.debug(f"Failed to fit CJM: {e}")
        return np.nan


def main():
    """Quick lambda scan for demonstration"""

    # Load configuration
    config_path = 'configs/daily_K3.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    feature_names = config['features']['cols']

    # QUICK scan parameters
    lambda_grid = np.logspace(-1, 3, 7)  # 7 points: 0.1 to 1000
    n_simulations = 20  # Only 20 for quick results

    logger.info("="*80)
    logger.info("QUICK EMPIRICAL LAMBDA SCAN FOR K=3")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  K = 3")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  Lambda grid: {len(lambda_grid)} points")
    logger.info(f"  Lambda values: {[f'{lam:.1e}' for lam in lambda_grid]}")
    logger.info(f"  Using {n_simulations} simulations (quick mode)")

    # Load simulations
    sim_dir = Path('outputs/step2_sim/K3_features')
    sim_files = sorted(sim_dir.glob('sim_T*.parquet'))[:n_simulations]

    logger.info(f"\nLoading {len(sim_files)} simulations...")
    simulations = []
    for sim_file in sim_files:
        df = pd.read_parquet(sim_file)
        simulations.append(df)

    logger.info(f"Loaded {len(simulations)} simulations")

    # Lambda scan
    results = []
    start_time = time.time()

    for lambda_idx, lam in enumerate(lambda_grid):
        logger.info(f"\nProcessing λ = {lam:.1e} ({lambda_idx+1}/{len(lambda_grid)})...")

        bac_scores = []
        for sim_data in tqdm(simulations, desc=f"λ={lam:.1e}", leave=False):
            bac = fit_cjm_to_simulation(sim_data, lam, feature_names)
            if not np.isnan(bac):
                bac_scores.append(bac)

        if bac_scores:
            mean_bac = np.mean(bac_scores)
            std_bac = np.std(bac_scores)
            n_valid = len(bac_scores)
        else:
            mean_bac = np.nan
            std_bac = np.nan
            n_valid = 0

        results.append({
            'lambda': lam,
            'mean_bac': mean_bac,
            'std_bac': std_bac,
            'n_valid': n_valid,
            'n_total': len(simulations)
        })

        logger.info(f"  Result: BAC = {mean_bac:.4f} ± {std_bac:.4f} ({n_valid}/{len(simulations)} valid)")

    elapsed = time.time() - start_time
    logger.info(f"\nCompleted in {elapsed:.1f} seconds")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Find best lambda
    valid_results = df_results[~df_results['mean_bac'].isna()]

    if len(valid_results) > 0:
        best_idx = valid_results['mean_bac'].idxmax()
        best_lambda = valid_results.loc[best_idx, 'lambda']
        best_bac = valid_results.loc[best_idx, 'mean_bac']
        best_std = valid_results.loc[best_idx, 'std_bac']

        logger.info("\n" + "="*80)
        logger.info("LAMBDA SELECTION RESULTS")
        logger.info("="*80)
        logger.info(f"Best lambda: {best_lambda:.1e}")
        logger.info(f"Best mean BAC: {best_bac:.4f} ± {best_std:.4f}")

        # Display results table
        logger.info("\nFull Results:")
        logger.info("-" * 60)
        logger.info(f"{'Lambda':>10} | {'Mean BAC':>10} | {'Std BAC':>10} | {'Valid':>7}")
        logger.info("-" * 60)
        for _, row in df_results.iterrows():
            if pd.isna(row['mean_bac']):
                logger.info(f"{row['lambda']:10.1e} |        NaN |        NaN | {row['n_valid']:4.0f}/{row['n_total']:<3.0f}")
            else:
                logger.info(f"{row['lambda']:10.1e} | {row['mean_bac']:10.4f} | {row['std_bac']:10.4f} | {row['n_valid']:4.0f}/{row['n_total']:<3.0f}")

        # Save results
        output_dir = Path('outputs/step3_lambda')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        scores_path = output_dir / 'lambda_scores_K3_quick.csv'
        df_results.to_csv(scores_path, index=False)
        logger.info(f"\nScores saved to: {scores_path}")

        # Save best lambda JSON
        best_lambda_data = {
            'lambda': float(best_lambda),
            'mean_bac': float(best_bac),
            'std_bac': float(best_std),
            'K': 3,
            'grid_size': 0.05,
            'n_simulations': len(simulations),
            'model_type': 'CJM_mode',
            'features': feature_names,
            'method': 'quick_empirical'
        }

        best_path = output_dir / 'best_lambda_K3_final.json'
        with open(best_path, 'w') as f:
            json.dump(best_lambda_data, f, indent=2)
        logger.info(f"Best lambda saved to: {best_path}")

        # Create plot
        create_plot(df_results, best_lambda, output_dir)

        logger.info("\n✅ Quick empirical lambda scan completed successfully!")
        logger.info(f"Recommended lambda: {best_lambda:.1e}")

        return best_lambda_data

    else:
        logger.error("All lambda values failed!")
        return None


def create_plot(df_results, best_lambda, output_dir):
    """Create BAC vs Lambda plot"""
    df_valid = df_results[~df_results['mean_bac'].isna()].copy()

    if len(df_valid) == 0:
        return

    plt.figure(figsize=(10, 6))

    # Plot with error bars
    plt.errorbar(df_valid['lambda'], df_valid['mean_bac'],
                yerr=df_valid['std_bac'],
                marker='o', markersize=8,
                linewidth=2, capsize=5,
                label='Mean BAC ± σ', color='blue')

    # Mark best
    best_row = df_valid[df_valid['lambda'] == best_lambda].iloc[0]
    plt.plot(best_lambda, best_row['mean_bac'],
            'r*', markersize=20,
            label=f'Best λ={best_lambda:.1e}')

    # Random baseline
    plt.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5,
                label='Random (BAC=0.333)')

    plt.xscale('log')
    plt.xlabel('Lambda (λ)', fontsize=12)
    plt.ylabel('Mean Balanced Accuracy', fontsize=12)
    plt.title('Empirical Lambda Selection: K=3 CJM (Quick Scan)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / 'bac_vs_lambda_K3_final.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()