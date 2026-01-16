#!/usr/bin/env python
"""
Step 3: Lambda scan for CJM K=3 using PRICE-BASED simulations
Features: ADX, DMI_Plus, DMI_Minus, URSI, BB_PCTB, BBWP
Optimized for Mac Air M4 with memory-efficient processing
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import JumpModel from jumpmodels package
from jumpmodels.jump import JumpModel


def best_balanced_accuracy_K3(s_pred, s_true):
    """
    Compute best balanced accuracy over all 6 permutations for K=3
    """
    from itertools import permutations
    from sklearn.metrics import balanced_accuracy_score

    K = 3
    best_bac = 0

    for perm in permutations(range(K)):
        # Map predictions using this permutation
        mapping = {old: new for old, new in enumerate(perm)}
        s_pred_mapped = pd.Series(s_pred).map(mapping).values

        # Compute BAC
        bac = balanced_accuracy_score(s_true, s_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac


def fit_cjm_single_sim(sim_data, lambda_val, K, grid_size, config):
    """
    Fit CJM on a single simulation with given lambda

    Args:
        sim_data: DataFrame with simulation data
        lambda_val: Lambda value to use
        K: Number of states
        grid_size: Grid size for discretization
        config: Configuration dictionary
    """
    try:
        from sklearn.preprocessing import StandardScaler

        # Extract features - HARDCODED for price-based simulations
        feature_names = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP']
        X_raw = sim_data[feature_names].values  # numpy array (T, 6)

        # Standardize features (CRITICAL for CJM!)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Get returns and true states
        ret_ser = sim_data['returns'].values  # Use 'returns' column from simulation

        # DEBUG: Log data types and shapes
        logger.debug(f"X shape: {X.shape}, dtype: {X.dtype}")
        logger.debug(f"ret_ser shape: {ret_ser.shape}, dtype: {ret_ser.dtype}")
        logger.debug(f"X has NaN: {np.isnan(X).any()}, ret_ser has NaN: {np.isnan(ret_ser).any()}")

        # Fit CJM using JumpModel
        model = JumpModel(
            n_components=K,
            jump_penalty=lambda_val,
            cont=True,
            mode_loss=True,
            grid_size=grid_size,
            n_init=config['lambda_scan']['jumpmodels'].get('n_init', 3),
            max_iter=500,  # Increased from 200 to 500 for better convergence
            tol=float(config['lambda_scan']['jumpmodels'].get('tol', 1e-5)),  # Convert to float
            random_state=123,
            verbose=0
        )

        logger.debug(f"About to fit model with lambda={lambda_val}")
        # Fit with returns for sorting (KEY: same as real data fitting!)
        model.fit(X, ret_ser=ret_ser, sort_by='cumret')
        logger.debug(f"Model fit completed successfully")

        # Get predictions
        s_pred = model.predict(X)
        s_true = sim_data['s_true'].values

        # Calculate best BAC
        bac = best_balanced_accuracy_K3(s_pred, s_true)

        return bac

    except Exception as e:
        logger.error(f"Error fitting simulation: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return np.nan


def process_lambda_batch(lambda_val, simulations, K, grid_size, config):
    """
    Process all simulations for a single lambda value

    Args:
        lambda_val: Lambda value to test
        simulations: List of simulation DataFrames
        K: Number of states
        grid_size: Grid size
        config: Configuration
    """
    bac_scores = []

    for sim_idx, sim_data in enumerate(simulations):
        bac = fit_cjm_single_sim(sim_data, lambda_val, K, grid_size, config)

        if not np.isnan(bac):
            bac_scores.append(bac)
            if sim_idx % 10 == 0:
                logger.debug(f"  λ={lambda_val:.2e}, Sim {sim_idx}: BAC={bac:.3f}")

    if len(bac_scores) > 0:
        mean_bac = np.mean(bac_scores)
        std_bac = np.std(bac_scores)
        n_valid = len(bac_scores)
    else:
        mean_bac = np.nan
        std_bac = np.nan
        n_valid = 0

    return {
        'lambda': lambda_val,
        'mean_bac': mean_bac,
        'std_bac': std_bac,
        'n_valid': n_valid,
        'n_total': len(simulations)
    }


def lambda_scan_features(config_path='configs/daily_K3.yaml'):
    """Perform lambda scan using feature-based simulations"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # HARDCODED for K=3 price-based simulations
    K = 3

    # Lambda grid - HARDCODED specific values for testing
    lambda_grid = np.array([5, 10, 15, 20, 50, 100])

    # CJM parameters
    grid_size = config['lambda_scan']['cjm']['grid_size']
    n_jobs = min(2, config['lambda_scan']['jumpmodels'].get('n_jobs', 2))

    logger.info("="*80)
    logger.info("LAMBDA SCAN FOR PRICE-BASED HMM (K=3)")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  Features = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP']")
    logger.info(f"  grid_size = {grid_size}")
    logger.info(f"  Lambda values to test: {lambda_grid}")
    logger.info(f"  n_jobs = {n_jobs}")

    # Load simulations into memory - HARDCODED path for price-based simulations
    sim_dir = Path('outputs/step2_sim/K3_price_based')
    sim_files = sorted(sim_dir.glob('sim_*.parquet'))[:10]  # Use 10 simulations

    if not sim_files:
        raise ValueError(f"No simulation files found in {sim_dir}")

    logger.info(f"\nLoading {len(sim_files)} simulations into memory...")
    simulations = []
    for sim_file in tqdm(sim_files, desc="Loading simulations"):
        df = pd.read_parquet(sim_file)
        simulations.append(df)

    logger.info(f"Loaded {len(simulations)} simulations")
    logger.info(f"Memory usage: {sum(df.memory_usage(deep=True).sum() for df in simulations) / 1e6:.1f} MB")

    # Scan lambda values
    logger.info(f"\nStarting lambda scan with {len(lambda_grid)} values...")
    logger.info("Lambda values: " + ", ".join([f"{lam:.0f}" for lam in lambda_grid]))

    results = []

    # Sequential processing for better progress tracking on Mac Air M4
    for idx, lambda_val in enumerate(lambda_grid):
        logger.info(f"\nProcessing λ={lambda_val:.0f} ({idx+1}/{len(lambda_grid)})...")

        result = process_lambda_batch(
            lambda_val, simulations, K, grid_size, config
        )

        results.append(result)

        if not np.isnan(result['mean_bac']):
            logger.info(f"  Result: Mean BAC = {result['mean_bac']:.4f} ± {result['std_bac']:.4f} "
                       f"({result['n_valid']}/{result['n_total']} valid)")
        else:
            logger.info(f"  Result: All simulations failed")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Find best lambda
    if not df_results['mean_bac'].isna().all():
        best_idx = df_results['mean_bac'].idxmax()
        best_lambda = df_results.loc[best_idx, 'lambda']
        best_bac = df_results.loc[best_idx, 'mean_bac']
        best_std = df_results.loc[best_idx, 'std_bac']

        logger.info("\n" + "="*80)
        logger.info("LAMBDA SELECTION RESULTS")
        logger.info("="*80)
        logger.info(f"Best lambda: {best_lambda:.2e}")
        logger.info(f"Best mean BAC: {best_bac:.4f} ± {best_std:.4f}")

        # Save results
        output_dir = Path('outputs/step3_lambda')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save scores
        scores_path = output_dir / 'lambda_scores_K3_price_based.csv'
        df_results.to_csv(scores_path, index=False)
        logger.info(f"\nScores saved to: {scores_path}")

        # Save best lambda
        best_lambda_data = {
            'lambda': float(best_lambda),
            'mean_bac': float(best_bac),
            'std_bac': float(best_std),
            'K': K,
            'grid_size': grid_size,
            'n_simulations': len(simulations),
            'model_type': 'CJM_mode',
            'features': ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP'],
            'simulation_type': 'price_based'
        }

        best_path = output_dir / 'best_lambda_K3_price_based.json'
        with open(best_path, 'w') as f:
            json.dump(best_lambda_data, f, indent=2)
        logger.info(f"Best lambda saved to: {best_path}")

        # Create visualization
        create_lambda_scan_plot(df_results, best_lambda, output_dir)

    else:
        logger.error("All lambda values failed!")
        return None

    return best_lambda_data


def create_lambda_scan_plot(df_results, best_lambda, output_dir):
    """Create visualization of lambda scan results"""
    import plotly.graph_objects as go

    # Filter valid results
    df_valid = df_results[~df_results['mean_bac'].isna()].copy()

    if len(df_valid) == 0:
        return

    fig = go.Figure()

    # Add mean BAC line
    fig.add_trace(go.Scatter(
        x=df_valid['lambda'],
        y=df_valid['mean_bac'],
        mode='lines+markers',
        name='Mean BAC',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Add confidence band
    fig.add_trace(go.Scatter(
        x=df_valid['lambda'],
        y=df_valid['mean_bac'] + df_valid['std_bac'],
        mode='lines',
        name='Upper bound (μ+σ)',
        line=dict(color='lightblue', width=1, dash='dash'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_valid['lambda'],
        y=df_valid['mean_bac'] - df_valid['std_bac'],
        mode='lines',
        name='Lower bound (μ-σ)',
        line=dict(color='lightblue', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.3)',
        showlegend=False
    ))

    # Mark best lambda
    best_row = df_valid[df_valid['lambda'] == best_lambda].iloc[0]
    fig.add_trace(go.Scatter(
        x=[best_lambda],
        y=[best_row['mean_bac']],
        mode='markers',
        name=f'Best λ={best_lambda:.2e}',
        marker=dict(color='red', size=15, symbol='star')
    ))

    # Add horizontal line at random performance (0.333 for K=3)
    fig.add_hline(y=1/3, line_dash="dash", line_color="gray",
                  annotation_text="Random (BAC=0.333)")

    fig.update_layout(
        title='Lambda Scan Results: Price-Based HMM (K=3)',
        xaxis_title='Lambda (λ)',
        yaxis_title='Mean Balanced Accuracy',
        xaxis_type='log',
        height=500,
        width=900,
        showlegend=True,
        yaxis=dict(range=[0.3, max(0.8, df_valid['mean_bac'].max() + 0.05)])
    )

    plot_path = output_dir / 'lambda_scan_K3_price_based.html'
    fig.write_html(plot_path)
    logger.info(f"Lambda scan plot saved to: {plot_path}")


def main():
    """Main function"""
    try:
        result = lambda_scan_features()
        if result:
            logger.info("\n✅ Lambda scan completed successfully!")
            logger.info(f"Optimal λ = {result['lambda']:.2e} with BAC = {result['mean_bac']:.4f}")
        else:
            logger.error("Lambda scan failed")
            return 1
    except Exception as e:
        logger.error(f"Error in lambda scan: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())