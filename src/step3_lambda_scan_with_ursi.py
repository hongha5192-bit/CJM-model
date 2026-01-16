"""
Lambda scan for K=3 CJM with all features including URSI breadth indicators
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
import plotly.graph_objects as go
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

        # Extract all 8 features including URSI breadth indicators
        feature_names = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP', 'URSI_0_50', 'URSI_0_20']
        X_raw = sim_data[feature_names].values  # numpy array (T, 8)

        # Standardize features (CRITICAL for CJM!)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Get returns and true states
        ret_ser = sim_data['returns'].values

        # Fit CJM using JumpModel
        model = JumpModel(
            n_components=K,
            jump_penalty=lambda_val,
            cont=True,
            mode_loss=True,
            grid_size=grid_size,
            n_init=config['lambda_scan']['jumpmodels'].get('n_init', 3),
            max_iter=500,  # Using 500 for better convergence
            tol=float(config['lambda_scan']['jumpmodels'].get('tol', 1e-5)),
            random_state=123,
            verbose=0
        )

        # Fit with returns for sorting
        model.fit(X, ret_ser=ret_ser, sort_by='cumret')

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
    Process a batch of simulations for a single lambda value

    Args:
        lambda_val: Lambda value to use
        simulations: List of simulation DataFrames
        K: Number of states
        grid_size: Grid size
        config: Configuration dictionary

    Returns:
        List of BAC scores
    """
    bac_scores = []

    for sim_idx, sim_data in enumerate(simulations):
        bac = fit_cjm_single_sim(sim_data, lambda_val, K, grid_size, config)
        bac_scores.append(bac)

        if sim_idx == 0 or (sim_idx + 1) % 10 == 0:
            valid_bacs = [b for b in bac_scores if not np.isnan(b)]
            if valid_bacs:
                logger.debug(f"  λ={lambda_val:.2e}, Sim {sim_idx}: BAC={bac:.3f}")

    return bac_scores

def main():
    """Main lambda scan function for simulations with URSI breadth indicators"""

    logger.info("="*80)
    logger.info("LAMBDA SCAN FOR PRICE-BASED HMM WITH URSI BREADTH (K=3)")
    logger.info("="*80)

    # Load configuration
    config_path = Path('configs/daily_K3.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Hardcoded parameters for K=3 with URSI breadth
    K = 3
    grid_size = config['lambda_scan']['cjm']['grid_size']
    n_jobs = config['lambda_scan']['jumpmodels'].get('n_jobs', 2)

    # Feature names - now including URSI breadth indicators
    feature_names = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP', 'URSI_0_50', 'URSI_0_20']

    # Lambda values to test
    lambda_grid = np.array([5, 10, 15, 20, 50, 100])

    logger.info("Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  grid_size = {grid_size}")
    logger.info(f"  Lambda values to test: {lambda_grid}")
    logger.info(f"  n_jobs = {n_jobs}")

    # Load simulations from the new directory with URSI breadth
    sim_dir = Path('outputs/step2_sim/K3_with_ursi_breadth')
    sim_files = sorted(sim_dir.glob('sim_*.parquet'))[:50]  # Use 50 simulations

    if not sim_files:
        raise ValueError(f"No simulation files found in {sim_dir}")

    logger.info(f"\nLoading {len(sim_files)} simulations into memory...")

    simulations = []
    for f in tqdm(sim_files, desc="Loading simulations"):
        df = pd.read_parquet(f)
        simulations.append(df)

    # Calculate memory usage
    memory_mb = sum(df.memory_usage(deep=True).sum() for df in simulations) / 1e6
    logger.info(f"Loaded {len(simulations)} simulations")
    logger.info(f"Memory usage: {memory_mb:.1f} MB")

    # Scan lambda values
    logger.info(f"\nStarting lambda scan with {len(lambda_grid)} values...")
    logger.info(f"Lambda values: {', '.join([str(l) for l in lambda_grid])}")

    results = []
    for i, lambda_val in enumerate(lambda_grid):
        logger.info(f"\nProcessing λ={lambda_val} ({i+1}/{len(lambda_grid)})...")

        # Process this lambda value
        bac_scores = process_lambda_batch(lambda_val, simulations, K, grid_size, config)

        # Calculate statistics
        valid_bacs = [b for b in bac_scores if not np.isnan(b)]
        n_valid = len(valid_bacs)

        if n_valid > 0:
            mean_bac = np.mean(valid_bacs)
            std_bac = np.std(valid_bacs)
            logger.info(f"  Result: Mean BAC = {mean_bac:.4f} ± {std_bac:.4f} ({n_valid}/{len(bac_scores)} valid)")
        else:
            mean_bac = np.nan
            std_bac = np.nan
            logger.info(f"  Result: All simulations failed")

        results.append({
            'lambda': lambda_val,
            'mean_bac': mean_bac,
            'std_bac': std_bac,
            'n_valid': n_valid,
            'n_total': len(bac_scores)
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Find best lambda
    if results_df['mean_bac'].notna().any():
        best_idx = results_df['mean_bac'].idxmax()
        best_lambda = results_df.loc[best_idx, 'lambda']
        best_bac = results_df.loc[best_idx, 'mean_bac']
        best_std = results_df.loc[best_idx, 'std_bac']

        logger.info("\n" + "="*80)
        logger.info("LAMBDA SELECTION RESULTS")
        logger.info("="*80)
        logger.info(f"Best lambda: {best_lambda:.2e}")
        logger.info(f"Best mean BAC: {best_bac:.4f} ± {best_std:.4f}")

        # Save results
        output_dir = Path('outputs/step3_lambda')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save scores
        scores_path = output_dir / 'lambda_scores_K3_with_ursi.csv'
        results_df.to_csv(scores_path, index=False)
        logger.info(f"\nScores saved to: {scores_path}")

        # Save best lambda
        best_lambda_info = {
            'lambda': float(best_lambda),
            'mean_bac': float(best_bac),
            'std_bac': float(best_std),
            'K': K,
            'grid_size': grid_size,
            'n_simulations': len(simulations),
            'model_type': 'CJM_mode',
            'features': feature_names,
            'simulation_type': 'price_based_with_ursi_breadth'
        }

        best_path = output_dir / 'best_lambda_K3_with_ursi.json'
        with open(best_path, 'w') as f:
            json.dump(best_lambda_info, f, indent=2)
        logger.info(f"Best lambda saved to: {best_path}")

        # Create visualization
        fig = go.Figure()

        # Add mean BAC line
        fig.add_trace(go.Scatter(
            x=results_df['lambda'],
            y=results_df['mean_bac'],
            mode='lines+markers',
            name='Mean BAC',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))

        # Add error bars
        fig.add_trace(go.Scatter(
            x=results_df['lambda'],
            y=results_df['mean_bac'] + results_df['std_bac'],
            mode='lines',
            name='Mean + Std',
            line=dict(color='lightblue', width=0.5, dash='dash'),
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=results_df['lambda'],
            y=results_df['mean_bac'] - results_df['std_bac'],
            mode='lines',
            name='Mean - Std',
            line=dict(color='lightblue', width=0.5, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0,100,200,0.2)',
            showlegend=True
        ))

        # Highlight best lambda
        fig.add_trace(go.Scatter(
            x=[best_lambda],
            y=[best_bac],
            mode='markers',
            name=f'Best λ={best_lambda:.0f}',
            marker=dict(size=15, color='red', symbol='star')
        ))

        # Add horizontal line at 0.333 (random baseline for K=3)
        fig.add_hline(y=0.333, line_dash="dash", line_color="gray",
                     annotation_text="Random (BAC=0.333)")

        fig.update_layout(
            title=f'Lambda Scan Results (K={K}, WITH URSI Breadth)',
            xaxis_title='Lambda (Jump Penalty)',
            yaxis_title='Balanced Accuracy',
            xaxis_type="log",
            height=600,
            width=800,
            showlegend=True,
            hovermode='x unified'
        )

        plot_path = output_dir / 'lambda_scan_K3_with_ursi.html'
        fig.write_html(str(plot_path))
        logger.info(f"Lambda scan plot saved to: {plot_path}")

        logger.info(f"\n✅ Lambda scan completed successfully!")
        logger.info(f"Optimal λ = {best_lambda:.2e} with BAC = {best_bac:.4f}")

    else:
        logger.error("All lambda values failed!")
        logger.error("Lambda scan failed")

if __name__ == "__main__":
    main()