"""
Step 3: Lambda scan - ONE simulation study for jump-penalty selection
"""
import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel  # Correct import from actual package
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')
from src.utils_io import load_config, load_parquet, save_csv, save_json, setup_logging, numpy_to_python
from src.metrics import best_balanced_accuracy_K2

logger = setup_logging('step3_lambda_scan')


def fit_jump_model(y, lam, model_type, config, seed):
    """
    Fit a jump model (JM or CJM) to data

    Parameters:
    -----------
    y : np.ndarray
        Observations (T,)
    lam : float
        Jump penalty parameter
    model_type : str
        'JM' for discrete, 'CJM_mode' for continuous with mode loss
    config : dict
        Configuration dictionary
    seed : int
        Random seed

    Returns:
    --------
    np.ndarray
        Predicted labels
    """
    X = y.reshape(-1, 1)

    if model_type == 'JM':
        # Discrete Jump Model
        model = JumpModel(
            n_components=2,
            jump_penalty=lam,
            cont=False,
            n_init=config['lambda_scan']['jumpmodels']['n_init'],
            max_iter=config['lambda_scan']['jumpmodels']['max_iter'],
            tol=config['lambda_scan']['jumpmodels']['tol'],
            random_state=seed,
        )
        model.fit(X, ret_ser=y, sort_by="cumret")
        y_pred = model.predict(X)

    elif model_type == 'CJM_mode':
        # Continuous Jump Model with mode loss
        model = JumpModel(
            n_components=2,
            jump_penalty=lam,
            cont=True,
            grid_size=config['lambda_scan']['cjm']['grid_size'],
            mode_loss=config['lambda_scan']['cjm']['mode_loss'],
            n_init=config['lambda_scan']['jumpmodels']['n_init'],
            max_iter=config['lambda_scan']['jumpmodels']['max_iter'],
            tol=config['lambda_scan']['jumpmodels']['tol'],
            random_state=seed,
        )
        model.fit(X, ret_ser=y, sort_by="cumret")
        proba = model.predict_proba(X)
        y_pred = proba.values.argmax(axis=1)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return y_pred


def process_single_lambda(lam, config, sim_data_list):
    """
    Process a single lambda value across all simulations

    Parameters:
    -----------
    lam : float
        Lambda value
    config : dict
        Configuration
    sim_data_list : list
        List of (y, s_true) tuples for all simulations

    Returns:
    --------
    dict
        Results for this lambda
    """
    N_sim = len(sim_data_list)
    model_types = ['JM', 'CJM_mode']

    results = {model_type: [] for model_type in model_types}

    for sim_id, (y, s_true) in enumerate(sim_data_list):
        seed = config['simulation']['seed0'] + sim_id

        for model_type in model_types:
            try:
                # Fit model and get predictions
                y_pred = fit_jump_model(y, lam, model_type, config, seed)

                # Calculate best BAC (with permutation fix)
                bac = best_balanced_accuracy_K2(s_true, y_pred)
                results[model_type].append(bac)

            except Exception as e:
                logger.warning(f"Error fitting {model_type} for lambda={lam}, sim_id={sim_id}: {e}")
                # Use NaN for failed fits
                results[model_type].append(np.nan)

    return results


def lambda_scan(config):
    """
    Main lambda scan function

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    """
    # Create lambda grid
    lambda_grid = np.logspace(
        config['lambda_scan']['lambda_grid']['log10_min'],
        config['lambda_scan']['lambda_grid']['log10_max'],
        config['lambda_scan']['lambda_grid']['n_grid']
    )

    logger.info(f"Lambda grid: {len(lambda_grid)} values from {lambda_grid.min():.2e} to {lambda_grid.max():.2e}")

    # Load all simulated data
    N_sim = config['simulation']['N_sim']
    T = config['simulation']['T']
    K = config['hmm']['K']

    logger.info(f"Loading {N_sim} simulated sequences...")
    sim_data_list = []

    for sim_id in range(N_sim):
        filepath = f"outputs/step2_sim/K{K}/sim_T{T:04d}_seed{sim_id:03d}.parquet"

        if not Path(filepath).exists():
            logger.warning(f"Missing simulation file: {filepath}")
            continue

        df_sim = load_parquet(filepath)
        y = df_sim['y'].values
        s_true = df_sim['s_true'].values
        sim_data_list.append((y, s_true))

    logger.info(f"Loaded {len(sim_data_list)} simulations")

    # Process each lambda value
    all_results = []

    for lam in tqdm(lambda_grid, desc="Lambda scan"):
        logger.info(f"Processing lambda = {lam:.2e}")
        lambda_results = process_single_lambda(lam, config, sim_data_list)

        # Aggregate results
        for model_type in ['JM', 'CJM_mode']:
            bac_values = np.array(lambda_results[model_type])
            # Remove NaN values for statistics
            valid_bac = bac_values[~np.isnan(bac_values)]

            if len(valid_bac) > 0:
                result = {
                    'K': K,
                    'model_type': model_type,
                    'lambda': lam,
                    'mean_BAC': np.mean(valid_bac),
                    'std_BAC': np.std(valid_bac),
                    'n_sim': len(valid_bac)
                }
            else:
                result = {
                    'K': K,
                    'model_type': model_type,
                    'lambda': lam,
                    'mean_BAC': np.nan,
                    'std_BAC': np.nan,
                    'n_sim': 0
                }

            all_results.append(result)

    return pd.DataFrame(all_results)


def select_best_lambda(df_results):
    """
    Select best lambda for each model type

    Parameters:
    -----------
    df_results : pd.DataFrame
        Lambda scan results

    Returns:
    --------
    dict
        Best lambda for each model type
    """
    best_lambda = {}

    for model_type in ['JM', 'CJM_mode']:
        df_model = df_results[df_results['model_type'] == model_type]
        # Find lambda with maximum mean_BAC
        idx_best = df_model['mean_BAC'].idxmax()
        best_lam = df_model.loc[idx_best, 'lambda']
        best_bac = df_model.loc[idx_best, 'mean_BAC']

        best_lambda[model_type] = float(best_lam)
        logger.info(f"Best lambda for {model_type}: {best_lam:.2e} (BAC={best_bac:.4f})")

    return best_lambda


def plot_bac_vs_lambda(df_results, output_path):
    """
    Plot BAC vs Lambda for each model type

    Parameters:
    -----------
    df_results : pd.DataFrame
        Lambda scan results
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    for model_type in ['JM', 'CJM_mode']:
        df_model = df_results[df_results['model_type'] == model_type]
        plt.plot(df_model['lambda'], df_model['mean_BAC'],
                 marker='o', label=model_type, linewidth=2)

    plt.xscale('log')
    plt.xlabel('Lambda (jump penalty)', fontsize=12)
    plt.ylabel('Mean Balanced Accuracy', fontsize=12)
    plt.title('BAC vs Lambda - VNINDEX Simulation Study', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Plot saved to {output_path}")


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('configs/daily.yaml')

    # Run lambda scan
    logger.info("Starting lambda scan...")
    df_results = lambda_scan(config)

    # Save results
    output_csv = 'outputs/step3_lambda/lambda_scores.csv'
    save_csv(df_results, output_csv)
    logger.info(f"Lambda scores saved to {output_csv}")

    # Select best lambda
    best_lambda = select_best_lambda(df_results)

    # Format for JSON output (matching required structure)
    best_lambda_json = {
        f"K{config['hmm']['K']}": best_lambda
    }

    output_json = 'outputs/step3_lambda/best_lambda.json'
    save_json(numpy_to_python(best_lambda_json), output_json)
    logger.info(f"Best lambda saved to {output_json}")

    # Create plot
    output_plot = 'outputs/step3_lambda/bac_vs_lambda.png'
    plot_bac_vs_lambda(df_results, output_plot)

    logger.info("Lambda scan completed!")


if __name__ == "__main__":
    main()