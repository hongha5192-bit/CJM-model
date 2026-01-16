"""
Step 4: Fit CJM to real VNINDEX data with selected features
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from jumpmodels.jump import JumpModel  # Correct import from actual package
import sys
sys.path.append('.')
from src.utils_io import load_config, load_json, save_csv, setup_logging

logger = setup_logging('step4_fit_cjm')


def prepare_features(df, feature_cols):
    """
    Prepare feature matrix X for modeling

    Parameters:
    -----------
    df : pd.DataFrame
        Data with feature columns
    feature_cols : list
        List of feature column names

    Returns:
    --------
    X_std : np.ndarray
        Standardized feature matrix
    scaler : StandardScaler
        Fitted scaler object
    valid_idx : np.ndarray
        Indices of valid rows (no NaN)
    """
    # Verify feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Extract features
    X = df[feature_cols].values

    # Check for non-numeric values
    for i, col in enumerate(feature_cols):
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"Feature column '{col}' is not numeric")

    # Handle missing values - drop rows with any NaN
    valid_idx = ~np.any(np.isnan(X), axis=1)
    X_clean = X[valid_idx]

    logger.info(f"Dropped {(~valid_idx).sum()} rows with NaN values")
    logger.info(f"Final feature matrix shape: {X_clean.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_clean)

    return X_std, scaler, valid_idx


def fit_cjm_model(X_std, ret, best_lambda, config):
    """
    Fit CJM model to real data

    Parameters:
    -----------
    X_std : np.ndarray
        Standardized feature matrix (N, D)
    ret : np.ndarray
        Return series
    best_lambda : float
        Selected lambda value
    config : dict
        Configuration

    Returns:
    --------
    model : JumpModel
        Fitted model
    proba : pd.DataFrame
        Predicted probabilities
    labels : np.ndarray
        Predicted labels
    """
    logger.info(f"Fitting CJM with lambda={best_lambda:.2e}")

    # Initialize CJM with mode loss
    model = JumpModel(
        n_components=2,
        jump_penalty=best_lambda,
        cont=True,
        grid_size=config['lambda_scan']['cjm']['grid_size'],
        mode_loss=config['lambda_scan']['cjm']['mode_loss'],
        n_init=config['lambda_scan']['jumpmodels']['n_init'],
        max_iter=config['lambda_scan']['jumpmodels']['max_iter'],
        tol=config['lambda_scan']['jumpmodels']['tol'],
        random_state=config['hmm']['seed'],
    )

    # Fit model
    model.fit(X_std, ret_ser=ret, sort_by="cumret")

    # Get predictions
    proba = model.predict_proba(X_std)
    labels = proba.values.argmax(axis=1)

    logger.info(f"Model fitted successfully")
    logger.info(f"Label distribution: {np.bincount(labels)}")

    return model, proba, labels


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('configs/daily.yaml')

    # Load prepared data
    df = pd.read_csv('outputs/prepared_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Loaded {len(df)} rows of data")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Load best lambda
    best_lambda_dict = load_json('outputs/step3_lambda/best_lambda.json')
    K = config['hmm']['K']
    model_type = config['apply']['model_type']

    # Extract best lambda for CJM_mode
    best_lambda = best_lambda_dict[f"K{K}"][model_type]
    logger.info(f"Using best lambda for {model_type}: {best_lambda:.2e}")

    # Prepare features
    feature_cols = config['features']['cols']
    logger.info(f"Using features: {feature_cols}")

    X_std, scaler, valid_idx = prepare_features(df, feature_cols)

    # Filter data for valid rows
    df_valid = df[valid_idx].reset_index(drop=True)
    ret = df_valid['ret'].values

    # Fit CJM model
    model, proba, labels = fit_cjm_model(X_std, ret, best_lambda, config)

    # Create output dataframe
    df_output = pd.DataFrame({
        'date': df_valid['date'],
        'close': df_valid['close'],
        'ret': df_valid['ret'],
        'label': labels,
        'proba_0': proba.iloc[:, 0],
        'proba_1': proba.iloc[:, 1]
    })

    # Add optional diagnostics
    if hasattr(model, 'centers_'):
        # Add cluster centers (mean feature values for each cluster)
        centers = model.centers_
        for i in range(centers.shape[0]):
            # Store only the first feature's center value as a scalar
            df_output[f'center_{i}_mean'] = centers[i].mean()

    if hasattr(model, 'transmat_'):
        # Add transition matrix entries
        P = model.transmat_
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                df_output[f'P_{i}{j}'] = P[i, j]

    # Save results
    output_path = 'outputs/step4_apply/regimes_daily.csv'
    save_csv(df_output, output_path)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Shape: {df_output.shape}")

    # Print summary statistics
    logger.info("\nRegime Summary:")
    for label_val in [0, 1]:
        mask = df_output['label'] == label_val
        n_days = mask.sum()
        pct = 100 * n_days / len(df_output)
        mean_ret = df_output.loc[mask, 'ret'].mean()
        std_ret = df_output.loc[mask, 'ret'].std()
        logger.info(f"  Label {label_val}: {n_days} days ({pct:.1f}%), mean_ret={mean_ret:.4f}, std_ret={std_ret:.4f}")


if __name__ == "__main__":
    main()