"""
Step 0: Prepare VNINDEX dataset for modeling
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from src.utils_io import load_config, save_csv, setup_logging

logger = setup_logging('step0_prepare')


def prepare_vnindex_data(config):
    """
    Prepare VNINDEX dataset for modeling

    Steps:
    1. Load CSV
    2. Parse date column to datetime, sort ascending
    3. Filter date range: 2018-01-01 to 2025-12-31
    4. Compute log return from close
    5. Build modeling dataframe with features
    6. Handle missing values

    Parameters:
    -----------
    config : dict
        Configuration dictionary

    Returns:
    --------
    pd.DataFrame
        Prepared dataframe with date, close, ret, and feature columns
    """
    # Load data
    logger.info(f"Loading data from {config['data']['path']}")
    df = pd.read_csv(config['data']['path'])

    # Parse date column
    date_col = config['data']['date_col']
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date ascending
    df = df.sort_values(date_col).reset_index(drop=True)

    # Filter date range
    start_date = pd.to_datetime(config['data']['start'])
    end_date = pd.to_datetime(config['data']['end'])

    logger.info(f"Filtering data from {start_date} to {end_date}")
    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

    if len(df) == 0:
        raise ValueError(f"No data found in date range {start_date} to {end_date}")

    logger.info(f"Data range: {df[date_col].min()} to {df[date_col].max()}")
    logger.info(f"Number of rows after date filter: {len(df)}")

    # Compute log return
    close_col = config['data']['close_col']
    df['ret'] = np.log(df[close_col] / df[close_col].shift(1))

    # Drop first row with NaN return
    df = df.dropna(subset=['ret']).reset_index(drop=True)

    # Build modeling dataframe
    feature_cols = config['features']['cols']
    output_cols = ['date', 'close', 'ret'] + feature_cols

    # Rename columns for consistency
    df['date'] = df[date_col]
    df['close'] = df[close_col]

    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Select columns
    df_model = df[output_cols].copy()

    # Handle missing values in features
    # Option 1: Forward fill then drop remaining NaN
    logger.info("Handling missing values in features...")
    initial_rows = len(df_model)

    # Forward fill feature columns
    for col in feature_cols:
        df_model[col] = df_model[col].fillna(method='ffill')

    # Drop rows with any remaining NaN in features
    df_model = df_model.dropna(subset=feature_cols).reset_index(drop=True)

    dropped_rows = initial_rows - len(df_model)
    logger.info(f"Dropped {dropped_rows} rows with NaN values")
    logger.info(f"Final dataset has {len(df_model)} rows")

    # Verify no NaN values remain
    for col in feature_cols:
        nan_count = df_model[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column {col} still has {nan_count} NaN values")

    return df_model


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('configs/daily.yaml')

    # Prepare data
    df = prepare_vnindex_data(config)

    # Save prepared data
    output_path = 'outputs/prepared_data.csv'
    save_csv(df, output_path)

    logger.info(f"Saved prepared data to {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Display basic statistics
    logger.info("\nFeature statistics:")
    for col in config['features']['cols']:
        logger.info(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")


if __name__ == "__main__":
    main()