#!/usr/bin/env python3
"""
Analyze OHLC gaps (high-close, low-close) by regime
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data_with_states():
    """Load VNINDEX data with HMM state assignments"""
    # Load the raw data with OHLC
    data_path = Path('data/processed/vnindex_daily_features_20180102_20250103.parquet')
    df = pd.read_parquet(data_path)

    # Check if we have OHLC data
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Warning: Missing OHLC columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Load HMM parameters to get state assignments
    params_path = Path('outputs/step1_params/hmm_K3_features.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    # We need to get the fitted states - check if we have a file with state assignments
    # Look for the fitted states file
    states_files = list(Path('outputs/step1_params').glob('*states*.csv'))
    if not states_files:
        print("Warning: No state assignment files found")
        print("We'll need to load from the CJM output or re-fit the model")
        return None

    return df

def analyze_ohlc_gaps_by_regime():
    """Calculate OHLC gap statistics by regime"""

    print("="*80)
    print("OHLC GAP ANALYSIS BY REGIME")
    print("="*80)

    # First, let's check what data we have
    print("\nChecking available data...")

    # Load the data
    data_path = Path('data/processed/vnindex_daily_features_20180102_20250103.parquet')
    df = pd.read_parquet(data_path)

    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if we have OHLC columns
    has_ohlc = all(col in df.columns for col in ['open', 'high', 'low', 'close'])

    if not has_ohlc:
        print("\n❌ OHLC data not found in the features file")
        print("We need to load the raw price data")

        # Try to load raw data
        raw_data_path = Path('data/raw/vnindex_daily_20180102_20250103.csv')
        if raw_data_path.exists():
            print(f"\n✓ Found raw data at: {raw_data_path}")
            df_raw = pd.read_csv(raw_data_path)
            print(f"Raw data shape: {df_raw.shape}")
            print(f"Raw data columns: {list(df_raw.columns)}")

            # Merge with features data to get dates aligned
            # Assuming the feature data has a date/time column
            if 'date' in df_raw.columns or 'time' in df_raw.columns:
                date_col = 'date' if 'date' in df_raw.columns else 'time'

                # Check feature data for date column
                feature_date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                print(f"\nFeature data date columns: {feature_date_cols}")

                if df.index.name and 'date' in df.index.name.lower():
                    df = df.reset_index()
                    feature_date_col = df.columns[0]
                elif feature_date_cols:
                    feature_date_col = feature_date_cols[0]
                else:
                    print("Cannot find date column in feature data")
                    return

                # Merge
                df_raw[date_col] = pd.to_datetime(df_raw[date_col])
                df[feature_date_col] = pd.to_datetime(df[feature_date_col])

                df_merged = df_raw.merge(df, left_on=date_col, right_on=feature_date_col, how='inner')
                df = df_merged
                print(f"\nMerged data shape: {df.shape}")
        else:
            print(f"\n❌ Raw data file not found at: {raw_data_path}")
            return

    # Now check for state assignments
    # The states might be in a separate file or we need to load from CJM results
    if 's_fitted' in df.columns or 'state' in df.columns:
        state_col = 's_fitted' if 's_fitted' in df.columns else 'state'
    else:
        print("\n❌ State assignments not found in data")
        print("Looking for CJM output files...")

        # Try to load from CJM output
        cjm_files = list(Path('outputs').rglob('*cjm*.csv'))
        if cjm_files:
            print(f"Found {len(cjm_files)} CJM output files")
            # Load the most recent one
            cjm_file = sorted(cjm_files, key=lambda x: x.stat().st_mtime)[-1]
            print(f"Loading: {cjm_file}")

            df_cjm = pd.read_csv(cjm_file)
            print(f"CJM data columns: {list(df_cjm.columns)}")

            # Find state column
            state_cols = [col for col in df_cjm.columns if 'state' in col.lower() or 's_' in col.lower()]
            if state_cols:
                print(f"Found state columns: {state_cols}")
                state_col = state_cols[0]

                # Merge with our data
                if 'date' in df_cjm.columns or 'time' in df_cjm.columns:
                    cjm_date_col = 'date' if 'date' in df_cjm.columns else 'time'
                    df_cjm[cjm_date_col] = pd.to_datetime(df_cjm[cjm_date_col])

                    # Get date column from our df
                    if 'date' in df.columns:
                        our_date_col = 'date'
                    elif 'time' in df.columns:
                        our_date_col = 'time'
                    else:
                        our_date_col = df.columns[0]

                    df[our_date_col] = pd.to_datetime(df[our_date_col])
                    df = df.merge(df_cjm[[cjm_date_col, state_col]],
                                 left_on=our_date_col, right_on=cjm_date_col, how='inner')
            else:
                print("No state columns found in CJM file")
                return
        else:
            print("No CJM output files found")
            return

    # Now we should have OHLC and states
    print(f"\n✓ Using state column: {state_col}")

    # Calculate percentage gaps
    df['high_close_gap_pct'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_close_gap_pct'] = (df['close'] - df['low']) / df['close'] * 100

    print("\n" + "="*80)
    print("RESULTS: OHLC GAP STATISTICS BY REGIME")
    print("="*80)

    # Calculate statistics by state
    states = sorted(df[state_col].unique())

    # State labels
    state_labels = {
        0: "Bullish (State 0)",
        1: "Neutral (State 1)",
        2: "Bearish (State 2)"
    }

    results = []

    for state in states:
        state_data = df[df[state_col] == state]

        result = {
            'State': state_labels.get(state, f"State {state}"),
            'Count': len(state_data),
            'High-Close Mean (%)': state_data['high_close_gap_pct'].mean(),
            'High-Close Std (%)': state_data['high_close_gap_pct'].std(),
            'High-Close Min (%)': state_data['high_close_gap_pct'].min(),
            'High-Close Max (%)': state_data['high_close_gap_pct'].max(),
            'Low-Close Mean (%)': state_data['low_close_gap_pct'].mean(),
            'Low-Close Std (%)': state_data['low_close_gap_pct'].std(),
            'Low-Close Min (%)': state_data['low_close_gap_pct'].min(),
            'Low-Close Max (%)': state_data['low_close_gap_pct'].max(),
        }
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('State')

    print("\n" + df_results.to_string(float_format=lambda x: f"{x:.4f}"))

    # Print detailed summary
    print("\n" + "="*80)
    print("DETAILED SUMMARY")
    print("="*80)

    for state in states:
        state_data = df[df[state_col] == state]
        state_label = state_labels.get(state, f"State {state}")

        print(f"\n{state_label}:")
        print(f"  Sample size: {len(state_data)} days")

        print(f"\n  High - Close Gap:")
        print(f"    Mean: {state_data['high_close_gap_pct'].mean():.4f}%")
        print(f"    Std:  {state_data['high_close_gap_pct'].std():.4f}%")
        print(f"    Range: [{state_data['high_close_gap_pct'].min():.4f}%, {state_data['high_close_gap_pct'].max():.4f}%]")

        print(f"\n  Close - Low Gap:")
        print(f"    Mean: {state_data['low_close_gap_pct'].mean():.4f}%")
        print(f"    Std:  {state_data['low_close_gap_pct'].std():.4f}%")
        print(f"    Range: [{state_data['low_close_gap_pct'].min():.4f}%, {state_data['low_close_gap_pct'].max():.4f}%]")

        # Average total daily range
        daily_range = state_data['high_close_gap_pct'] + state_data['low_close_gap_pct']
        print(f"\n  Average Daily Range (High-Low):")
        print(f"    Mean: {daily_range.mean():.4f}%")
        print(f"    Std:  {daily_range.std():.4f}%")

    # Save results
    output_dir = Path('outputs/ohlc_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(output_dir / 'ohlc_gaps_by_regime.csv')
    print(f"\n✓ Results saved to: {output_dir / 'ohlc_gaps_by_regime.csv'}")

    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_ohlc_gaps_by_regime()
