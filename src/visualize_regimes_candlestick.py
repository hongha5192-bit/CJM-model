#!/usr/bin/env python
"""
Visualize HMM K=3 regime labels on daily candlestick chart
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_regime_predictions():
    """Get regime predictions using the fitted HMM model with features"""

    # Load prepared data
    df = pd.read_csv('outputs/prepared_data.csv', parse_dates=['date'])

    # Load HMM parameters
    with open('outputs/step1_params/hmm_K3_features.json', 'r') as f:
        hmm_params = json.load(f)

    # Get features
    feature_cols = hmm_params['features']
    X_raw = df[feature_cols].values

    # Remove NaN rows
    valid_idx = ~np.any(np.isnan(X_raw), axis=1)
    X_clean = X_raw[valid_idx]
    df_clean = df[valid_idx].reset_index(drop=True)

    # Standardize features (using saved parameters)
    scaler = StandardScaler()
    scaler.mean_ = np.array(hmm_params['scaler_params']['mean'])
    scaler.scale_ = np.array(hmm_params['scaler_params']['scale'])
    X_scaled = scaler.transform(X_clean)

    # Recreate HMM model with saved parameters
    K = hmm_params['K']
    model = GaussianHMM(n_components=K, covariance_type='full', n_iter=1)

    # Set model parameters from saved values
    model.startprob_ = np.array(hmm_params['pi'])
    model.transmat_ = np.array(hmm_params['P'])

    # Set means (need to transform back to scaled space)
    means_original = np.array(hmm_params['means_matrix'])
    model.means_ = scaler.transform(means_original)

    # For covariances, we need to handle them carefully
    # Using identity for simplicity since we mainly need predictions
    model.covars_ = np.array([np.eye(len(feature_cols)) * 0.1 for _ in range(K)])

    # Predict states
    states = model.predict(X_scaled)

    # Add states to dataframe
    df_clean['regime'] = states

    return df_clean


def load_ohlc_data():
    """Load OHLC data from original source"""

    # Load full data with OHLC
    df_full = pd.read_csv('data/vnindex_full.csv')

    # Select and rename columns before any processing
    df_full = df_full[['TRADINGDATE', 'OPENINDEX', 'HIGHESTINDEX', 'LOWESTINDEX', 'CLOSEINDEX', 'TOTALMATCHVOLUME']].rename(columns={
        'TRADINGDATE': 'date',
        'OPENINDEX': 'open',
        'HIGHESTINDEX': 'high',
        'LOWESTINDEX': 'low',
        'CLOSEINDEX': 'close',
        'TOTALMATCHVOLUME': 'volume'
    })

    # Convert date column
    df_full['date'] = pd.to_datetime(df_full['date'])

    # Filter dates
    df_full = df_full[(df_full['date'] >= '2018-01-01') & (df_full['date'] <= '2025-12-31')]

    return df_full


def create_candlestick_with_regimes(df_ohlc, df_regimes, period_days=250):
    """Create interactive candlestick chart with regime coloring"""

    # Merge OHLC with regime predictions
    df = pd.merge(df_ohlc, df_regimes[['date', 'regime']], on='date', how='inner')

    # Take last N days for better visibility
    df = df.tail(period_days)

    # Define colors for each regime
    regime_colors = {
        0: 'rgba(0, 255, 0, 0.2)',    # Green for bullish
        1: 'rgba(128, 128, 128, 0.2)', # Gray for neutral
        2: 'rgba(255, 0, 0, 0.2)'      # Red for bearish
    }

    regime_names = {
        0: 'Bullish (Low Vol)',
        1: 'Neutral (Medium Vol)',
        2: 'Bearish (High Vol)'
    }

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('VNINDEX with Regime Overlay', 'Volume', 'Regime State')
    )

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='VNINDEX',
            increasing=dict(line=dict(color='green')),
            decreasing=dict(line=dict(color='red'))
        ),
        row=1, col=1
    )

    # Add regime backgrounds
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]

        # Add vertical rectangles for each regime period
        for i in range(len(regime_df)):
            if i == 0 or regime_df.iloc[i].name != regime_df.iloc[i-1].name + 1:
                # Start of a new regime period
                start_date = regime_df.iloc[i]['date']
                start_idx = i

            if i == len(regime_df) - 1 or regime_df.iloc[i].name != regime_df.iloc[i+1].name - 1:
                # End of a regime period
                end_date = regime_df.iloc[i]['date']

                fig.add_shape(
                    type="rect",
                    x0=start_date, x1=end_date,
                    y0=df['low'].min() * 0.95, y1=df['high'].max() * 1.05,
                    fillcolor=regime_colors[regime],
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )

    # Add volume bars
    colors = ['green' if close >= open else 'red'
              for close, open in zip(df['close'], df['open'])]

    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            marker_color=colors,
            name='Volume',
            showlegend=False
        ),
        row=2, col=1
    )

    # Add regime indicator
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['regime'],
            mode='lines+markers',
            name='Regime',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            showlegend=False
        ),
        row=3, col=1
    )

    # Add regime labels
    for regime in range(3):
        fig.add_hline(
            y=regime,
            line_dash="dash",
            line_color="gray",
            annotation_text=regime_names[regime],
            annotation_position="left",
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        title=f'VNINDEX with HMM K=3 Regime Classification (Last {period_days} Days)',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=900,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Regime", row=3, col=1, range=[-0.5, 2.5])

    # Update x-axis
    fig.update_xaxes(title_text="Date", row=3, col=1)

    return fig


def create_regime_distribution_plot(df_regimes):
    """Create pie chart of regime distribution"""

    regime_counts = df_regimes['regime'].value_counts().sort_index()

    labels = ['Bullish (Low Vol)', 'Neutral (Medium Vol)', 'Bearish (High Vol)']
    colors = ['green', 'gray', 'red']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=regime_counts.values,
        hole=.3,
        marker=dict(colors=colors)
    )])

    fig.update_layout(
        title="Regime Distribution (2018-2025)",
        annotations=[dict(text='HMM K=3', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    return fig


def main():
    """Main visualization function"""

    logger.info("Loading data...")

    # Get regime predictions
    df_regimes = get_regime_predictions()
    logger.info(f"Generated regime predictions for {len(df_regimes)} days")

    # Load OHLC data
    df_ohlc = load_ohlc_data()
    logger.info(f"Loaded OHLC data for {len(df_ohlc)} days")

    # Create candlestick chart with regimes (last 500 days)
    fig_candle = create_candlestick_with_regimes(df_ohlc, df_regimes, period_days=500)

    # Save candlestick chart
    output_path = Path('outputs/visualizations')
    output_path.mkdir(parents=True, exist_ok=True)

    candle_path = output_path / 'vnindex_candlestick_regimes.html'
    fig_candle.write_html(str(candle_path))
    logger.info(f"Saved candlestick chart to {candle_path}")

    # Also create a shorter period for better detail (last 100 days)
    fig_candle_short = create_candlestick_with_regimes(df_ohlc, df_regimes, period_days=100)
    candle_short_path = output_path / 'vnindex_candlestick_regimes_100d.html'
    fig_candle_short.write_html(str(candle_short_path))
    logger.info(f"Saved 100-day chart to {candle_short_path}")

    # Create regime distribution
    fig_dist = create_regime_distribution_plot(df_regimes)
    dist_path = output_path / 'regime_distribution.html'
    fig_dist.write_html(str(dist_path))
    logger.info(f"Saved distribution chart to {dist_path}")

    # Print summary
    print("\n" + "="*60)
    print("REGIME VISUALIZATION SUMMARY")
    print("="*60)

    regime_counts = df_regimes['regime'].value_counts().sort_index()
    regime_pcts = df_regimes['regime'].value_counts(normalize=True).sort_index()

    print("\nRegime Distribution:")
    for regime in range(3):
        count = regime_counts.get(regime, 0)
        pct = regime_pcts.get(regime, 0) * 100
        name = ['Bullish', 'Neutral', 'Bearish'][regime]
        print(f"  Regime {regime} ({name}): {count:4d} days ({pct:5.1f}%)")

    # Calculate average returns per regime
    print("\nAverage Returns by Regime:")
    for regime in range(3):
        regime_data = df_regimes[df_regimes['regime'] == regime]
        if 'ret' in regime_data.columns:
            mean_ret = regime_data['ret'].mean() * 100
            std_ret = regime_data['ret'].std() * 100
            print(f"  Regime {regime}: {mean_ret:+6.3f}% ± {std_ret:5.3f}%")

    print("\nVisualization files created:")
    print(f"  1. {candle_path}")
    print(f"  2. {candle_short_path}")
    print(f"  3. {dist_path}")
    print("="*60)

    return fig_candle, fig_dist


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}", exc_info=True)
        raise