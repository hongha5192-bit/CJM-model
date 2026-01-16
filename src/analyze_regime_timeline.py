#!/usr/bin/env python
"""
Analyze and visualize:
1. Full regime timeline from 2018-2025
2. Regime distribution by year
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import plotly.express as px

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

    # For covariances, use identity for simplicity
    model.covars_ = np.array([np.eye(len(feature_cols)) * 0.1 for _ in range(K)])

    # Predict states
    states = model.predict(X_scaled)

    # Add states to dataframe
    df_clean['regime'] = states
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['quarter'] = df_clean['date'].dt.quarter

    return df_clean


def create_full_timeline_visualization(df):
    """Create full regime timeline from 2018-2025"""

    # Define colors and names
    regime_colors = {
        0: '#00ff00',  # Green for bullish
        1: '#808080',  # Gray for neutral
        2: '#ff0000'   # Red for bearish
    }

    regime_names = {
        0: 'Bullish (Low Vol)',
        1: 'Neutral (Med Vol)',
        2: 'Bearish (High Vol)'
    }

    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            'VNINDEX Close Price with Regime Overlay',
            'Regime State Timeline',
            'Returns Distribution by Regime'
        )
    )

    # Plot 1: Close price with regime background
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='VNINDEX',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )

    # Add regime backgrounds
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]

        # Find continuous regime periods
        regime_periods = []
        start_idx = 0

        for i in range(1, len(regime_df)):
            if regime_df.index[i] != regime_df.index[i-1] + 1:
                # End of continuous period
                regime_periods.append({
                    'start': regime_df.iloc[start_idx]['date'],
                    'end': regime_df.iloc[i-1]['date']
                })
                start_idx = i

        # Add last period
        regime_periods.append({
            'start': regime_df.iloc[start_idx]['date'],
            'end': regime_df.iloc[-1]['date']
        })

        # Add rectangles for each period
        for period in regime_periods:
            fig.add_shape(
                type="rect",
                x0=period['start'], x1=period['end'],
                y0=df['close'].min() * 0.95, y1=df['close'].max() * 1.05,
                fillcolor=regime_colors[regime],
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
            )

    # Plot 2: Regime timeline
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['regime'],
            mode='lines',
            name='Regime',
            line=dict(color='black', width=1),
            fill='tozeroy',
            fillcolor='rgba(0,0,0,0.1)',
            showlegend=False
        ),
        row=2, col=1
    )

    # Add regime labels
    for regime in range(3):
        fig.add_hline(
            y=regime,
            line_dash="dash",
            line_color="gray",
            annotation_text=regime_names[regime],
            annotation_position="right",
            row=2, col=1
        )

    # Plot 3: Returns by regime
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]
        fig.add_trace(
            go.Scatter(
                x=regime_df['date'],
                y=regime_df['ret'] * 100,  # Convert to percentage
                mode='markers',
                name=regime_names[regime],
                marker=dict(
                    color=regime_colors[regime],
                    size=3,
                    opacity=0.5
                )
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        title='VNINDEX Full Regime Timeline (2018-2025)',
        height=900,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_yaxes(title_text="Index Value", row=1, col=1)
    fig.update_yaxes(title_text="Regime", row=2, col=1, range=[-0.5, 2.5])
    fig.update_yaxes(title_text="Return (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    return fig


def create_yearly_distribution(df):
    """Create regime distribution by year"""

    # Calculate regime distribution by year
    yearly_dist = df.groupby(['year', 'regime']).size().unstack(fill_value=0)
    yearly_pct = yearly_dist.div(yearly_dist.sum(axis=1), axis=0) * 100

    # Create stacked bar chart
    fig = go.Figure()

    colors = {
        0: '#00ff00',  # Green
        1: '#808080',  # Gray
        2: '#ff0000'   # Red
    }

    names = {
        0: 'Bullish',
        1: 'Neutral',
        2: 'Bearish'
    }

    for regime in [0, 1, 2]:
        if regime in yearly_pct.columns:
            fig.add_trace(go.Bar(
                name=names[regime],
                x=yearly_pct.index,
                y=yearly_pct[regime],
                marker_color=colors[regime],
                text=yearly_pct[regime].round(1).astype(str) + '%',
                textposition='inside'
            ))

    fig.update_layout(
        title='Regime Distribution by Year (2018-2025)',
        xaxis_title='Year',
        yaxis_title='Percentage of Trading Days',
        barmode='stack',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig, yearly_pct


def create_monthly_heatmap(df):
    """Create monthly regime heatmap"""

    # Calculate dominant regime for each month
    monthly_regime = df.groupby(['year', 'month'])['regime'].agg(lambda x: x.mode()[0])

    # Pivot for heatmap
    heatmap_data = monthly_regime.reset_index().pivot(index='month', columns='year', values='regime')

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        colorscale=[[0, 'green'], [0.5, 'gray'], [1, 'red']],
        colorbar=dict(
            title="Regime",
            tickvals=[0, 1, 2],
            ticktext=['Bullish', 'Neutral', 'Bearish']
        ),
        text=heatmap_data.values,
        hovertemplate='Year: %{x}<br>Month: %{y}<br>Regime: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title='Monthly Dominant Regime Heatmap (2018-2025)',
        xaxis_title='Year',
        yaxis_title='Month',
        height=400,
        template='plotly_white'
    )

    return fig


def main():
    """Main analysis function"""

    logger.info("Loading regime predictions...")
    df = get_regime_predictions()

    # Create output directory
    output_path = Path('outputs/visualizations')
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Create full timeline visualization
    logger.info("Creating full timeline visualization...")
    fig_timeline = create_full_timeline_visualization(df)
    timeline_path = output_path / 'regime_full_timeline.html'
    fig_timeline.write_html(str(timeline_path))
    logger.info(f"Saved timeline to {timeline_path}")

    # 2. Create yearly distribution
    logger.info("Creating yearly distribution...")
    fig_yearly, yearly_pct = create_yearly_distribution(df)
    yearly_path = output_path / 'regime_yearly_distribution.html'
    fig_yearly.write_html(str(yearly_path))
    logger.info(f"Saved yearly distribution to {yearly_path}")

    # 3. Create monthly heatmap
    logger.info("Creating monthly heatmap...")
    fig_heatmap = create_monthly_heatmap(df)
    heatmap_path = output_path / 'regime_monthly_heatmap.html'
    fig_heatmap.write_html(str(heatmap_path))
    logger.info(f"Saved heatmap to {heatmap_path}")

    # Print summary statistics
    print("\n" + "="*70)
    print("REGIME ANALYSIS SUMMARY (2018-2025)")
    print("="*70)

    print("\n1. OVERALL DISTRIBUTION:")
    overall_dist = df['regime'].value_counts(normalize=True).sort_index() * 100
    for regime in range(3):
        if regime in overall_dist.index:
            name = ['Bullish', 'Neutral', 'Bearish'][regime]
            print(f"  Regime {regime} ({name}): {overall_dist[regime]:.1f}%")

    print("\n2. YEARLY REGIME DISTRIBUTION (%):")
    print(yearly_pct.round(1).to_string())

    print("\n3. REGIME STATISTICS BY YEAR:")
    yearly_stats = df.groupby(['year', 'regime'])['ret'].agg(['mean', 'std', 'count'])
    yearly_stats.index = yearly_stats.index.set_names(['Year', 'Regime'])
    print("\nAverage Daily Returns by Year and Regime:")
    for year in df['year'].unique():
        print(f"\n{year}:")
        for regime in range(3):
            if (year, regime) in yearly_stats.index:
                stats = yearly_stats.loc[(year, regime)]
                mean_ret = stats['mean'] * 100
                std_ret = stats['std'] * 100
                count = stats['count']
                name = ['Bullish', 'Neutral', 'Bearish'][regime]
                print(f"  {name}: {mean_ret:+.3f}% ± {std_ret:.3f}% ({count} days)")

    print("\n4. NOTABLE PATTERNS:")

    # Find longest streaks
    regime_changes = df['regime'].ne(df['regime'].shift())
    regime_groups = regime_changes.cumsum()

    for regime in range(3):
        regime_df = df[df['regime'] == regime]
        if len(regime_df) > 0:
            # Find longest streak
            regime_streaks = regime_df.groupby(regime_groups[df['regime'] == regime]).size()
            longest_streak = regime_streaks.max()
            longest_idx = regime_streaks.idxmax()
            streak_dates = df[regime_groups == longest_idx]
            name = ['Bullish', 'Neutral', 'Bearish'][regime]
            print(f"\n  Longest {name} streak: {longest_streak} days")
            print(f"    From {streak_dates['date'].min().date()} to {streak_dates['date'].max().date()}")

    print("\n" + "="*70)
    print("Visualizations created:")
    print(f"  1. {timeline_path}")
    print(f"  2. {yearly_path}")
    print(f"  3. {heatmap_path}")
    print("="*70)

    return df, yearly_pct


if __name__ == "__main__":
    try:
        df, yearly_pct = main()
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
        raise