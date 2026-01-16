#!/usr/bin/env python
"""
Create candlestick chart for VNINDEX from 2019
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading VNINDEX data...")
df = pd.read_csv('data/vnindex_full.csv', parse_dates=['date'])

# Filter from 2019 onwards
df_2019 = df[df['date'] >= '2019-01-01'].copy()
print(f"Data from 2019: {len(df_2019)} trading days")
print(f"Date range: {df_2019['date'].min().date()} to {df_2019['date'].max().date()}")

# Remove rows with zero OHLC values
df_clean = df_2019[
    (df_2019['OPENINDEX'] > 0) &
    (df_2019['HIGHESTINDEX'] > 0) &
    (df_2019['LOWESTINDEX'] > 0) &
    (df_2019['CLOSEINDEX'] > 0)
].copy()

print(f"After removing zero values: {len(df_clean)} trading days")

# Create candlestick chart
fig, ax = plt.subplots(figsize=(20, 10))

# Set up data
dates = df_clean['date'].values
opens = df_clean['OPENINDEX'].values
highs = df_clean['HIGHESTINDEX'].values
lows = df_clean['LOWESTINDEX'].values
closes = df_clean['CLOSEINDEX'].values

# Colors
up_color = '#26a69a'  # Green for bullish
down_color = '#ef5350'  # Red for bearish
up_edge = '#26a69a'
down_edge = '#ef5350'

# Plot candlesticks
width = 0.6  # Width of candlestick body in days

for i in range(len(df_clean)):
    date = dates[i]
    open_price = opens[i]
    high = highs[i]
    low = lows[i]
    close = closes[i]

    # Determine color (up or down)
    if close >= open_price:
        color = up_color
        edge_color = up_edge
        body_height = close - open_price
        body_bottom = open_price
    else:
        color = down_color
        edge_color = down_edge
        body_height = open_price - close
        body_bottom = close

    # Plot high-low line (wick)
    ax.plot([date, date], [low, high], color=edge_color, linewidth=0.8, zorder=1)

    # Plot body (rectangle)
    rect = Rectangle(
        (date, body_bottom),
        width=pd.Timedelta(days=width),
        height=body_height,
        facecolor=color,
        edgecolor=edge_color,
        linewidth=0.8,
        zorder=2
    )
    ax.add_patch(rect)

# Formatting
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('VNINDEX', fontsize=14, fontweight='bold')
ax.set_title('VNINDEX Candlestick Chart (2019 - Present)', fontsize=18, fontweight='bold', pad=20)

# Format x-axis
ax.xaxis.set_major_locator(YearLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y'))
ax.xaxis.set_minor_locator(MonthLocator((1, 4, 7, 10)))
plt.xticks(rotation=0, ha='center')

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=up_color, edgecolor=up_edge, label='Bullish (Close > Open)'),
    Patch(facecolor=down_color, edgecolor=down_edge, label='Bearish (Close < Open)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

# Add statistics box
stats_text = f"""Statistics (2019-Present):
Start: {df_clean['CLOSEINDEX'].iloc[0]:.2f}
End: {df_clean['CLOSEINDEX'].iloc[-1]:.2f}
Change: {((df_clean['CLOSEINDEX'].iloc[-1] / df_clean['CLOSEINDEX'].iloc[0] - 1) * 100):.2f}%
High: {df_clean['HIGHESTINDEX'].max():.2f}
Low: {df_clean['LOWESTINDEX'].min():.2f}
Days: {len(df_clean)}"""

ax.text(0.98, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save plot
output_dir = Path('outputs/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'vnindex_candlestick_2019.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Candlestick chart saved to: {output_path}")

# Also save high-res version
output_path_hires = output_dir / 'vnindex_candlestick_2019_hires.png'
plt.savefig(output_path_hires, dpi=300, bbox_inches='tight')
print(f"✅ High-res version saved to: {output_path_hires}")

plt.close()

print("\n" + "="*60)
print("CHART GENERATION COMPLETE")
print("="*60)
