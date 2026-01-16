# CVD Analysis Results Storage Summary

## Storage Location
All CVD analysis results are stored in the `/outputs/` directory within the project:
`/Users/hanguyen/CJMModel/jump_lambda_vn/outputs/`

## Files Generated

### 1. Core CVD Data Files

#### `vnindex_cvd_analysis.csv` (311 KB)
- **Primary CVD calculation results**
- Contains all daily data from 2016-2025
- Columns included:
  - date
  - OHLC prices (OPENINDEX, HIGHESTINDEX, LOWESTINDEX, CLOSEINDEX)
  - TOTALMATCHVOLUME
  - TRADING_VALUE_BILLIONS
  - BUY_VOLUME (calculated)
  - SELL_VOLUME (calculated)
  - VOLUME_DELTA (calculated)
  - CVD (Cumulative Volume Delta)
  - CVD_SMOOTH (7-period smoothed)
  - DIVERGENCE flags

#### `cvd_regime_statistics.csv` (401 KB)
- **CVD analysis by market regime**
- Merged with HMM states
- Additional columns:
  - hmm_state (0=Bullish, 1=Neutral, 2=Bearish)
  - CVD_CHANGE
  - CVD_PCT_CHANGE
  - VOLUME_DELTA_RATIO
  - returns

#### `cvd_yearly_statistics.csv` (1.5 KB)
- **Aggregated yearly statistics**
- Summary by year (2016-2025)
- Columns:
  - Year
  - Trading_Days
  - Total_Buy_Vol
  - Total_Sell_Vol
  - Net_Delta
  - Buy_Sell_Ratio
  - Avg_Daily_Delta
  - Price_Return_%
  - CVD_Change
  - Avg_Daily_Volume

### 2. Visualization Files

#### `cvd_analysis.png` (211 KB)
- **4-panel visualization showing:**
  1. VNINDEX Price with Volume bars
  2. Daily Volume Delta (green/red bars)
  3. Cumulative Volume Delta with smoothing
  4. Price vs CVD overlay with divergences marked

#### `cvd_regime_analysis.png` (138 KB)
- **6-panel regime analysis:**
  1. Volume Delta distribution by regime (boxplot)
  2. CVD change distribution by regime (histogram)
  3. Buy/Sell ratio by regime (bar chart)
  4. Average volume by regime
  5. CVD evolution colored by regime
  6. Volume Delta efficiency by regime

#### `cvd_yearly_visualization.png` (320 KB)
- **Comprehensive yearly analysis:**
  1. Price & CVD overlay (2016-2025)
  2. Daily Volume Delta with 30-day MA
  3. Yearly Buy/Sell ratios
  4. Yearly Net Volume Delta
  5. Monthly heatmap of Volume Delta
  6. Quarterly CVD growth rates
  7. Volume Delta distribution (violin plots)
  8. Summary statistics table

#### `cvd_trend_analysis.png` (274 KB)
- **Trend analysis charts:**
  1. CVD with linear trend line
  2. CVD momentum (30-day change)
  3. Rolling Buy/Sell ratio (60-day)
  4. Price-CVD correlation (120-day rolling)
  5. Volume Delta with extremes highlighted
  6. CVD regime distribution by year

## How to Access the Results

### Reading CSV Files in Python:
```python
import pandas as pd

# Main CVD data
cvd_df = pd.read_csv('outputs/vnindex_cvd_analysis.csv', parse_dates=['date'])

# Regime analysis
regime_df = pd.read_csv('outputs/cvd_regime_statistics.csv', parse_dates=['date'])

# Yearly summary
yearly_df = pd.read_csv('outputs/cvd_yearly_statistics.csv')
```

### Viewing Images:
All PNG files can be opened with any image viewer or imported into reports.

### Key Columns in Main CVD File:

| Column | Description | Type |
|--------|-------------|------|
| date | Trading date | datetime |
| CLOSEINDEX | VNINDEX closing price | float |
| TOTALMATCHVOLUME | Daily trading volume | int |
| BUY_VOLUME | Calculated buying volume | float |
| SELL_VOLUME | Calculated selling volume | float |
| VOLUME_DELTA | Buy - Sell volume | float |
| CVD | Cumulative Volume Delta | float |
| CVD_SMOOTH | Smoothed CVD (7-RMA) | float |
| DIVERGENCE | -1=Bearish, 0=None, 1=Bullish | int |

## File Sizes Summary
- Total storage used: ~1.7 MB
- Largest file: cvd_regime_statistics.csv (401 KB)
- Smallest file: cvd_yearly_statistics.csv (1.5 KB)

## Usage Notes
1. All CSV files are comma-separated with headers
2. Dates are in YYYY-MM-DD format
3. Volume values are in shares (not lots)
4. CVD values are cumulative from 2016-05-27
5. Images are 100 DPI, suitable for reports

## Quick Access Commands

```bash
# View file structure
ls -la outputs/*cvd* outputs/*vnindex_cvd*

# Check file sizes
du -h outputs/*cvd* outputs/*vnindex_cvd*

# Quick preview of data
head outputs/vnindex_cvd_analysis.csv
tail outputs/cvd_yearly_statistics.csv

# Count rows in main file
wc -l outputs/vnindex_cvd_analysis.csv
```