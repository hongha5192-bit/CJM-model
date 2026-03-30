# Market Analysis Report — 2026-03-30

Data as of 2026-03-27 | VNINDEX: 1,673 | Top 200 stocks by market cap

---

## 1. Data & Features Built

### Data Files

| File | Content |
|---|---|
| `top200_ohlcv.csv` | OHLCV + NetForeignVal/Vol, 200 tickers, 2018-2026 |
| `top200_pe_pb.csv` | PE, PB daily + rolling 256d percentiles, 200 tickers, 2018-2026 |
| `top200_features_ursi_bb.csv` | All computed indicators |
| `top200_sector.csv` | L1, L2 sector, McapClass, Freefloat |
| `VNINDEX_OHLCV.csv` | VNINDEX daily, 2016-2026 (2,459 rows) |
| `top100_quarterly/` | 41 quarterly CSVs (Top 100 by mktcap, 2016Q1-2026Q1) |
| `vn100_all_ohlcv.csv` | 220 VN100 tickers, 2016-2026 |

### Indicators

| # | Indicator | Parameters | Columns |
|---|---|---|---|
| 1 | Ultimate RSI | length=14, RMA, signal EMA(14) | UltRSI, UltRSI_Signal |
| 2 | Bollinger Bands 20d | SMA(20), +/-2.0 StdDev, ddof=0 | BB_Basis, BB_Upper, BB_Lower, BB_Width, BB_PctB |
| 3 | Bollinger Bands 256d | SMA(256), +/-2.5 StdDev, ddof=0 | BB256_Basis, BB256_Upper, BB256_Lower, BB256_Width, BB256_PctB |
| 4 | 3M High & Drawdown | 63-day rolling high | High_3M, Drawdown_3M |
| 5 | EMA 200 | Computed on-the-fly | Not stored as column |
| 6 | PE Rolling Percentile | 256-day window, positive PE only | PE_Pctile_256 (in top200_pe_pb.csv) |
| 7 | PB Rolling Percentile | 256-day window | PB_Pctile_256 (in top200_pe_pb.csv) |

### Sector Map (L2)

Source: `dbo.Sector_Map` — 19 sub-sectors, 157/200 mapped, 43 unknown.

---

## 2. UltRSI Oversold (<=20): 13 stocks

| Ticker | UltRSI | Signal | Close | L2 |
|---|---|---|---|---|
| FTS | 11.4 | 13.8 | 25,600 | Brokerage |
| BMP | 13.1 | 12.3 | 128,200 | Industrials |
| CTS | 13.4 | 14.2 | 26,850 | Brokerage |
| HVN | 13.6 | 16.3 | 21,950 | Transportation |
| BCM | 14.4 | 14.9 | 54,300 | IP |
| VDS | 15.1 | 20.9 | 14,400 | Brokerage |
| VND | 15.2 | 18.9 | 15,850 | Brokerage |
| FPT | 16.2 | 15.2 | 76,100 | IT |
| CMG | 16.3 | 14.6 | 29,150 | IT |
| GAS | 17.3 | 31.5 | 82,100 | Energy |
| PLX | 19.1 | 34.2 | 42,250 | Energy |
| BID | 19.4 | 21.8 | 39,850 | Banking |
| BSI | 19.7 | 21.1 | 34,300 | Brokerage |

Dominated by Brokerage (5) and Energy (2).

---

## 3. Below BB20 Lower Band: 2 stocks

| Ticker | Close | BB Lower | %B | L2 |
|---|---|---|---|---|
| TMS | 39,100 | 39,991 | -0.516 | Transportation |
| TID | 21,000 | 21,017 | -0.009 | Unknown |

---

## 4. Between BB20 Middle & Lower Band: 111 stocks

### Closest to breaking down (<5% drop needed)

| Ticker | Close | BB Lower | Drop% | L2 |
|---|---|---|---|---|
| TDM | 53,200 | 53,165 | 0.07% | Utilities |
| SCG | 63,600 | 63,310 | 0.46% | Unknown |
| VSH | 43,300 | 43,079 | 0.51% | Utilities |
| CHP | 28,200 | 27,889 | 1.10% | Utilities |
| KDC | 49,350 | 48,667 | 1.38% | Consumer Staples |
| STB | 59,900 | 58,925 | 1.63% | Banking |
| QNS | 47,600 | 46,809 | 1.66% | Consumer Staples |
| SBH | 40,600 | 39,879 | 1.78% | Unknown |
| DHG | 101,000 | 99,139 | 1.84% | Health Care |

Summary: < 3% away: 3 stocks | < 5%: 9 stocks | Median: 6.71%

---

## 5. Below BB256 Lower Band (+-2.5 StdDev): 4 stocks

| Ticker | Close | BB256 Lower | %B | L2 |
|---|---|---|---|---|
| TMS | 39,100 | 39,647 | -0.193 | Transportation |
| FPT | 76,100 | 79,454 | -0.090 | IT |
| DGC | 51,800 | 54,010 | -0.032 | Chemicals |
| ACV | 45,400 | 45,462 | -0.003 | Transportation |

### Between BB256 Middle & Lower: 87 stocks

Closest: FTS (1.78%), TDM (2.18%), KDC (2.73%)
Median drop needed: 21.09%

---

## 6. 3M High Drawdown >= 20%: 73 stocks (36.5%)

### Severity

| Decline | Count | Notable names |
|---|---|---|
| > 50% | 2 | F88 (-56.9%), DCV (-51.8%) |
| 30-40% | 18 | OIL (-40.2%), PLX (-40.0%), GAS (-37.6%), FPT (-30.0%), BID (-29.6%), VIC (-30.2%) |
| 20-30% | 52 | VCB (-25.4%), CTG (-20.0%), CMG (-25.3%), SSI (-22.7%), SAB (-22.7%) |

### Sector breakdown

| L2 | Count |
|---|---|
| Brokerage | 9 (FTS, VND, VDS, CTS, VIX, SHS, SSI, BSI, MBS) |
| Residential | 9 (VRE, VHM, DXS, CEO, KDH, DIG, PDR, DXG, CII) |
| Energy | 6 (OIL, PLX, GAS, BSR, PVS, PVD) |
| Transportation | 6 (PVT, HVN, ACV, VJC, VTP, HAH) |
| Banking | 5 (KLB, BID, VCB, NVB, CTG) |
| IT | 4 (VGI, FPT, CMG, FOX) |

---

## 7. 3M High Drawdown >= 15%: 104 stocks (52%)

Additional 31 stocks in the 15-20% band including: MCH (-19.9%), VNM (-18.5%), TCB (-19.8%), VPB (-15.9%), HDB (-15.7%), KBC (-19.6%), HCM (-19.1%)

---

## 8. Below EMA200: 115 stocks (57.5%)

### Furthest below (>15%)

| Ticker | Close | EMA200 | % vs EMA | L2 |
|---|---|---|---|---|
| DGC | 51,800 | 83,232 | -37.76% | Chemicals |
| FTS | 25,600 | 33,760 | -24.17% | Brokerage |
| VEF | 95,200 | 124,995 | -23.84% | Unknown |
| DCV | 140,000 | 183,716 | -23.80% | Unknown |
| DXS | 6,760 | 8,648 | -21.83% | Residential |
| FPT | 76,100 | 96,211 | -20.90% | IT |
| CMG | 29,150 | 36,680 | -20.53% | IT |

### Just crossed below (<1%)

ACB (-0.02%), HAH (-0.02%), E1VFVN30 (-0.06%), SIP (-0.11%), CII (-0.24%), VNM (-0.55%), VPB (-0.61%)

### Sector breakdown (below EMA200)

| L2 | Count |
|---|---|
| Residential | 15 |
| Banking | 14 |
| Brokerage | 14 |
| Consumer Staples | 9 |
| Transportation | 6 |
| Utilities | 5 |

---

## 9. Historical Comparison: Current vs Major Bottoms

### Major Bottom Dates

11/7/2018, 3/1/2019, 31/3/2020, 28/1/2021, 19/7/2021, 16/5/2022, 15/11/2022, 31/10/2023, 9/4/2025

### Indicator Readings (% of stocks)

| Date | VNI | URSI<=20 | BB<Lower | BB256<Low | DD>=20% | DD>=15% | <EMA200 |
|---|---|---|---|---|---|---|---|
| 2018-07-11 | 893 | 41.6% | 8.4% | 0.0% | 63.6% | 74.7% | 84.4% |
| 2019-01-03 | 878 | 26.9% | 19.2% | 0.0% | 32.7% | 49.4% | 64.1% |
| 2020-03-31 | 663 | 63.8% | 8.6% | 40.5% | 76.1% | 84.7% | 90.8% |
| 2021-01-28 | 1,024 | 8.4% | 65.7% | 0.0% | 49.4% | 72.5% | 27.0% |
| 2021-07-19 | 1,244 | 12.6% | 22.0% | 0.5% | 40.7% | 63.7% | 29.1% |
| 2022-05-16 | 1,172 | 62.8% | 32.4% | 14.9% | 77.7% | 86.7% | 81.4% |
| 2022-11-15 | 912 | 67.2% | 49.2% | 26.5% | 79.9% | 86.8% | 92.1% |
| 2023-10-31 | 1,028 | 45.7% | 45.2% | 4.8% | 53.7% | 74.5% | 77.7% |
| 2025-04-09 | 1,094 | 66.7% | 76.6% | 34.9% | 69.3% | 86.5% | 84.4% |
| **NOW** | **1,673** | **6.5%** | **1.0%** | **2.0%** | **36.5%** | **52.0%** | **57.5%** |

---

## 10. Conclusion

**The current market is NOT at a major bottom.** Key evidence:

1. **Oversold breadth is too shallow.** Only 6.5% of stocks have URSI<=20 vs 40-67% at true bottoms. No capitulation selling.

2. **Almost no Bollinger Band breakdowns.** 1% below BB20 lower band (vs 8-77% at bottoms). 2% below BB256 (vs 15-41% at deep corrections). Market is drifting, not crashing.

3. **Drawdown metrics match early-stage correction.** 36.5% stocks down 20%+ from 3M high — comparable to Jan 2019 (32.7%) and Jul 2021 (40.7%), not deep bottoms like Mar 2020 (76%) or Nov 2022 (80%).

4. **EMA200 breadth at 57.5%** is concerning but not extreme. True bottoms reach 80-92%.

5. **VNINDEX at 1,673** is well above prior major bottom levels (663-1,172 for deep corrections).

**Assessment**: The market is in a correction, roughly comparable to early stages of Jan 2019 or Jul 2021 — moderate stress, not capitulation. For a major bottom signal, we'd need URSI oversold breadth at 40%+ and BB breakdown at 20%+. Currently about 1/3 to 1/2 of the way to typical major bottom readings.

### Weakest Sectors (consistent across all indicators)

1. **Brokerage** — most oversold, most below EMA200, deepest drawdowns
2. **Residential** — broad weakness, 15 stocks below EMA200
3. **Energy** — OIL, PLX, GAS down 37-40% from 3M highs
4. **Banking** — 14 below EMA200, BID/VCB/CTG in 20%+ drawdown
5. **Transportation** — HVN, ACV, VJC all down 27-30%

---

## 11. Valuation Analysis: PE/PB at Current Level vs Major Bottoms

### Market-Wide PE/PB (absolute)

| Metric | May-22 | Nov-22 | Oct-23 | Apr-25 | **NOW** |
|---|---|---|---|---|---|
| **Median PE** | 12.7 | **8.3** | 14.4 | 14.1 | **15.0** |
| **Median PB** | 1.68 | **1.11** | 1.36 | 1.40 | **1.83** |
| PE < 10 | 38% | **61%** | 28% | 29% | **27%** |
| PB < 1 | 13% | **43%** | 23% | 25% | **10%** |

Current median PE (15.0) is above all 4 recent bottoms. Only 10% of stocks trade below book value vs 43% at Nov 2022.

### Sector PE — Current vs Cheapest Bottom

| Sector | Cheapest Bottom | NOW | Premium vs Bottom |
|---|---|---|---|
| Banking | 4.8 (Nov-22) | 7.9 | +64% |
| Brokerage | 5.9 (Nov-22) | 14.7 | +150% |
| Chemicals | 2.9 (Nov-22) | 11.0 | +279% |
| Residential | 11.9 (Nov-22) | 20.1 | +69% |
| Energy | 13.9 (Oct-23) | 22.0 | +59% |
| IT | 14.3 (Nov-22) | 17.4 | +21% (cheapest relative) |

### Sector PB — Current vs Cheapest Bottom

| Sector | Cheapest Bottom | NOW | Premium vs Bottom |
|---|---|---|---|
| Banking | 1.00 (Nov-22) | 1.25 | +25% |
| Brokerage | 0.77 (Nov-22) | 1.73 | +125% |
| Chemicals | 0.95 (Nov-22) | 1.83 | +92% |
| Residential | 0.84 (Nov-22) | 1.34 | +59% |
| Energy | 0.71 (Nov-22) | 1.79 | +153% |
| Consumer Staples | 1.44 (Oct-23) | 1.86 | +30% |

---

## 12. Rolling 256-Day PE Percentile — The Key Valuation Signal

Using a rolling 256-day window to measure where each stock's PE sits relative to its own past year. This adapts to the current earnings cycle and avoids pollution from old regimes.

### Market-Wide PE Percentile Breadth

| Period | Median Pctile | PE <= 10th pctile | PE <= 20th pctile |
|---|---|---|---|
| **May 2022** | **3%** | 97/160 (61%) | 109/160 (68%) |
| **Nov 2022** | **0%** | **133/164 (81%)** | **145/164 (88%)** |
| Oct 2023 | 56% | 26/159 (16%) | 38/159 (24%) |
| **Apr 2025** | **1%** | **107/172 (62%)** | **116/172 (67%)** |
| **NOW** | **22%** | 64/184 (35%) | 90/184 (49%) |

At deep bottoms (Nov-22, Apr-25), 81-88% of stocks sit at their 1-year PE floor (<=20th percentile). Currently 49% — significant stress but not capitulation.

### Top Bouncers — Rolling 256d PE Percentile at Bottoms vs Now

| Ticker | L2 | May-22 | Nov-22 | Oct-23 | Apr-25 | **NOW** |
|---|---|---|---|---|---|---|
| **DGC** | Chemicals | 0% | 0% | 90% | 0% | **0%** |
| **SGP** | Unknown | 0% | 0% | 62% | 79% | **0%** |
| **DDV** | Chemicals | 6% | 0% | 26% | 0% | **2%** |
| **BSI** | Brokerage | 0% | 10% | 19% | 82% | **2%** |
| **CTS** | Brokerage | 0% | 2% | 19% | 4% | **2%** |
| **MBS** | Brokerage | 0% | 0% | 51% | 0% | **2%** |
| **MWG** | Retailing | 22% | 0% | 99% | 0% | **3%** |
| **FRT** | Retailing | 5% | 0% | n/a | 0% | **7%** |
| **HAG** | Consumer Staples | 0% | 0% | 79% | 82% | **8%** |
| **SHS** | Brokerage | 1% | 14% | 25% | 7% | **9%** |
| **HPG** | Metals | 0% | 7% | 56% | 0% | **15%** |
| **CEO** | Residential | 6% | 0% | 70% | 0% | **43%** |
| **GEE** | Industrials | n/a | n/a | n/a | 37% | **45%** |
| **DCM** | Chemicals | 0% | 0% | 91% | 0% | **97%** |

### Valuation Verdict on Top Bouncers

**Already at 1-year valuation floor (PE pctile < 10%) — setup forming:**
- **DGC** (0%), **SGP** (0%) — at absolute 1-year low PE
- **DDV** (2%), **BSI** (2%), **CTS** (2%), **MBS** (2%), **MWG** (3%) — near floor
- **FRT** (7%), **HAG** (8%), **SHS** (9%) — bottom decile

**Mid-range — no valuation edge yet:**
- **HPG** (15%), **CEO** (43%), **GEE** (45%)

**Avoid — expensive relative to own history:**
- **DCM** (97%) — near 1-year PE ceiling, already re-rated on fertilizer upcycle. NOT a value bounce play this time despite being a repeat bouncer.

### Valuation Conclusion

1. **49% of stocks are in their bottom-20% PE zone** (rolling 256d). This is significant but short of the 67-88% seen at major bottoms.

2. **Among proven bouncers, 10 of 14 are already at their 1-year valuation floor.** The valuation setup is forming for DGC, SGP, DDV, BSI, CTS, MBS, MWG, FRT, HAG, SHS.

3. **DCM is the trap.** At the 97th PE percentile, it's priced for perfection. History shows it bounces at bottoms, but only when its PE is at the floor (0% percentile at 3 of 4 prior bottoms). Current setup does NOT support a DCM bounce play.

4. **Banking (sector PE 7.9x, +64% vs bottom)** is not distressed. PB at 1.25x vs 1.00x at Nov 2022. Banking needs more compression before it becomes a valuation buy.

5. **The gap between current and bottom-level valuations** confirms the technical picture: the market is in a correction, roughly 50-60% of the way to typical bottom valuation levels.

---

## 13. PE Percentile Lookback Window Analysis

At any major bottom, we only have data available up to that date. Different lookback windows give different signals.

### Two Types of Bottoms

- **Type A — Valuation bottoms** (PE compresses): May-22, Nov-22, Apr-25
- **Type B — Sentiment bottoms** (PE stays mid-range): Oct-23

### PE Percentile Breadth (% of stocks with PE pctile <= 20%)

| Period | 256d | 3Y | 5Y | ALL |
|---|---|---|---|---|
| May-22 (A) | **65%** | 36% | 36% | 36% |
| Nov-22 (A) | **86%** | **61%** | 60% | 60% |
| Oct-23 (B) | 23% | 27% | 23% | 23% |
| Apr-25 (A) | **67%** | 33% | 30% | 28% |
| **NOW** | **48%** | **41%** | 31% | 21% |

### Recommendation: Use 256d + 3Y Together

- **256d** is the PRIMARY signal — most discriminating for Type A bottoms (65-86% at bottoms vs 48% now)
- **3Y** is the SECONDARY signal — captures cycle-adjusted picture
- **5Y/ALL** add no incremental insight over 3Y

### Reading the Dual Signal

| 256d | 3Y | Interpretation |
|---|---|---|
| High (65%+) | High (60%+) | Definite bottom (Nov 2022) |
| High (65%+) | Moderate (33-36%) | Sharp correction bottom (May-22, Apr-25) |
| Low (23%) | Low (27%) | Sentiment bottom — need other signals (Oct-23) |
| **48%** | **41%** | **Current: cheap on 3Y, not extreme on 256d — halfway** |

---

## 14. PB Deep Value Analysis

### PB < 1 Count at Each Period

| Period | PB < 1 | PB <= 1.2 |
|---|---|---|
| May 2022 | 23 (13%) | 43 (25%) |
| **Nov 2022** | **75 (43%)** | **93 (53%)** |
| Oct 2023 | 41 (23%) | 71 (39%) |
| Apr 2025 | 45 (25%) | 75 (41%) |
| **NOW** | **20 (10%)** | **40 (20%)** |

Only 20 stocks below book value — lowest count. At bottoms this reaches 23-75.

### PB < 1 Stocks Bounce HARDER Than the Market

| Bottom | PB<1 Median 2W | Market Median 2W | Edge |
|---|---|---|---|
| Nov 2022 | +31.7% | +19.1% | **+12.6pp** |
| Apr 2025 | +13.5% | +10.2% | **+3.3pp** |

### PB Percentile Breadth (256d)

| Period | Median | <=10% | <=20% |
|---|---|---|---|
| May 2022 | 10% | 50% | 60% |
| **Nov 2022** | **0%** | **89%** | **91%** |
| Oct 2023 | 40% | 23% | 33% |
| Apr 2025 | 0% | 63% | 71% |
| **NOW** | **36%** | **21%** | **32%** |

PB breadth at 32% — about 1/3 of the way to bottom levels (60-91%).

### Sector PB Percentile (256d)

| Sector | At Bottoms (May/Nov-22, Apr-25) | NOW | Reading |
|---|---|---|---|
| Transportation | 0-33% | **32%** | Near bottom zone |
| Residential | 0-3% | **25%** | Getting there |
| Brokerage | 0-2% | **26%** | Getting there |
| Banking | 0% | **33%** | Mid-range |
| Energy | 0-10% | **88%** | **Expensive — already re-rated** |
| Chemicals | 0-32% | **55%** | Mid-range |

### PE vs PB: Which Is Cheaper for Top Bouncers?

| Ticker | PE Pctile | PB Pctile | Signal |
|---|---|---|---|
| **DGC** | 0% | 0% | **BOTH at floor** — strongest |
| **CTS** | 2% | 2% | **BOTH at floor** — strongest |
| **BSI** | 2% | 2% | **BOTH at floor** — strongest |
| **MBS** | 2% | 3% | **BOTH at floor** — strongest |
| SGP | 0% | 6% | PE floor, PB near floor |
| DDV | 2% | 14% | PE strong, PB ok |
| MWG | 3% | 47% | PE cheap only, PB NOT confirming |
| **DCM** | **97%** | **97%** | **BOTH expensive — AVOID** |

### Current PB < 1.2 Stocks (40 names)

Dominated by **Banking** (10): MSB (0.83), VAB (0.85), OCB (0.88), BAB (0.89), KLB (0.90), NAB (0.97), ABB (0.97), SHB (1.04), TPB (1.07), EVF (1.10)

**Residential** (6): DXS (0.62), NVL (0.70), DIG (0.94), IJC (0.81), NLG (1.12), DXG (1.14)

**Metals** (2): HSG (0.81), NKG (0.81)

---

## 15. Stock Selection Scorecard

### 7 Bottom Signals

| # | Signal | Threshold |
|---|---|---|
| 1 | URSI oversold | <= 20 |
| 2 | BB20 lower zone | %B < 0.3 |
| 3 | 3M drawdown | >= 20% |
| 4 | Below EMA200 | Close < EMA200 |
| 5 | PE percentile (256d) | <= 10% |
| 6 | PB percentile (256d) | <= 10% |
| 7 | Absolute PB | <= 1.2 |

### Tier 1 — Highest Conviction (Score 5-6 + Proven Bouncer)

| Ticker | Score | PB | PE | DD 3M | Sector | Signals |
|---|---|---|---|---|---|---|
| **DGC** | 5 | 1.31 | 6.9 | -36.8% | Chemicals | BB, DD, EMA, PE_pct(0%), PB_pct(0%) |
| **BSI** | 5 | 1.52 | 17.0 | -21.3% | Brokerage | URSI, DD, EMA, PE_pct(2%), PB_pct(2%) |
| **CTS** | 5 | 2.00 | 9.1 | -24.8% | Brokerage | URSI, DD, EMA, PE_pct(2%), PB_pct(2%) |

### Tier 2 — Strong Setup (Score 5-6, strong numbers)

| Ticker | Score | PB | PE | DD 3M | Sector | Key Feature |
|---|---|---|---|---|---|---|
| **BMP** | 6 | 3.65 | 8.5 | -30.6% | Industrials | Highest score (6/7 signals) |
| **FPT** | 5 | 3.55 | 13.8 | -30.0% | IT | Blue chip, rarely this oversold |
| **HVN** | 5 | 11.34 | 8.6 | -30.1% | Transportation | URSI 13.6, PE at bottom |
| **ACV** | 5 | 2.33 | 15.0 | -29.9% | Transportation | 5 signals, deep DD |
| **VND** | 5 | 1.15 | 11.9 | -26.1% | Brokerage | PB near 1, URSI oversold |
| **CMG** | 5 | 2.22 | 16.8 | -25.3% | IT | URSI 16.3, deep below EMA200 |
| **BCM** | 5 | 2.47 | 16.3 | -35.1% | IP | URSI 14.4, deepest DD in IP |
| **DBC** | 5 | 1.11 | 5.9 | -21.8% | Consumer Staples | PB near 1, PE 5.9x, triple cheap |
| **DIG** | 5 | 0.94 | 16.5 | -22.7% | Residential | PB < 1, deep value |
| **DNH** | 5 | 3.46 | 20.0 | -27.6% | Unknown | All 5 signals aligned |
| **NTP** | 5 | 2.24 | 9.6 | -31.3% | Unknown | PE 9.6x, deep DD |
| **VDS** | 5 | 1.29 | 13.8 | -25.6% | Brokerage | URSI 15.1, PE floor |
| **FTS** | 5 | 2.01 | 22.1 | -27.7% | Brokerage | URSI 11.4 (most oversold broker) |
| **ACG** | 5 | 1.19 | 11.6 | -10.6% | Other Materials | PE+PB both at 2%, PB near 1 |

### Tier 3 — Watchlist (Score 4, repeat bouncers or notable names)

| Ticker | Score | PB | PE | Sector | Note |
|---|---|---|---|---|---|
| **MBS** | 4 | 1.97 | 13.6 | Brokerage | 3x repeat bouncer |
| **SGP** | 4 | 1.88 | 14.3 | Unknown | 3x repeat bouncer, PE at 0% |
| **BID** | 4 | 1.67 | 9.3 | Banking | URSI 19.4, cheapest large-cap bank |
| **SSI** | 4 | 1.76 | 14.2 | Brokerage | DD -22.7%, PE pctile 3% |
| **KLB** | 4 | 0.90 | 4.1 | Banking | PB < 1, PE 4.1x, deepest value bank |
| **PLX** | 4 | 2.05 | 22.5 | Energy | URSI 19.1, DD -40%, but PB 81st pctile — caution |

---

## 16. Sector Conclusion

| Sector | Avg Score | Score>=4 | Med PE | Med PB | Verdict |
|---|---|---|---|---|---|
| **Brokerage** | **3.0** | **7** | 14.7 | 1.73 | **#1 sector for bottom play.** Most signals triggered. BSI, CTS, VND, FTS, VDS, SSI, MBS all lining up. Historically leads every bounce. |
| **IT** | **3.0** | **2** | 17.4 | 4.60 | FPT and CMG deeply oversold. High conviction blue-chip IT at these levels. |
| **Transportation** | **2.3** | **3** | 15.0 | 2.56 | HVN, ACV, TMS — deep drawdowns. HVN cheapest at PE 8.6x. |
| **Energy** | 2.3 | 1 | 22.0 | 1.79 | PLX only candidate. **PB at 81st percentile — already re-rated. AVOID on valuation.** |
| **IP** | 2.2 | 1 | 15.0 | 2.21 | BCM only candidate (-35% DD, URSI oversold). |
| **Banking** | 1.8 | 3 | 7.9 | 1.25 | Cheapest PE but **historically lags bounces.** BID best large-cap. Mid-tier KLB, NAB have PB < 1. |
| **Residential** | 1.7 | 1 | 20.1 | 1.34 | DIG only candidate (PB 0.94). Most stocks not oversold enough yet. Needs more selling. |
| **Consumer Staples** | 1.5 | 2 | 13.2 | 1.86 | DBC (PB 1.11, PE 5.9x) standout value. HAG (repeat bouncer) close to triggering. |
| **Chemicals** | 1.5 | 1 | 11.0 | 1.83 | DGC is the top pick. **DCM is a TRAP (PE+PB both 97th pctile). DDV not enough signals yet.** |
| **Retailing** | 0.8 | 0 | 18.3 | 2.93 | MWG and FRT cheap on PE but not enough signals. Early stage. |
| **Metals** | 1.0 | 0 | 23.0 | 1.19 | HPG/HSG low PB but high PE. Not cheap enough. |

---

## 17. Final Assessment

**The market is in a correction, not at a bottom.** But individual stocks are diverging significantly:

- **34 stocks (17%)** already have 4+ bottom signals firing — the watchlist is forming
- **Brokerage** is building the strongest sector setup (7 stocks at score >= 4)
- **DGC, BSI, CTS** are highest-conviction: proven bouncers + top scores + PE/PB both at floor
- **FPT at score 5** is the blue-chip opportunity — rarely this oversold
- **Banking is cheap but lags** every bounce — BID is the least-bad option
- **DCM and Energy are traps** — prices fell from highs but valuations already re-rated
- **PB < 1 at bottoms is a proven alpha signal** (+3.3 to +12.6pp edge) — currently only 20 stocks qualify, needs 40+ for bottom

### Trigger to Deploy

Wait for breadth to reach:
- **PE pctile (256d) breadth**: 65%+ of stocks at <= 20th percentile (currently 48%)
- **URSI oversold breadth**: 40%+ of stocks (currently 6.5%)
- **PB < 1 count**: 40+ stocks (currently 20)

When 2 of 3 triggers fire, deploy into Tier 1 and Tier 2 names.
