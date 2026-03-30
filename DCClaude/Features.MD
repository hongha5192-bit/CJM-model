// This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
// © LuxAlgo

//@version=5
indicator("Ultimate RSI [LuxAlgo]", "LuxAlgo - Ultimate RSI")
//------------------------------------------------------------------------------
//Settings
//-----------------------------------------------------------------------------{
length = input.int(14, minval = 2)
smoType1 = input.string('RMA', 'Method', options = ['EMA', 'SMA', 'RMA', 'TMA'])
src = input(close, 'Source')

arsiCss = input(color.silver, 'Color', inline = 'rsicss')
autoCss = input(true, 'Auto', inline = 'rsicss')

//Signal Line
smooth = input.int(14, minval = 1, group = 'Signal Line')
smoType2 = input.string('EMA', 'Method', options = ['EMA', 'SMA', 'RMA', 'TMA'], group = 'Signal Line')

signalCss = input(#ff5d00, 'Color', group = 'Signal Line')

//OB/OS Style
obValue = input.float(80, 'Overbought', inline = 'ob', group = 'OB/OS Style')
obCss = input(#089981, '', inline = 'ob', group = 'OB/OS Style')
obAreaCss = input(color.new(#089981, 80), '', inline = 'ob', group = 'OB/OS Style')

osValue = input.float(20, 'Oversold    ', inline = 'os', group = 'OB/OS Style')
osCss = input(#f23645, '', inline = 'os', group = 'OB/OS Style')
osAreaCss = input(color.new(#f23645, 80), '', inline = 'os', group = 'OB/OS Style')

//-----------------------------------------------------------------------------}
//Functions
//-----------------------------------------------------------------------------{
ma(x, len, maType)=>
    switch maType
        'EMA' => ta.ema(x, len)
        'SMA' => ta.sma(x, len) 
        'RMA' => ta.rma(x, len) 
        'TMA' => ta.sma(ta.sma(x, len), len)
 
//-----------------------------------------------------------------------------}
//Augmented RSI
//-----------------------------------------------------------------------------{
upper = ta.highest(src, length)
lower = ta.lowest(src, length)
r = upper - lower

d = src - src[1]
diff = upper > upper[1] ? r 
  : lower < lower[1] ? -r 
  : d

num = ma(diff, length, smoType1)
den = ma(math.abs(diff), length, smoType1)
arsi = num / den * 50 + 50

signal = ma(arsi, smooth, smoType2)

//-----------------------------------------------------------------------------}
//Plots
//-----------------------------------------------------------------------------{
plot_rsi = plot(arsi, 'Ultimate RSI'
  , arsi > obValue ? obCss
  : arsi < osValue ? osCss 
  : autoCss ? chart.fg_color : arsiCss)

plot(signal, 'Signal Line', signalCss)

//Levels
plot_up = plot(obValue, color = na, editable = false)
plot_avg = plot(50, color = na, editable = false)
plot_dn = plot(osValue, color = na, editable = false)

//OB-OS
fill(plot_rsi, plot_up, arsi > obValue ? obAreaCss : na)
fill(plot_dn, plot_rsi, arsi < osValue ? osAreaCss : na)

//Gradient
fill(plot_rsi, plot_avg, obValue, 50, obAreaCss, color.new(chart.bg_color, 100))
fill(plot_avg, plot_rsi, 50, osValue, color.new(chart.bg_color, 100), osAreaCss)

hline(obValue, 'Overbought')
hline(50, 'Midline')
hline(osValue, 'Oversold')

//-----------------------------------------------------------------------------}
----
//@version=6
indicator(shorttitle="BB", title="Bollinger Bands", overlay=true, timeframe="", timeframe_gaps=true)

TT_LENGTH  = "The time period to be used in calculating the MA which creates the base for the Upper and Lower Bands."
TT_MA_TYPE = "Determines the type of Moving Average that is applied to the basis plot line."
TT_SOURCE  = "Determines what data from each bar will be used in calculations."
TT_MULT    = "The number of Standard Deviations away from the MA that the Upper and Lower Bands should be."
TT_OFFSET  = "Changing this number will move the Bollinger Bands either Forwards or Backwards relative to the current market."

length = input.int(20,       "Length", minval = 1, tooltip =  TT_LENGTH)
maType = input.string("SMA", "Basis MA Type",      options = ["SMA", "EMA", "SMMA (RMA)", "WMA", "VWMA"], tooltip = TT_MA_TYPE)
src =    input(close,        "Source", tooltip = TT_SOURCE)
mult =   input.float(2.0,    "StdDev", minval = 0.001, maxval = 50,  tooltip = TT_MULT)
offset = input.int(0,        "Offset", minval = -500,  maxval = 500, tooltip =  TT_OFFSET, display = display.none)

ma(source, length, _type) =>
    switch _type
        "SMA" => ta.sma(source, length)
        "EMA" => ta.ema(source, length)
        "SMMA (RMA)" => ta.rma(source, length)
        "WMA" => ta.wma(source, length)
        "VWMA" => ta.vwma(source, length)

basis = ma(src, length, maType)
dev = mult * ta.stdev(src, length)
upper = basis + dev
lower = basis - dev

plot(basis, "Basis", color=#2962FF, offset = offset)
p1 = plot(upper, "Upper", color=#F23645, offset = offset)
p2 = plot(lower, "Lower", color=#089981, offset = offset)
fill(p1, p2, title = "Background", color=color.rgb(33, 150, 243, 95))