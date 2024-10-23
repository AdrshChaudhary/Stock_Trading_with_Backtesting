import os
import talib
import pandas as pd
from vectorbt.portfolio.enums import Direction, DirectionConflictMode
from vectorbt import settings

APCA_API_KEY_ID = "PKVQA6W5332B7EODVYF7"
APCA_API_SECRET = "lncXdYSbdfpffyP2zokRCI4l1qml9NZTZzuSSF1G"
BASE_URL = "https://paper-api.alpaca.markets"
periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo']
patterns = talib.get_function_groups()['Pattern Recognition']
stats_table_columns = ["Metric", "Buy & Hold", "Random (Median)", "Strategy", "Z-Score"]
directions = Direction._fields
conflict_modes = DirectionConflictMode._fields
plot_types = ['OHLC', 'Candlestick']

# Colors
color_schema = settings['plotting']['color_schema']
bgcolor = "#272a32"
dark_bgcolor = "#1d2026"
fontcolor = "#9fa6b7"
dark_fontcolor = "#7b7d8d"
gridcolor = "#323b56"
loadcolor = "#387c9e"
active_color = "#88ccee"

# Defaults
data_path = 'data/data.h5'
default_metric = 'Total Return [%]'
default_symbol = 'AAPL'
default_period = '1y'
default_interval = '1d'
default_date_range = [0, 1]
default_fees = 0.1
default_fixed_fees = 0.
default_slippage = 5.
default_yf_options = ['auto_adjust']
default_exit_n_random = default_entry_n_random = 5
default_prob_options = ['mimic_strategy']
default_entry_prob = 0.1
default_exit_prob = 0.1
default_entry_patterns = [
    'CDLHAMMER',
    'CDLINVERTEDHAMMER',
    'CDLPIERCING',
    'CDLMORNINGSTAR',
    'CDL3WHITESOLDIERS'
]
default_exit_options = []
default_exit_patterns = [
    'CDLHANGINGMAN',
    'CDLSHOOTINGSTAR',
    'CDLEVENINGSTAR',
    'CDL3BLACKCROWS',
    'CDLDARKCLOUDCOVER'
]
default_candle_settings = pd.DataFrame({
    'SettingType': [
        'BodyLong',
        'BodyVeryLong',
        'BodyShort',
        'BodyDoji',
        'ShadowLong',
        'ShadowVeryLong',
        'ShadowShort',
        'ShadowVeryShort',
        'Near',
        'Far',
        'Equal'
    ],
    'RangeType': [
        'RealBody',
        'RealBody',
        'RealBody',
        'HighLow',
        'RealBody',
        'RealBody',
        'Shadows',
        'HighLow',
        'HighLow',
        'HighLow',
        'HighLow'
    ],
    'AvgPeriod': [
        10,
        10,
        10,
        10,
        0,
        0,
        10,
        10,
        5,
        5,
        5
    ],
    'Factor': [
        1.0,
        3.0,
        1.0,
        0.1,
        1.0,
        2.0,
        1.0,
        0.1,
        0.2,
        0.6,
        0.05
    ]
})
default_entry_dates = []
default_exit_dates = []
default_direction = directions[0]
default_conflict_mode = conflict_modes[0]
default_sim_options = ['allow_accumulate']
default_n_random_strat = 50
default_stats_options = ['incl_open']
default_layout = dict(
    autosize=True,
    margin=dict(b=40, t=20),
    font=dict(
        color=fontcolor
    ),
    plot_bgcolor=bgcolor,
    paper_bgcolor=bgcolor,
    legend=dict(
        font=dict(size=10),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
)
default_subplots = ['orders', 'trade_pnl', 'cum_returns']
default_plot_type = 'OHLC'
# Settings Configuration
APP_SETTINGS = {
    "USE_CACHING": os.environ.get("USE_CACHING", "True") == "True",
    "HOST": os.environ.get("HOST", "0.0.0.0"),
    "PORT": int(os.environ.get("PORT", 8050)),
    "DEBUG": os.environ.get("DEBUG", "True") == "True",
}
