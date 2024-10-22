from dash.dependencies import Input, Output, State
import talib.abstract as ta
import yfinance as yf
from utils import fetch_data
import pandas as pd
import dash
import talib
import json
from vectorbt.portfolio.base import Portfolio
from vectorbt.utils.colors import adjust_opacity
import random
from flask_caching import Cache
from talib._ta_lib import (
    CandleSettingType,
    RangeType,
    _ta_set_candle_settings
)
import dash_html_components as html
from vectorbt.utils.config import merge_dicts
import numpy as np
import plotly.graph_objects as go
from settings import *
USE_CACHING = APP_SETTINGS["USE_CACHING"]
import alpaca_trade_api as tradeapi
import yfinance as yf
import talib.abstract as ta
import numpy as np
import pandas as pd
import time
from settings import (APCA_API_KEY_ID, APCA_API_SECRET, BASE_URL)
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET, BASE_URL, api_version='v2')
last_trade_time = None
cooldown_period = 0
def execute_trading_strategy(symbol, strategy):
    global last_trade_time
    for index, signal in strategy.iterrows():
        if signal['combined_buy_signal']:
            if last_trade_time is None or (time.time() - last_trade_time) > cooldown_period:
                place_order(symbol, qty=1, side='buy')
                last_trade_time = time.time()
        elif signal['combined_sell_signal']:
            if last_trade_time is None or (time.time() - last_trade_time) > cooldown_period:
                place_order(symbol, qty=1, side='sell')
                last_trade_time = time.time()

def update_trades(n_clicks, symbol, period, interval):
    """Callback to update trades based on the given symbol and strategy."""
    if n_clicks > 0:
        data = yf.download(symbol, period=period, interval=interval)
        data['MACD'], data['MACD_Signal'], _ = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['buy_signal_macd'] = data['MACD'] > data['MACD_Signal']
        data['sell_signal_macd'] = data['MACD'] < data['MACD_Signal']
        data['EMA_short'] = ta.EMA(data['Close'], timeperiod=12)
        data['EMA_long'] = ta.EMA(data['Close'], timeperiod=26)
        data['buy_signal_ema'] = data['EMA_short'] > data['EMA_long']
        data['sell_signal_ema'] = data['EMA_short'] < data['EMA_long']
        data['combined_buy_signal'] = data['buy_signal_macd'] & data['buy_signal_ema']
        data['combined_sell_signal'] = data['sell_signal_macd'] & data['sell_signal_ema']
        execute_trading_strategy(symbol, data)
        return f"Trades executed based on combined strategy for {symbol}"
    return "Waiting for update."

def get_account(n_intervals):
    """Update account info using Alpaca API."""
    account = api.get_account()
    return f"Account Equity: {account.equity}, Cash: {account.cash}"

def get_positions(n_intervals):
    """Update positions info using Alpaca API."""
    positions = api.list_positions()
    return [f"{position.symbol}: {position.qty} shares @ {position.avg_entry_price}" for position in positions]

def place_order(symbol, qty, side, type='market', time_in_force='gtc'):
    """Place order using Alpaca API."""
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side, type=type, time_in_force=time_in_force)
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        return None

def register_callbacks(app):
    CACHE_CONFIG = {
        'CACHE_TYPE': 'filesystem' if USE_CACHING else 'null',
        'CACHE_DIR': 'data',
        'CACHE_DEFAULT_TIMEOUT': 0,
        'CACHE_THRESHOLD': 50
    }
    app.clientside_callback(
        """
        function(href) {
            return window.innerWidth;
        }
        """,
        Output("window_width", "children"),
        [Input("url", "href")],
    )
    cache = Cache()
    cache.init_app(app.server, config=CACHE_CONFIG)
    @cache.memoize()
    def fetch_data(symbol, period, interval, auto_adjust, back_adjust):
        """Fetch OHLCV data from Yahoo! Finance."""
        return yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            actions=False,
            auto_adjust=auto_adjust,
            back_adjust=back_adjust
        )
    # Callback to update trades
    app.callback(
        Output('trade_status', 'children'),
        [Input('update_button', 'n_clicks')],
        [State('symbol', 'value'), State('period', 'value'), State('interval', 'value')]
    )(update_trades)

    # Callback to update account info
    app.callback(
        Output('account_info', 'children'),
        [Input('interval_component', 'n_intervals')]
    )(get_account)

    # Callback to update positions info
    app.callback(
        Output('positions_info', 'children'),
        [Input('interval_component', 'n_intervals')]
    )(get_positions)
    
    # Callback to fetch and store data in hidden divs
    @app.callback(
        [Output('data_signal', 'children'),
         Output('index_signal', 'children')],
        [Input('symbol_input', 'value'),
         Input('period_dropdown', 'value'),
         Input('interval_dropdown', 'value'),
         Input("yf_checklist", "value")]
    )
    def update_data(symbol, period, interval, yf_options):
        auto_adjust = 'auto_adjust' in yf_options
        back_adjust = 'back_adjust' in yf_options
        df = fetch_data(symbol, period, interval, auto_adjust, back_adjust)
        return df.to_json(date_format='iso', orient='split'), df.index.tolist()

    # Callback to update date slider range based on available dates
    @app.callback(
        [Output('date_slider', 'min'),
         Output('date_slider', 'max'),
         Output('date_slider', 'value')],
        [Input('index_signal', 'children')]
    )
    def update_date_slider(date_list):
        return 0, len(date_list) - 1, [0, len(date_list) - 1]
    @app.callback(
        [Output('custom_entry_dropdown', 'options'),
        Output('custom_exit_dropdown', 'options')],
        [Input('index_signal', 'children'),
        Input('date_slider', 'value')]
    )
    def update_custom_options(date_list, date_range):
        """Once dates have changed, update entry/exit dates in custom pattern section.

        If selected dates cannot be found in new dates, they will be automatically removed."""
        filtered_dates = np.asarray(date_list)[date_range[0]:date_range[1] + 1].tolist()
        custom_options = [{"value": i, "label": i} for i in filtered_dates]
        return custom_options, custom_options
    # Callback to select entry patterns (all, random, clear, reset)
    @app.callback(
        Output('entry_pattern_dropdown', 'value'),
        [Input("entry_all_button", "n_clicks"),
         Input("entry_random_button", "n_clicks"),
         Input("entry_clear_button", "n_clicks"),
         Input("reset_button", "n_clicks")],
        [State("entry_n_random_input", "value")]
    )
    def select_entry_patterns(_1, _2, _3, _4, n_random):
        ctx = dash.callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'entry_all_button':
                return patterns
            elif button_id == 'entry_random_button':
                return random.sample(patterns, n_random)
            elif button_id == 'entry_clear_button':
                return []
            elif button_id == 'reset_button':
                return default_entry_patterns
        return dash.no_update

    # Callback to select exit patterns (same as entry or other options)
    @app.callback(
        [Output("exit_settings", "hidden"),
         Output("exit_n_random_input", "value"),
         Output('exit_pattern_dropdown', 'value')],
        [Input("exit_checklist", "value"),
         Input("exit_all_button", "n_clicks"),
         Input("exit_random_button", "n_clicks"),
         Input("exit_clear_button", "n_clicks"),
         Input("reset_button", "n_clicks"),
         Input("entry_n_random_input", "value"),
         Input('entry_pattern_dropdown', 'value')],
        [State("exit_n_random_input", "value")]
    )
    def select_exit_patterns(exit_options, _1, _2, _3, _4, entry_n_random, entry_patterns, exit_n_random):
        ctx = dash.callback_context
        same_as_entry = 'same_as_entry' in exit_options
        if ctx.triggered:
            control_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if control_id == 'exit_checklist':
                return same_as_entry, entry_n_random, entry_patterns
            elif control_id == 'exit_all_button':
                return same_as_entry, exit_n_random, patterns
            elif control_id == 'exit_random_button':
                return same_as_entry, exit_n_random, random.sample(patterns, exit_n_random)
            elif control_id == 'exit_clear_button':
                return same_as_entry, exit_n_random, []
            elif control_id == 'reset_button':
                default_same_as_entry = 'same_as_entry' in default_exit_options
                return default_same_as_entry, default_exit_n_random, default_exit_patterns
            elif control_id in ('entry_n_random_input', 'entry_pattern_dropdown'):
                if same_as_entry:
                    return same_as_entry, entry_n_random, entry_patterns
        return dash.no_update
    @app.callback(
        Output('candle_settings_signal', 'children'),
        [Input('candle_settings_table', 'data')]
    )
    def set_candle_settings(data):
        """Update candle settings in TA-Lib."""
        for d in data:
            AvgPeriod = d["AvgPeriod"]
            if isinstance(AvgPeriod, float) and float.is_integer(AvgPeriod):
                AvgPeriod = int(AvgPeriod)
            Factor = float(d["Factor"])
            _ta_set_candle_settings(
                getattr(CandleSettingType, d["SettingType"]),
                getattr(RangeType, d["RangeType"]),
                AvgPeriod,
                Factor
            )
    # Callback to update the OHLCV chart based on selected patterns
    @app.callback(
        [Output('ohlcv_graph', 'figure'),
        Output("prob_settings", "hidden"),
        Output("entry_prob_input", "value"),
        Output("exit_prob_input", "value")],
        [Input('window_width', 'children'),
        Input('plot_type_dropdown', 'value'),
        Input('data_signal', 'children'),
        Input('date_slider', 'value'),
        Input('entry_pattern_dropdown', 'value'),
        Input('exit_pattern_dropdown', 'value'),
        Input('candle_settings_signal', 'children'),
        Input('custom_entry_dropdown', 'value'),
        Input('custom_exit_dropdown', 'value'),
        Input('prob_checklist', 'value'),
        Input("reset_button", "n_clicks")],
        [State("entry_prob_input", "value"),
        State("exit_prob_input", "value")]
    )
    def update_ohlcv(window_width, plot_type, df_json, date_range, entry_patterns, exit_patterns, _1,
                 entry_dates, exit_dates, prob_options, _2, entry_prob, exit_prob):
        """Update OHLCV graph.

        Also update probability settings, as they also depend upon conversion of patterns into signals."""
        df = pd.read_json(df_json, orient='split')

        # Filter by date
        df = df.iloc[date_range[0]:date_range[1] + 1]

        # Run pattern recognition indicators and combine results
        talib_inputs = {
            'open': df['Open'].values,
            'high': df['High'].values,
            'low': df['Low'].values,
            'close': df['Close'].values,
            'volume': df['Volume'].values
        }
        entry_patterns += ['CUSTOM']
        exit_patterns += ['CUSTOM']
        all_patterns = list(set(entry_patterns + exit_patterns))
        signal_df = pd.DataFrame.vbt.empty(
            (len(df.index), len(all_patterns)),
            fill_value=0.,
            index=df.index,
            columns=all_patterns
        )
        for pattern in all_patterns:
            if pattern != 'CUSTOM':
                signal_df[pattern] = ta.Function(pattern)(talib_inputs)
        signal_df['CUSTOM'].loc[entry_dates] += 100.
        signal_df['CUSTOM'].loc[exit_dates] += -100.
        entry_signal_df = signal_df[entry_patterns]
        exit_signal_df = signal_df[exit_patterns]

        # Entry patterns
        entry_df = entry_signal_df[(entry_signal_df > 0).any(axis=1)]
        entry_patterns = []
        for row_i, row in entry_df.iterrows():
            entry_patterns.append('<br>'.join(row.index[row != 0]))
        entry_patterns = np.asarray(entry_patterns)

        # Exit patterns
        exit_df = exit_signal_df[(exit_signal_df < 0).any(axis=1)]
        exit_patterns = []
        for row_i, row in exit_df.iterrows():
            exit_patterns.append('<br>'.join(row.index[row != 0]))
        exit_patterns = np.asarray(exit_patterns)

        # Prepare scatter data
        highest_high = df['High'].max()
        lowest_low = df['Low'].min()
        distance = (highest_high - lowest_low) / 5
        entry_y = df.loc[entry_df.index, 'Low'] - distance
        entry_y.index = pd.to_datetime(entry_y.index)
        exit_y = df.loc[exit_df.index, 'High'] + distance
        exit_y.index = pd.to_datetime(exit_y.index)

        # Prepare signals
        entry_signals = pd.Series.vbt.empty_like(entry_y, True)
        exit_signals = pd.Series.vbt.empty_like(exit_y, True)

        # Build graph
        height = int(9 / 21 * 2 / 3 * window_width)
        fig = df.vbt.ohlcv.plot(
            plot_type=plot_type,
            **merge_dicts(
                default_layout,
                dict(
                    width=None,
                    height=max(500, height),
                    margin=dict(r=40),
                    hovermode="closest",
                    xaxis2=dict(
                        title='Date'
                    ),
                    yaxis2=dict(
                        title='Volume'
                    ),
                    yaxis=dict(
                        title='Price',
                    )
                )
            )
        )
        entry_signals.vbt.signals.plot_as_entry_markers(
            y=entry_y,
            trace_kwargs=dict(
                customdata=entry_patterns[:, None],
                hovertemplate='%{x}<br>%{customdata[0]}',
                name='Bullish signal'
            ),
            add_trace_kwargs=dict(row=1, col=1),
            fig=fig
        )
        exit_signals.vbt.signals.plot_as_exit_markers(
            y=exit_y,
            trace_kwargs=dict(
                customdata=exit_patterns[:, None],
                hovertemplate='%{x}<br>%{customdata[0]}',
                name='Bearish signal'
            ),
            add_trace_kwargs=dict(row=1, col=1),
            fig=fig
        )
        fig.update_xaxes(gridcolor=gridcolor)
        fig.update_yaxes(gridcolor=gridcolor, zerolinecolor=gridcolor)
        figure = dict(data=fig.data, layout=fig.layout)

        mimic_strategy = 'mimic_strategy' in prob_options
        ctx = dash.callback_context
        if ctx.triggered:
            control_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if control_id == 'reset_button':
                mimic_strategy = 'mimic_strategy' in default_prob_options
                entry_prob = default_entry_prob
                exit_prob = default_exit_prob
        if mimic_strategy:
            entry_prob = np.round(len(entry_df.index) / len(df.index) * 100, 4)
            exit_prob = np.round(len(exit_df.index) / len(df.index) * 100, 4)
        return figure, mimic_strategy, entry_prob, exit_prob
    def simulate_portfolio(df, interval, date_range, selected_data, entry_patterns, exit_patterns,
                       entry_dates, exit_dates, fees, fixed_fees, slippage, direction, conflict_mode,
                       sim_options, n_random_strat, prob_options, entry_prob, exit_prob):
        """Simulate portfolio of the main strategy, buy & hold strategy, and a bunch of random strategies."""
        # Filter by date
        df = df.iloc[date_range[0]:date_range[1] + 1]

        # Run pattern recognition indicators and combine results
        talib_inputs = {
            'open': df['Open'].values,
            'high': df['High'].values,
            'low': df['Low'].values,
            'close': df['Close'].values,
            'volume': df['Volume'].values
        }
        entry_patterns += ['CUSTOM']
        exit_patterns += ['CUSTOM']
        all_patterns = list(set(entry_patterns + exit_patterns))
        entry_i = [all_patterns.index(p) for p in entry_patterns]
        exit_i = [all_patterns.index(p) for p in exit_patterns]
        signals = np.full((len(df.index), len(all_patterns)), 0., dtype=np.float_)
        for i, pattern in enumerate(all_patterns):
            if pattern != 'CUSTOM':
                signals[:, i] = ta.Function(pattern)(talib_inputs)
        signals[np.flatnonzero(df.index.isin(entry_dates)), all_patterns.index('CUSTOM')] += 100.
        signals[np.flatnonzero(df.index.isin(exit_dates)), all_patterns.index('CUSTOM')] += -100.
        signals /= 100.  # TA-Lib functions have output in increments of 100

        # Filter signals
        if selected_data is not None:
            new_signals = np.full_like(signals, 0.)
            for point in selected_data['points']:
                if 'customdata' in point:
                    point_patterns = point['customdata'][0].split('<br>')
                    pi = df.index.get_loc(point['x'])
                    for p in point_patterns:
                        pc = all_patterns.index(p)
                        new_signals[pi, pc] = signals[pi, pc]
            signals = new_signals

        # Generate size for main
        def _generate_size(signals):
            entry_signals = signals[:, entry_i]
            exit_signals = signals[:, exit_i]
            return np.where(entry_signals > 0, entry_signals, 0).sum(axis=1) + \
                np.where(exit_signals < 0, exit_signals, 0).sum(axis=1)

        main_size = np.empty((len(df.index),), dtype=np.float_)
        main_size[0] = 0  # avoid looking into future
        main_size[1:] = _generate_size(signals)[:-1]

        # Generate size for buy & hold
        hold_size = np.full_like(main_size, 0.)
        hold_size[0] = np.inf

        # Generate size for random
        def _shuffle_along_axis(a, axis):
            idx = np.random.rand(*a.shape).argsort(axis=axis)
            return np.take_along_axis(a, idx, axis=axis)

        rand_size = np.empty((len(df.index), n_random_strat), dtype=np.float_)
        rand_size[0] = 0  # avoid looking into future
        if 'mimic_strategy' in prob_options:
            for i in range(n_random_strat):
                rand_signals = _shuffle_along_axis(signals, 0)
                rand_size[1:, i] = _generate_size(rand_signals)[:-1]
        else:
            entry_signals = pd.DataFrame.vbt.signals.generate_random(
                (rand_size.shape[0] - 1, rand_size.shape[1]), prob=entry_prob / 100).values
            exit_signals = pd.DataFrame.vbt.signals.generate_random(
                (rand_size.shape[0] - 1, rand_size.shape[1]), prob=exit_prob / 100).values
            rand_size[1:, :] = np.where(entry_signals, 1., 0.) - np.where(exit_signals, 1., 0.)

        # Simulate portfolio
        def _simulate_portfolio(size, init_cash='autoalign'):
            return Portfolio.from_signals(
                close=df['Close'],
                entries=size > 0,
                exits=size < 0,
                price=df['Open'],
                size=np.abs(size),
                direction=direction,
                upon_dir_conflict=conflict_mode,
                accumulate='allow_accumulate' in sim_options,
                init_cash=init_cash,
                fees=float(fees) / 100,
                fixed_fees=float(fixed_fees),
                slippage=(float(slippage) / 100) * (df['High'] / df['Open'] - 1),
                freq=interval
            )

        # Align initial cash across main and random strategies
        aligned_portfolio = _simulate_portfolio(np.hstack((main_size[:, None], rand_size)))
        # Fixate initial cash for indexing
        aligned_portfolio = aligned_portfolio.replace(
            init_cash=aligned_portfolio.init_cash
        )
        # Separate portfolios
        main_portfolio = aligned_portfolio.iloc[0]
        rand_portfolio = aligned_portfolio.iloc[1:]

        # Simulate buy & hold portfolio
        hold_portfolio = _simulate_portfolio(hold_size, init_cash=main_portfolio.init_cash)

        return main_portfolio, hold_portfolio, rand_portfolio


    @app.callback(
        [Output('portfolio_graph', 'figure'),
        Output('stats_table', 'data'),
        Output('stats_signal', 'children'),
        Output('metric_dropdown', 'options'),
        Output('metric_dropdown', 'value')],
        [Input('window_width', 'children'),
        Input('subplot_dropdown', 'value'),
        Input('data_signal', 'children'),
        Input('symbol_input', 'value'),
        Input('interval_dropdown', 'value'),
        Input('date_slider', 'value'),
        Input('ohlcv_graph', 'selectedData'),
        Input('entry_pattern_dropdown', 'value'),
        Input('exit_pattern_dropdown', 'value'),
        Input('candle_settings_signal', 'children'),
        Input('custom_entry_dropdown', 'value'),
        Input('custom_exit_dropdown', 'value'),
        Input('fees_input', 'value'),
        Input('fixed_fees_input', 'value'),
        Input('slippage_input', 'value'),
        Input('direction_dropdown', 'value'),
        Input('conflict_mode_dropdown', 'value'),
        Input('sim_checklist', 'value'),
        Input('n_random_strat_input', 'value'),
        Input('prob_checklist', 'value'),
        Input("entry_prob_input", "value"),
        Input("exit_prob_input", "value"),
        Input('stats_checklist', 'value'),
        Input("reset_button", "n_clicks")],
        [State('metric_dropdown', 'value')]
    )
    def update_stats(window_width, subplots, df_json, symbol, interval, date_range, selected_data,
                    entry_patterns, exit_patterns, _1, entry_dates, exit_dates, fees, fixed_fees,
                    slippage, direction, conflict_mode, sim_options, n_random_strat, prob_options,
                    entry_prob, exit_prob, stats_options, _2, curr_metric):
        """Final stage where we calculate key performance metrics and compare strategies."""
        df = pd.read_json(df_json, orient='split')

        # Simulate portfolio
        main_portfolio, hold_portfolio, rand_portfolio = simulate_portfolio(
            df, interval, date_range, selected_data, entry_patterns, exit_patterns,
            entry_dates, exit_dates, fees, fixed_fees, slippage, direction, conflict_mode,
            sim_options, n_random_strat, prob_options, entry_prob, exit_prob)

        subplot_settings = dict()
        if 'cum_returns' in subplots:
            subplot_settings['cum_returns'] = dict(
                benchmark_kwargs=dict(
                    trace_kwargs=dict(
                        line=dict(
                            color=adjust_opacity(color_schema['yellow'], 0.5)
                        ),
                        name=symbol
                    )
                )
            )
        height = int(6 / 21 * 2 / 3 * window_width)
        fig = main_portfolio.plot(
            subplots=subplots,
            subplot_settings=subplot_settings,
            **merge_dicts(
                default_layout,
                dict(
                    width=None,
                    height=len(subplots) * max(300, height) if len(subplots) > 1 else max(350, height)
                )
            )
        )
        fig.update_traces(xaxis="x" if len(subplots) == 1 else "x" + str(len(subplots)))
        fig.update_xaxes(
            gridcolor=gridcolor
        )
        fig.update_yaxes(
            gridcolor=gridcolor,
            zerolinecolor=gridcolor
        )

        def _chop_microseconds(delta):
            return delta - pd.Timedelta(microseconds=delta.microseconds, nanoseconds=delta.nanoseconds)

        def _metric_to_str(x):
            if isinstance(x, float):
                return '%.2f' % x
            if isinstance(x, pd.Timedelta):
                return str(_chop_microseconds(x))
            return str(x)

        incl_open = 'incl_open' in stats_options
        use_positions = 'use_positions' in stats_options
        main_stats = main_portfolio.stats(settings=dict(incl_open=incl_open, use_positions=use_positions))
        hold_stats = hold_portfolio.stats(settings=dict(incl_open=True, use_positions=use_positions))
        rand_stats = rand_portfolio.stats(settings=dict(incl_open=incl_open, use_positions=use_positions), agg_func=None)
        rand_stats_median = rand_stats.iloc[:, 3:].median(axis=0)
        rand_stats_mean = rand_stats.iloc[:, 3:].mean(axis=0)
        rand_stats_std = rand_stats.iloc[:, 3:].std(axis=0, ddof=0)
        stats_mean_diff = main_stats.iloc[3:] - rand_stats_mean

        def _to_float(x):
            if pd.isnull(x):
                return np.nan
            if isinstance(x, float):
                if np.allclose(x, 0):
                    return 0.
            if isinstance(x, pd.Timedelta):
                return float(x.total_seconds())
            return float(x)

        z = stats_mean_diff.apply(_to_float) / rand_stats_std.apply(_to_float)

        table_data = pd.DataFrame(columns=stats_table_columns)
        table_data.iloc[:, 0] = main_stats.index
        table_data.iloc[:, 1] = hold_stats.apply(_metric_to_str).values
        table_data.iloc[:3, 2] = table_data.iloc[:3, 1]
        table_data.iloc[3:, 2] = rand_stats_median.apply(_metric_to_str).values
        table_data.iloc[:, 3] = main_stats.apply(_metric_to_str).values
        table_data.iloc[3:, 4] = z.apply(_metric_to_str).values

        metric = curr_metric
        ctx = dash.callback_context
        if ctx.triggered:
            control_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if control_id == 'reset_button':
                metric = default_metric
        if metric is None:
            metric = default_metric
        return dict(data=fig.data, layout=fig.layout), \
            table_data.to_dict("records"), \
            json.dumps({
                'main': {m: [_to_float(main_stats[m])] for m in main_stats.index[3:]},
                'hold': {m: [_to_float(hold_stats[m])] for m in main_stats.index[3:]},
                'rand': {m: rand_stats[m].apply(_to_float).values.tolist() for m in main_stats.index[3:]}
            }), \
            [{"value": i, "label": i} for i in main_stats.index[3:]], \
            metric


    @app.callback(
        Output('metric_graph', 'figure'),
        [Input('window_width', 'children'),
        Input('stats_signal', 'children'),
        Input('metric_dropdown', 'value')]
    )
    def update_metric_stats(window_width, stats_json, metric):
        """Once a new metric has been selected, plot its distribution."""
        stats_dict = json.loads(stats_json)
        height = int(9 / 21 * 2 / 3 * 2 / 3 * window_width)
        return dict(
            data=[
                go.Box(
                    x=stats_dict['rand'][metric],
                    quartilemethod="linear",
                    jitter=0.3,
                    pointpos=1.8,
                    boxpoints='all',
                    boxmean='sd',
                    hoveron="points",
                    hovertemplate='%{x}<br>Random',
                    name='',
                    marker=dict(
                        color=color_schema['blue'],
                        opacity=0.5,
                        size=8,
                    ),
                ),
                go.Box(
                    x=stats_dict['hold'][metric],
                    quartilemethod="linear",
                    boxpoints="all",
                    jitter=0,
                    pointpos=1.8,
                    hoveron="points",
                    hovertemplate='%{x}<br>Buy & Hold',
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name='',
                    marker=dict(
                        color=color_schema['orange'],
                        size=8,
                    ),
                ),
                go.Box(
                    x=stats_dict['main'][metric],
                    quartilemethod="linear",
                    boxpoints="all",
                    jitter=0,
                    pointpos=1.8,
                    hoveron="points",
                    hovertemplate='%{x}<br>Strategy',
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name='',
                    marker=dict(
                        color=color_schema['green'],
                        size=8,
                    ),
                ),
            ],
            layout=merge_dicts(
                default_layout,
                dict(
                    height=max(350, height),
                    showlegend=False,
                    margin=dict(l=60, r=20, t=40, b=20),
                    hovermode="closest",
                    xaxis=dict(
                        gridcolor=gridcolor,
                        title=metric,
                        side='top'
                    ),
                    yaxis=dict(
                        gridcolor=gridcolor
                    ),
                )
            )
        )

    @app.callback(
        [Output('symbol_input', 'value'),
        Output('period_dropdown', 'value'),
        Output('interval_dropdown', 'value'),
        Output("yf_checklist", "value"),
        Output("entry_n_random_input", "value"),
        Output("exit_checklist", "value"),
        Output('candle_settings_table', 'data'),
        Output('custom_entry_dropdown', 'value'),
        Output('custom_exit_dropdown', 'value'),
        Output('fees_input', 'value'),
        Output('fixed_fees_input', 'value'),
        Output('slippage_input', 'value'),
        Output('conflict_mode_dropdown', 'value'),
        Output('direction_dropdown', 'value'),
        Output('sim_checklist', 'value'),
        Output('n_random_strat_input', 'value'),
        Output("prob_checklist", "value"),
        Output('stats_checklist', 'value')],
        [Input("reset_button", "n_clicks")],
        prevent_initial_call=True
    )

    def reset_settings(_):
        """Reset most settings. Other settings are reset in their callbacks."""
        return default_symbol, \
            default_period, \
            default_interval, \
            default_yf_options, \
            default_entry_n_random, \
            default_exit_options, \
            default_candle_settings.to_dict("records"), \
            default_entry_dates, \
            default_exit_dates, \
            default_fees, \
            default_fixed_fees, \
            default_slippage, \
            default_conflict_mode, \
            default_direction, \
            default_sim_options, \
            default_n_random_strat, \
            default_prob_options, \
            default_stats_options
