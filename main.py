from dash import Dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import State, Input, Output
from flask_caching import Cache
import plotly.graph_objects as go
import alpaca_trade_api as tradeapi
import talib.abstract as ta

import os
import numpy as np
import pandas as pd
import random
import time
import yfinance as yf
from talib import abstract
from vectorbt.portfolio.base import Portfolio

# Import callbacks and settings from your project
from callbacks import register_callbacks
from settings import (
    APP_SETTINGS,
    APCA_API_KEY_ID,
    APCA_API_SECRET,
    BASE_URL,
    periods,
    intervals,
    patterns,
    stats_table_columns,
    directions,
    conflict_modes,
    plot_types,
    color_schema,
    bgcolor,
    dark_bgcolor,
    fontcolor,
    dark_fontcolor,
    gridcolor,
    loadcolor,
    active_color,
    data_path,
    default_metric,
    default_symbol,
    default_period,
    default_interval,
    default_date_range,
    default_fees,
    default_fixed_fees,
    default_slippage,
    default_yf_options,
    default_entry_n_random,
    default_exit_n_random,
    default_prob_options,
    default_entry_prob,
    default_exit_prob,
    default_entry_patterns,
    default_exit_options,
    default_exit_patterns,
    default_candle_settings,
    default_entry_dates,
    default_exit_dates,
    default_direction,
    default_conflict_mode,
    default_sim_options,
    default_n_random_strat,
    default_stats_options,
    default_layout,
    default_subplots,
    default_plot_type
)


app = Dash(__name__, external_stylesheets=[dbc.themes.GRID])

# Layout setup
app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            children=[
                html.H6("vectorbt: candlestick patterns"),
                html.Div(
                ),
            ],
        ),
        dbc.Row(
            children=[
                dbc.Col(
                    lg=8, sm=12,
                    children=[
                        html.Div(
                            className="pretty-container",
                            children=[
                                html.Div(
                                    className="banner",
                                    children=[
                                        html.H6("OHLCV and signals")
                                    ],
                                ),
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            lg=4, sm=12,
                                            children=[
                                                html.Label("Select plot type:"),
                                                dcc.Dropdown(
                                                    id="plot_type_dropdown",
                                                    options=[{"value": i, "label": i} for i in plot_types],
                                                    value=default_plot_type,
                                                ),
                                            ]
                                        )
                                    ],
                                ),
                                dcc.Loading(
                                    id="ohlcv_loading",
                                    type="default",
                                    color=loadcolor,
                                    children=[
                                        dcc.Graph(
                                            id="ohlcv_graph",
                                            figure={
                                                "layout": default_layout
                                            }
                                        )
                                    ],
                                ),
                                html.Small("Hint: Use Box and Lasso Select to filter signals"),
                            ],
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        html.Div(
                                            className="pretty-container",
                                            children=[
                                                html.Div(
                                                    className="banner",
                                                    children=[
                                                        html.H6("Portfolio")
                                                    ],
                                                ),
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            lg=6, sm=12,
                                                            children=[
                                                                html.Label("Select subplots:"),
                                                                dcc.Dropdown(
                                                                    id="subplot_dropdown",
                                                                    options=[
                                                                        {"value": k, "label": v['title']}
                                                                        for k, v in Portfolio.subplots.items()
                                                                    ],
                                                                    multi=True,
                                                                    value=default_subplots,
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                dcc.Loading(
                                                    id="portfolio_loading",
                                                    type="default",
                                                    color=loadcolor,
                                                    children=[
                                                        dcc.Graph(
                                                            id="portfolio_graph",
                                                            figure={
                                                                "layout": default_layout
                                                            }
                                                        )
                                                    ],
                                                ),
                                            ],
                                        ),

                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        html.Div(
                                            className="pretty-container",
                                            children=[
                                                html.Div(
                                                    className="banner",
                                                    children=[
                                                        html.H6("Stats")
                                                    ],
                                                ),
                                                dcc.Loading(
                                                    id="stats_loading",
                                                    type="default",
                                                    color=loadcolor,
                                                    children=[
                                                        dash_table.DataTable(
                                                            id="stats_table",
                                                            columns=[
                                                                {
                                                                    "name": c,
                                                                    "id": c,
                                                                }
                                                                for c in stats_table_columns
                                                            ],
                                                            style_data_conditional=[{
                                                                "if": {"column_id": stats_table_columns[1]},
                                                                "fontWeight": "bold",
                                                                "borderLeft": "1px solid dimgrey"
                                                            }, {
                                                                "if": {"column_id": stats_table_columns[2]},
                                                                "fontWeight": "bold",
                                                            }, {
                                                                "if": {"column_id": stats_table_columns[3]},
                                                                "fontWeight": "bold",
                                                            }, {
                                                                "if": {"column_id": stats_table_columns[4]},
                                                                "fontWeight": "bold",
                                                            }, {
                                                                "if": {"state": "selected"},
                                                                "backgroundColor": dark_bgcolor,
                                                                "color": active_color,
                                                                "border": "1px solid " + active_color,
                                                            }, {
                                                                "if": {"state": "active"},
                                                                "backgroundColor": dark_bgcolor,
                                                                "color": active_color,
                                                                "border": "1px solid " + active_color,
                                                            }],
                                                            style_header={
                                                                "border": "none",
                                                                "backgroundColor": bgcolor,
                                                                "fontWeight": "bold",
                                                                "padding": "0px 5px"
                                                            },
                                                            style_data={
                                                                "border": "none",
                                                                "backgroundColor": bgcolor,
                                                                "color": dark_fontcolor,
                                                                "paddingRight": "10px"
                                                            },
                                                            style_table={
                                                                'overflowX': 'scroll',
                                                            },
                                                            style_as_list_view=False,
                                                            editable=False,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        html.Div(
                                            className="pretty-container",
                                            children=[
                                                html.Div(
                                                    className="banner",
                                                    children=[
                                                        html.H6("Metric stats")
                                                    ],
                                                ),
                                                dcc.Loading(
                                                    id="metric_stats_loading",
                                                    type="default",
                                                    color=loadcolor,
                                                    children=[
                                                        html.Label("Metric:"),
                                                        dbc.Row(
                                                            children=[
                                                                dbc.Col(
                                                                    lg=4, sm=12,
                                                                    children=[
                                                                        dcc.Dropdown(
                                                                            id="metric_dropdown"
                                                                        ),
                                                                    ]
                                                                ),
                                                                dbc.Col(
                                                                    lg=8, sm=12,
                                                                    children=[
                                                                        dcc.Graph(
                                                                            id="metric_graph",
                                                                            figure={
                                                                                "layout": default_layout
                                                                            }
                                                                        )
                                                                    ]
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ]
                                )
                            ]
                        ),

                    ]
                ),
                dbc.Col(
                    lg=4, sm=12,
                    children=[
                        html.Div(
                            className="pretty-container",
                            children=[
                                html.Div(
                                    className="banner",
                                    children=[
                                        html.H6("Settings")
                                    ],
                                ),
                                html.Button(
                                    "Reset",
                                    id="reset_button"
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Data",
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Yahoo! Finance symbol:"),
                                                        dcc.Input(
                                                            id="symbol_input",
                                                            className="input-control",
                                                            type="text",
                                                            value=default_symbol,
                                                            placeholder="Enter symbol...",
                                                            debounce=True
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col()
                                            ],
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Period:"),
                                                        dcc.Dropdown(
                                                            id="period_dropdown",
                                                            options=[{"value": i, "label": i} for i in periods],
                                                            value=default_period,
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Interval:"),
                                                        dcc.Dropdown(
                                                            id="interval_dropdown",
                                                            options=[{"value": i, "label": i} for i in intervals],
                                                            value=default_interval,
                                                        ),
                                                    ]
                                                )
                                            ],
                                        ),
                                        html.Label("Filter period:"),
                                        dcc.RangeSlider(
                                            id="date_slider",
                                            min=0.,
                                            max=1.,
                                            value=default_date_range,
                                            allowCross=False,
                                            tooltip={
                                                'placement': 'bottom'
                                            }
                                        ),
                                        dcc.Checklist(
                                            id="yf_checklist",
                                            options=[{
                                                "label": "Adjust all OHLC automatically",
                                                "value": "auto_adjust"
                                            }, {
                                                "label": "Use back-adjusted data to mimic true historical prices",
                                                "value": "back_adjust"
                                            }],
                                            value=default_yf_options,
                                            style={
                                                "color": dark_fontcolor
                                            }
                                        ),
                                    ],
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Entry patterns",
                                        ),
                                        html.Div(
                                            id='entry_settings',
                                            children=[
                                                html.Button(
                                                    "All",
                                                    id="entry_all_button"
                                                ),
                                                html.Button(
                                                    "Random",
                                                    id="entry_random_button"
                                                ),
                                                html.Button(
                                                    "Clear",
                                                    id="entry_clear_button"
                                                ),
                                                html.Label("Number of random patterns:"),
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            children=[
                                                                dcc.Input(
                                                                    id="entry_n_random_input",
                                                                    className="input-control",
                                                                    value=default_entry_n_random,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=1, max=len(patterns), step=1
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col()
                                                    ],
                                                ),
                                                html.Label("Select patterns:"),
                                                dcc.Dropdown(
                                                    id="entry_pattern_dropdown",
                                                    options=[{"value": i, "label": i} for i in patterns],
                                                    multi=True,
                                                    value=default_entry_patterns,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Exit patterns",
                                        ),
                                        dcc.Checklist(
                                            id="exit_checklist",
                                            options=[{
                                                "label": "Same as entry patterns",
                                                "value": "same_as_entry"
                                            }],
                                            value=default_exit_options,
                                            style={
                                                "color": dark_fontcolor
                                            }
                                        ),
                                        html.Div(
                                            id='exit_settings',
                                            hidden="same_as_entry" in default_exit_options,
                                            children=[
                                                html.Button(
                                                    "All",
                                                    id="exit_all_button"
                                                ),
                                                html.Button(
                                                    "Random",
                                                    id="exit_random_button"
                                                ),
                                                html.Button(
                                                    "Clear",
                                                    id="exit_clear_button"
                                                ),
                                                html.Label("Number of random patterns:"),
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            children=[
                                                                dcc.Input(
                                                                    id="exit_n_random_input",
                                                                    className="input-control",
                                                                    value=default_exit_n_random,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=1, max=len(patterns), step=1
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col()
                                                    ],
                                                ),
                                                html.Label("Select patterns:"),
                                                dcc.Dropdown(
                                                    id="exit_pattern_dropdown",
                                                    options=[{"value": i, "label": i} for i in patterns],
                                                    multi=True,
                                                    value=default_exit_patterns,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Details(
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Candle settings",
                                        ),
                                        dash_table.DataTable(
                                            id="candle_settings_table",
                                            columns=[
                                                {
                                                    "name": c,
                                                    "id": c,
                                                    "editable": i in (2, 3),
                                                    "type": "numeric" if i in (2, 3) else "any"
                                                }
                                                for i, c in enumerate(default_candle_settings.columns)
                                            ],
                                            data=default_candle_settings.to_dict("records"),
                                            style_data_conditional=[{
                                                "if": {"column_editable": True},
                                                "backgroundColor": dark_bgcolor,
                                                "border": "1px solid dimgrey"
                                            }, {
                                                "if": {"state": "selected"},
                                                "backgroundColor": dark_bgcolor,
                                                "color": active_color,
                                                "border": "1px solid " + active_color,
                                            }, {
                                                "if": {"state": "active"},
                                                "backgroundColor": dark_bgcolor,
                                                "color": active_color,
                                                "border": "1px solid " + active_color,
                                            }],
                                            style_header={
                                                "border": "none",
                                                "backgroundColor": bgcolor,
                                                "fontWeight": "bold",
                                                "padding": "0px 5px"
                                            },
                                            style_data={
                                                "border": "none",
                                                "backgroundColor": bgcolor,
                                                "color": dark_fontcolor
                                            },
                                            style_table={
                                                'overflowX': 'scroll',
                                            },
                                            style_as_list_view=False,
                                            editable=True,
                                        ),
                                    ],
                                ),
                                html.Details(
                                    open=False,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Custom pattern",
                                        ),
                                        html.Div(
                                            id='custom_settings',
                                            children=[
                                                html.Label("Select entry dates:"),
                                                dcc.Dropdown(
                                                    id="custom_entry_dropdown",
                                                    options=[],
                                                    multi=True,
                                                    value=default_entry_dates,
                                                ),
                                                html.Label("Select exit dates:"),
                                                dcc.Dropdown(
                                                    id="custom_exit_dropdown",
                                                    options=[],
                                                    multi=True,
                                                    value=default_exit_dates,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Simulation settings",
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Fees (in %):"),
                                                        dcc.Input(
                                                            id="fees_input",
                                                            className="input-control",
                                                            type="number",
                                                            value=default_fees,
                                                            placeholder="Enter fees...",
                                                            debounce=True,
                                                            min=0, max=100
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Fixed fees:"),
                                                        dcc.Input(
                                                            id="fixed_fees_input",
                                                            className="input-control",
                                                            type="number",
                                                            value=default_fixed_fees,
                                                            placeholder="Enter fixed fees...",
                                                            debounce=True,
                                                            min=0
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Slippage (in % of H-O):"),
                                                        dcc.Input(
                                                            id="slippage_input",
                                                            className="input-control",
                                                            type="number",
                                                            value=default_slippage,
                                                            placeholder="Enter slippage...",
                                                            debounce=True,
                                                            min=0, max=100
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col()
                                            ],
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Direction:"),
                                                        dcc.Dropdown(
                                                            id="direction_dropdown",
                                                            options=[{"value": i, "label": i} for i in directions],
                                                            value=default_direction,
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Conflict Mode:"),
                                                        dcc.Dropdown(
                                                            id="conflict_mode_dropdown",
                                                            options=[{"value": i, "label": i} for i in conflict_modes],
                                                            value=default_conflict_mode
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                        dcc.Checklist(
                                            id="sim_checklist",
                                            options=[{
                                                "label": "Allow signal accumulation",
                                                "value": "allow_accumulate"
                                            }],
                                            value=default_sim_options,
                                            style={
                                                "color": dark_fontcolor
                                            }
                                        ),
                                        html.Label("Number of random strategies to test against:"),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        dcc.Input(
                                                            id="n_random_strat_input",
                                                            className="input-control",
                                                            value=default_n_random_strat,
                                                            placeholder="Enter number...",
                                                            debounce=True,
                                                            type="number",
                                                            min=10, max=1000, step=1
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col()
                                            ],
                                        ),
                                        dcc.Checklist(
                                            id="prob_checklist",
                                            options=[{
                                                "label": "Mimic strategy by shuffling",
                                                "value": "mimic_strategy"
                                            }],
                                            value=default_prob_options,
                                            style={
                                                "color": dark_fontcolor
                                            }
                                        ),
                                        html.Div(
                                            id='prob_settings',
                                            hidden="mimic_strategy" in default_prob_options,
                                            children=[
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            children=[
                                                                html.Label("Entry probability (in %):"),
                                                                dcc.Input(
                                                                    id="entry_prob_input",
                                                                    className="input-control",
                                                                    value=default_entry_prob,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=0, max=100
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col(
                                                            children=[
                                                                html.Label("Exit probability (in %):"),
                                                                dcc.Input(
                                                                    id="exit_prob_input",
                                                                    className="input-control",
                                                                    value=default_exit_prob,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=0, max=100
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        dcc.Checklist(
                                            id="stats_checklist",
                                            options=[{
                                                "label": "Include open trades in stats",
                                                "value": "incl_open"
                                            }, {
                                                "label": "Use positions instead of trades in stats",
                                                "value": "use_positions"
                                            }],
                                            value=default_stats_options,
                                            style={
                                                "color": dark_fontcolor
                                            }
                                        ),
                                    ],
                                ),
                                    html.Div(
                                        id='execute_trade',
                                        children=[
                                        html.H1("Trading Dashboard"),
                                        dcc.Input(id='symbol', type='text', value=default_symbol, className="input-control", placeholder='Enter Symbol...'),
                                        dcc.Dropdown(
                                                            id="period",
                                                            options=[{"value": i, "label": i} for i in periods],
                                                            value=default_period,
                                                        ),
                                        dcc.Dropdown(
                                                            id="interval",
                                                            options=[{"value": i, "label": i} for i in intervals],
                                                            value=default_interval,
                                                        ),
                                        html.Button('Update', id='update_button', n_clicks=0),
                                        html.Div(id='trade_status'),
                                        html.Div(id='account_info'),
                                        html.Div(id='positions_info'),
                                        dcc.Interval(
                                        id='interval_component',
                                        interval=60*1000,  # Update every minute
                                        n_intervals=0
                                        )
                                    ]
                                ),
                            ],
                        ),
                    ]
                )
            ]
        ),
        html.Div(id='data_signal', style={'display': 'none'}),
        html.Div(id='index_signal', style={'display': 'none'}),
        html.Div(id='candle_settings_signal', style={'display': 'none'}),
        html.Div(id='stats_signal', style={'display': 'none'}),
        html.Div(id="window_width", style={'display': 'none'}),
        dcc.Location(id="url")
    ],
)



# Register callbacks
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(host=APP_SETTINGS["HOST"], port=APP_SETTINGS["PORT"], debug=APP_SETTINGS["DEBUG"])
