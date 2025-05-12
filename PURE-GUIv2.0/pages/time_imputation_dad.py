import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.exceptions import PreventUpdate
import plotly.express as px
import base64
import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output, State, callback, no_update
import os
import plotly.graph_objects as go

UPLOAD_DIR="uploaded_files"
if os.path.exists(UPLOAD_DIR):
    uploaded_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.csv')]
else:
    uploaded_files = []
try:
    with open('pages/logs.txt', 'r', encoding='utf-8') as file:
        text_content = file.read()
except FileNotFoundError:
    text_content = "未找到 text.txt 文件，请检查文件路径。"
models = ['MTSIR3-GAN','SSGAN','TimesNet']

time_imputation_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4("Available Datasets"),
            dcc.Dropdown(id="uploaded_dataset", options=uploaded_files, placeholder="dataset"),
            html.Div(id="dataset-info"),
            html.H4("Model Types"),
            dcc.Dropdown(id="models", options=models, placeholder="model",),
            # dcc.Dropdown(id="model-selector", options=model_files, placeholder="trained models"),
        ], width=4,style={"border-right": "1px solid #ccc"}),
        dbc.Col([
            html.H4("Hyper Parameters"),
            html.Div([
                html.Label("BatchSize",style={"margin-right":"10px"}),
                dcc.Slider(
                    id='batchsize',
                        min=0,
                        max=4,
                        step=1,
                        marks={
                            0: '16',
                            1: '32',
                            2: '64',
                            3: '128',
                            4: '256'
                        },
                        value=0
                    ),
            ]),
            
            html.Div([
                html.Label("LearningRate",style={"margin-right":"10px"}),
                dcc.Slider(
        id='learning-rate',
        min=0,
        max=4,
        step=1,
        marks={
            0: '0.0001',
            1: '0.0002',
            2: '0.0005',
            3: '0.001',
            4: '0.002'
        },
        value=0
                    ),
            ]),
        ], width=4, style={"border-right": "1px solid #ccc"}),
        dbc.Col([
            html.H4("Model Parameters"),
            html.Div([
                html.Label("WidthPerStage",style={"margin-right":"10px"}),
                dcc.Slider(
                    id='width-per-stage',
                        min=0,
                        max=4,
                        step=1,
                        marks={
                            0: '128',
                            1: '256',
                            2: '512',
                            3: '768',
                            4:'1024'
                        },
                        value=0
                    ),
            ]),
            
            html.Div([
                html.Label("CardinalityPerStage",style={"margin-right":"10px"}),
                dcc.Slider(
        id='cardinality-per-stage',
        min=0,
        max=4,
        step=1,
        marks={
            0: '4',
            1: '8',
            2: '16',
            3: '32',
            4: '64'
        },
        value=0
                    ),
            ]),
        ], width=4, style={"border-right": "1px solid #ccc"}),
        
    ]),
    dbc.Row([ 
        dbc.Col([
            html.Button(
                'Train',
                id='train-button',
                n_clicks=0,
                style={
                    "padding": "7px 50px",
                    "fontSize": 16,
                    "margin-right": "20px",
                    "backgroundColor": "#007BFF",  # 默认为蓝色
                    "color": "white",
                    "borderRadius": "10px",
                    "border": "none",
                    "cursor": "pointer",
                    "margin": "0 auto"
                }
            ),
        ], width=2,),
        dbc.Col([
            html.Div([
                html.Label("Training Progress"),
                dbc.Progress(id='progress-bar',
                striped=True,
                animated=False,
                style={"height": "15px"},
                ),
            ]),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Training Details"),
            dcc.Textarea(
            id='model-structure-textarea',
            value="No Results",
            style={
                "width": "100%",
                "height": "450px",
                "overflowY": "scroll", 
                "border": "1px solid #ccc",
                "padding": "10px",
            }
        ),
        ],  )
    ]),    
    dcc.Interval(id='progress-interval',interval=100,n_intervals=0),
    dcc.Store(id='global-state'),
    dcc.Store(id='training-complete')  
])

@callback(
    [Output('train-button', 'disabled'),
     Output('train-button', 'style')],
    [Input('uploaded_dataset', 'value'), 
     Input('models', 'value')]
)
def update_train_button(dataset_value, model_value):
    if dataset_value and model_value:
        return False, {
            "padding": "7px 50px",
            "fontSize": 16,
            "margin-right": "20px",
            "backgroundColor": "#007BFF",  # 蓝色
            "color": "white",
            "borderRadius": "10px",
            "border": "none",
            "cursor": "pointer",
            "margin": "0 auto"
        }
    else:
        return True, {
            "padding": "7px 50px",
            "fontSize": 16,
            "margin-right": "20px",
            "backgroundColor": "#cccccc",  # 灰色
            "color": "gray",
            "borderRadius": "10px",
            "border": "none",
            "cursor": "not-allowed",
            "margin": "0 auto"
        }

@callback(
    [Output('progress-interval', 'n_intervals'),
     Output('progress-interval', 'interval'),
     Output('model-structure-textarea', 'value'),
     Output('global-state', 'data'),
     Output('progress-bar', 'value'),
     Output('progress-bar', 'label'),
     Output('training-complete', 'data')],
    [Input('train-button', 'n_clicks'),
     Input('progress-interval', 'n_intervals'),],
    State('models', 'value'),
    prevent_initial_call=True
)
def combined_callback(train_clicks, interval_n_intervals, model_name):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if train_clicks < 1:
        raise dash.exceptions.PreventUpdate
    
    if trigger_id == 'train-button':
        # 初始化状态
        return (
            0,                # 重置计数器
            500,              # 初始间隔 0.5 秒（中间阶段）
            "",               # 清空文本框
            model_name,       # 存储模型名称
            0,                # 初始进度 0%
            f"{model_name} Training - 0%",  # 初始标签
            False             # 未完成
        )
    elif trigger_id == 'progress-interval':
        try:
            with open(f'pages/{model_name}_logs.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except FileNotFoundError:
            lines = ["未找到 logs.txt 文件，请检查文件路径。\n"] * 100  # 填充默认日志
        
        total_lines = len(lines)
        max_intervals = 100  # 总更新次数（100%）
        
        # 获取当前进度
        progress = min(interval_n_intervals, 100)  # 限制最大进度 100%
        
        # 动态设置更新间隔（非匀速）
        if progress < 20:
            interval = 4357  # 每 1000ms 更新 1%（初始阶段慢）
        elif progress < 90:
            interval = 5259   # 每 500ms 更新 1%（中间阶段快）
        else:
            interval = 3066  # 每 1000ms 更新 1%（收尾阶段慢）
        
        # 日志更新：每次进度+1%时显示下一行
        log_line_index = progress  # 当前进度百分比 = 已显示日志行数
        displayed_lines = lines[:log_line_index] if log_line_index <= total_lines else lines
        current_text = ''.join(displayed_lines)

        if interval_n_intervals >= max_intervals:
            return (
                dash.no_update,
                dash.no_update,
                current_text,
                model_name,
                100,
                f"{model_name} Training - 100%",
                True  
            )
        
        return (
            interval_n_intervals + 1,  # 下一次更新
            interval,                  # 设置新的更新间隔
            current_text,
            model_name,
            progress,
            f"{model_name} Training - {int(progress)}%",
            False
        )
    
    return dash.no_update

