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
MODEL_DIR="model_files"
if os.path.exists(MODEL_DIR):
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth') or f.endswith('.pkl')]
else:
    model_files = []

feature_dropdown = dcc.Dropdown(
    id='feature-dropdown',
    options=[{'label': f'Feature {i}', 'value': i} for i in range(36)],
    value=0,
    clearable=False
)

model_visualization_layout = html.Div([
    dbc.Row([
            html.H4("Available Models"),
            dcc.Dropdown(id="model-selector", options=model_files, placeholder="trained models"),
        ],style={"border-right": "1px solid #ccc"}),
    dbc.Row([
        html.Div([  feature_dropdown,
                html.H4("Visualized Results"),
                dcc.Graph(
                    id='visual',
                )
            ], style={"border": "1px solid #ccc", "padding": "20px", "flex": 1}),
    ])    
])
@callback(
    Output('visual', 'figure'),
    Input('feature-dropdown', 'value'),
    Input('model-selector', 'value'),
    prevent_initial_call=True
)
def update_visualization(feature_index, model_name):
    if not model_name:
        return dash.no_update
    model_name = model_name.split('_')[1].split('.')[0]
    try:
        true_data = np.load(f'model_results/{model_name}/true.npy')
        pred_data = np.load(f'model_results/{model_name}/pred.npy')
                
        t = np.arange(true_data.shape[0])  # 时间轴
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, 
            y=np.mean(true_data,axis=1)[:,feature_index],
            name='True Data',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=t, 
            y=np.mean(pred_data,axis=1)[:,feature_index],
            name='Imputed Data',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f"Comparison for Feature {feature_index}",
            xaxis_title="Time Step",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception as e:
        return go.Figure(data=[go.Scatter(x=[], y=[], mode='lines')])