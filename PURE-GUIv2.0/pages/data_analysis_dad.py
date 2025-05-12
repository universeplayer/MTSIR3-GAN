import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import base64
import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output, State, callback, no_update
import os
import matplotlib.pyplot as plt
import seaborn as sns
def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    encodings = ['utf-8', 'gbk', 'latin1']  # 尝试的编码列表
    if file_extension == '.csv':
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("无法使用支持的编码读取 CSV 文件。")
    elif file_extension == '.xlsx':
        return pd.read_excel(file_path)
    elif file_extension == '.txt':
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, sep='\t', encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("无法使用支持的编码读取 TXT 文件。")
    elif file_extension == '.npy':
        data = np.load(file_path)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# 数据分析页面布局
data_analysis_layout = html.Div([
    html.Div(id="dropdown-cell-placeholder"),
    dbc.Row([
        dbc.Col([
            html.H4("Upload Datasets"),
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "Click to Upload",
                    html.Div(id='upload-status')
                ]),
                multiple=True,
                style={
                    "width": "100%",
                    "height": "100px",
                    "lineHeight": "100px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                },
            ),
        ], width=4, style={"border-right": "1px solid #ccc"}),
        dbc.Col([
            html.H4("Upload Process"),
            html.Div(
                id="upload-progress-bars",
                style={
                    "max-height": "100px",
                    "overflow-y": "auto"
                }
            )
        ], width=4, style={"border-right": "1px solid #ccc"}),
        dbc.Col([
            html.H4("Dataset Selector"),
            dcc.Dropdown(id="dataset-selector", options=[], placeholder="dataset"),
            html.Div(id="dataset-info"),
            html.H4("Feature Selector"),
            dcc.Dropdown(id="feature-selector", options=[], multi=True, placeholder="feature"),
        ], width=4)
    ]),
    html.Div(id="uploaded-files-list"),
    dbc.Row([
        dbc.Col([
            html.H4("Variable Feature Statistics"),
            html.Div(id="data-feature-stats", style={"border": "1px solid #ccc", "padding": "10px"})
        ], width=12, style={"border-bottom": "1px solid #ccc", "padding-bottom": "20px"}),
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id="data-format-selector",
                    options=[
                        {"label": "Origin", "value": "Origin"},
                        {"label": "Moving Avg", "value": "Moving Avg"},
                        {"label": "Patch", "value": "Patch"}
                    ],
                    multi=True,
                    value=["Origin"],
                    placeholder="preprocess mode"
                ),
            ]),
            dcc.Graph(
                id="visualization-graph",
                figure=px.line(),
            )
        ], width=12)
    ]),
    dbc.Row([
        # dbc.Col([
        #     dcc.Graph(id="violin-graph",figure={
        #         'layout':{
        #             'width': '100%',  # 100%宽度自适应
        #             'height': 400,     # 容器高度（可根据需求调整）
        #             'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},  # 去除边距
        #             'xaxis':{'showgrid' : False,'visible':False},
        #             'yaxis':{'showgrid' : False,'visible':False},
        #         }
        #     })
        # ]),
        # dbc.Col([
        #     dcc.Graph(id="heatmap-graph",figure={
        #         'layout':{
        #             'width': '100%',  # 100%宽度自适应
        #             'height': 400,     # 容器高度（可根据需求调整）
        #             'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},  # 去除边距
        #             'xaxis':{'showgrid' : False,'visible':False},
        #             'yaxis':{'showgrid' : False,'visible':False},
        #         }
        #     })
        # ]),
        html.Div([
            dcc.Graph(id='violin-graph'),
            dcc.Graph(id='heatmap-graph')
        ],style={
        'display': 'flex',
        'flexDirection': 'row',
        'justifyContent': 'space-around',
        'alignItems': 'center'
    })
    ]),
    dcc.Store(id="stored-filenames", data=[]),
    dcc.Store(id="upload-progress-data", data=[]),
    dcc.Store(id="selected-dataset", data=None),
    dcc.Store(id="selected-features", data=[]),
    dcc.Store(id="data-feature-stats-store", data=[]),
    dcc.Store(id="feature-rows", data=[]),  # 新增存储 rows 数据的组件



    dcc.Store(id="data-analysis-store",data={})
], style={"border": "1px solid #ccc", "padding": "20px"})

# 合并后的文件上传和更新进度条回调
@callback(
    [Output("upload-progress-bars", "children"),
     Output("upload-progress-data", "data", allow_duplicate=True),
     Output("stored-filenames", "data", allow_duplicate=True),
     ],
    [Input("upload-data", "filename"),
     Input("upload-data", "contents"),
     Input("upload-data", "last_modified")],
    [State("stored-filenames", "data"),
     State("upload-progress-data", "data"),
     State("url", "pathname")],
    prevent_initial_call=True
)
def handle_upload_and_update_progress(filenames, contents_list, last_modified, stored_files, progress_data, current_path):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "upload-data":
        if current_path != "/data-analysis":
            raise PreventUpdate

        new_files = []
        new_progress = []

        if not all([filenames, contents_list]):
            return no_update, no_update, no_update

        try:
            for name, data in zip(filenames, contents_list):
                if name not in stored_files:
                    if 'base64,' in data:
                        header, content_string = data.split(';base64,')
                    else:
                        content_string = data.split(',')[-1]

                    decoded_data = base64.b64decode(content_string)
                    file_path = os.path.join(UPLOAD_DIR, name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    with open(file_path, "wb") as f:
                        f.write(decoded_data)
                    new_files.append(name)
                    new_progress.append({"filename": name, "status": "Done"})

            updated_files = stored_files + new_files
            updated_progress_data = progress_data + new_progress

        except Exception as e:
            print(f"文件处理异常：{str(e)}")
            error_msg = html.Div(f"上传失败：{str(e)}", style={"color": "red"})
            return [error_msg], no_update, no_update

    elif trigger_id == "upload-progress-data":
        # 这里原逻辑中没有使用到这个分支，可根据实际情况考虑是否保留
        updated_progress_data = progress_data
        updated_files = stored_files

    progress_bars = []
    for i, progress in enumerate(updated_progress_data[:4]):
        status = html.Span(progress["status"], style={"color": "green", "font-size": "12px", "float": "right"})
        progress_bar = dbc.Progress(
            value=100,
            striped=True,
            animated=False,
            style={"height": "15px"},
            label="100%"
        )
        progress_bars.append(
            html.Div([
                html.P([progress["filename"], status], style={"margin-bottom": "3px"}),
                progress_bar
            ], style={"margin-bottom": "5px"})
        )

    return progress_bars,  updated_progress_data, updated_files

# 数据集选择器回调
@callback(
    [Output("dataset-selector", "options"),
     Output("dataset-selector", "value", allow_duplicate=True)],
    Input("stored-filenames", "data"),
    [State("selected-dataset", "data")],
    prevent_initial_call=True
)
def update_dataset_selector(filenames, selected_dataset):
    options = [{"label": filename, "value": filename} for filename in filenames]
    return options, selected_dataset

# 特征选择器回调
@callback(
    [Output("feature-selector", "options"),
     Output("feature-selector", "value")],
    Input("dataset-selector", "value")
)
def update_feature_selector(selected_dataset):
    if selected_dataset:
        file_path = os.path.join(UPLOAD_DIR, selected_dataset)
        try:
            df = read_file(file_path)
            columns = df.columns.tolist()[1:]
            options = [{"label": col, "value": col} for col in columns]
            return options, []  # 避免依赖 selected-features.data，初始值设为空列表
        except Exception as e:
            print(f"读取文件失败：{str(e)}")
    return [], []

@callback(
    [Output("selected-dataset", "data"),
     Output("selected-features", "data")],
    [Input("dataset-selector", "value"),
     Input("feature-selector", "value")]
)
def save_selections(selected_dataset, selected_features):
    return selected_dataset, selected_features

# 数据特征统计回调
@callback(
    [Output("data-feature-stats", "children"),
     Output("feature-rows", "data"),
     Output("data-feature-stats-store", "data")],  # 新增输出
    [Input("dataset-selector", "value"),
     Input("feature-selector", "value")]
)
def calculate_data_features(selected_dataset, selected_features):
    if selected_dataset and selected_features:
        file_path = os.path.join(UPLOAD_DIR, selected_dataset)
        try:
            df = read_file(file_path)
            rows = []
            for feature in selected_features:
                series = df[feature]
                num_points = len(series)
                start_timestamp = df.iloc[0, 0]
                end_timestamp = df.iloc[-1, 0]
                min_val = series.min()
                max_val = series.max()
                mean_val = series.mean()
                variance = series.var()
                std_dev = series.std()
                missing_rate = series.isna().mean() * 100
                if df.columns[-1] == "label":
                    anomaly_rate = df["label"].mean() * 100
                else:
                    anomaly_rate = 0

                rows.append([
                    num_points,
                    start_timestamp,
                    end_timestamp,
                    f"{min_val:.2f}",
                    f"{max_val:.3f}",
                    f"{mean_val:.3f}",
                    f"{variance:.3f}",
                    f"{std_dev:.3f}",
                    f"{missing_rate:.2f}%",
                    f"{anomaly_rate:.2f}%"
                ])

            if len(selected_features) >= 1:
                dropdown_options = [{"label": f"{selected_dataset}-{feature}", "value": i} for i, feature in
                                    enumerate(selected_features)]
                dropdown_value = 0
            else:
                dropdown_options = []
                dropdown_value = None

            data_row = rows[0] if rows else []
            table = html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Dataset-Feature", style={"text-align": "center", "padding": "3px"}),
                        html.Th("Points", style={"text-align": "center", "padding": "3px"}),
                        html.Th("StartTime", style={"text-align": "center", "padding": "3px"}),
                        html.Th("EndTime", style={"text-align": "center", "padding": "3px"}),
                        html.Th("Min", style={"text-align": "center", "padding": "3px"}),
                        html.Th("Max", style={"text-align": "center", "padding": "3px"}),
                        html.Th("Mean", style={"text-align": "center", "padding": "3px"}),
                        html.Th("Var", style={"text-align": "center", "padding": "3px"}),
                        html.Th("Std", style={"text-align": "center", "padding": "3px"}),
                        html.Th("MissingRate", style={"text-align": "center", "padding": "3px"}),
                        html.Th("AnomalyRate", style={"text-align": "center", "padding": "3px"})
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            dcc.Dropdown(
                                id="feature-dropdown",
                                options=dropdown_options,
                                value=dropdown_value,
                                clearable=False,
                                style={"width": "250px"}  # 拓宽下拉框大小
                            ),
                            style={"text-align": "center"}
                        ),
                        *[html.Td(cell, style={"text-align": "center", "background-color": "#f2f2f2",
                                               "border": "1px solid #ddd"}) for cell in data_row]
                    ], id="table-row")
                ])
            ], style={"border-collapse": "collapse", "width": "100%"})
            # 将表格直接返回给 data-feature-stats
            result = html.Div([table])
            return result, rows, result  # 新增返回存储的数据
        except Exception as e:
            print(f"计算数据特征失败：{str(e)}")
    return html.P("No Results"), [], None

# 更新表格行的回调
@callback(
    Output("table-row", "children"),
    [Input("feature-dropdown", "value"),
     Input("feature-rows", "data"),
     Input("selected-dataset", "data"),
     Input("selected-features", "data")]
)
def update_table(selected_index, feature_rows, selected_dataset, selected_features):
    if selected_index is not None and feature_rows:
        dropdown_options = [{"label": f"{selected_dataset}-{feature}", "value": i} for i, feature in
                            enumerate(selected_features)]
        selected_row = feature_rows[selected_index]
        return [
            html.Td(
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=dropdown_options,
                    value=selected_index,
                    clearable=False,
                    style={"width": "200px"}  # 拓宽下拉框大小
                ),
                style={"text-align": "center"}
            ),
            *[html.Td(cell, style={"text-align": "center", "padding": "3px", "background-color": "#f2f2f2",
                                   "border": "1px solid #ddd"}) for cell in selected_row]
        ]
    return []

# 可视化回调
@callback(
    Output("visualization-graph", "figure"),
    [Input("dataset-selector", "value"),
     Input("feature-selector", "value"),
     Input("data-format-selector", "value")]
)
def visualize_data(selected_dataset, selected_features, data_formats):
    if not selected_dataset or not selected_features or not data_formats:
        return px.line()

    file_path = os.path.join(UPLOAD_DIR, selected_dataset)
    df = read_file(file_path)

    fig = px.line()

    for feature in selected_features:
        if "Origin" in data_formats:
            x_values = df[df.columns[0]]
            y_values = df[feature]
            fig.add_scatter(x=x_values, y=y_values, mode='lines', name=f"{feature} - Origin")

        if "Moving Avg" in data_formats:
            ema = df[feature].ewm(span=24).mean()
            x_values = df[df.columns[0]]
            fig.add_scatter(x=x_values, y=ema, mode='lines', name=f"{feature} - Moving Avg")

        if "Patch" in data_formats:
            patch_size = 16
            num_patches = len(df[feature]) // patch_size
            patch_means = [df[feature][i * patch_size:(i + 1) * patch_size].mean() for i in range(num_patches)]
            patch_index = [i * patch_size for i in range(num_patches)]
            x_values = df[df.columns[0]].iloc[patch_index]
            fig.add_scatter(x=x_values, y=patch_means, mode='lines', name=f"{feature} - Patch")

    # 设置图例位置在图像左侧
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0
        )
    )

    # 可以进一步设置 x 轴的显示格式
    fig.update_xaxes(title_text=df.columns[0])

    return fig
    
@callback(
    [Output("violin-graph", "figure"),
     Output("heatmap-graph", "figure")],
    [Input("dataset-selector", "value")]
)
def update_detail_graph(dataset):
    if not dataset:
        return go.Figure(),go.Figure()
    file_path = os.path.join(UPLOAD_DIR, dataset)
    df = read_file(file_path)
    df['date'] = pd.to_datetime(df['date'])
    data_columns = df.columns.drop('date')

    df_melted = pd.melt(df,id_vars=['date'],var_name='Feature',value_name='Value')
    violin_fig = px.violin(df_melted, y='Value',x='Feature',color='Feature',box=True,
                           points=False,title='Violin Graph & Heat Map',
                           color_discrete_sequence=px.colors.qualitative.Set1,
                            template='plotly_white')
    violin_fig.update_layout(showlegend=False,width=600)

    corr_matrix = df[data_columns].corr()
    heatmap_fig = go.Figure(data = go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis'
    )).update_layout(
        xaxis_nticks=len(data_columns)//2,  # X轴刻度数
        yaxis_nticks=len(data_columns)//2,  # Y轴刻度数
        width=425,                       # 宽度稍大实现正方形效果
        height=350,
        margin=dict(t=50, b=0, l=0, r=70)  # 调整边距
    )
    return violin_fig, heatmap_fig