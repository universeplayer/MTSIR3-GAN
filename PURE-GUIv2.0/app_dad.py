import os
import dash

from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from pages.time_imputation_dad import time_imputation_layout
from pages.data_analysis_dad import data_analysis_layout
from pages.model_visualization import model_visualization_layout
# 初始化应用
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

logo_src = 'logo.png'
# 侧边导航栏
sidebar = html.Div(
    [
        # html.Img(src=logo_src, style={"width": "100%", "max-width": "100%", "height": "auto", "margin-bottom": "1rem"}),
        html.H3("Time-Series Analysis System", className="display-5", style={"font-size": "1.5rem"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Data Analysis", href="/data-analysis", active="exact"),
                dbc.NavLink("Data Imputation", href="/time-imputation", active="exact"),
                dbc.NavLink("Model Visualization", href="/model-visualization", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem",
        "background-color": "#f8f9fa",
    },
)

# 主布局
content = html.Div(
    id="page-content",
    style={"marginLeft": "18rem", "marginRight": "2rem", "padding": "2rem 1rem"}
)

app.layout = html.Div([
    dcc.Location(id="url",refresh=False),
    sidebar,
    content,
    dcc.Store(id="stored-filenames", data=[]),
    dcc.Store(id="upload-progress-data", data=[]),
    dcc.Store(id="selected-dataset", data=None),
    dcc.Store(id="selected-features", data=[]),
    html.Div(id="dropdown-cell-placeholder"),
    dcc.Store(id="feature-rows", data=[]),
    dcc.Store(id="visualization-graph-store", data=None)  # 新增存储可视化图形数据的组件
])


# 回调：页面路由
@callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    if pathname == "/data-analysis":
        return data_analysis_layout
    elif pathname == "/time-imputation":
        return time_imputation_layout
    elif pathname == "/model-visualization":
        return model_visualization_layout
    return html.P("Choose a page from the sidebar.")



if __name__ == "__main__":
    app.run(debug=True)