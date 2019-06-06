# coding: utf-8
'''
    Plotly Express Demo
'''

import json
from textwrap import dedent as d

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objs as go

ext_style = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'auto',
        'white-space': 'pre-wrap',
        'word-wrap': 'break-word',

    }
}

app = dash.Dash(__name__, external_stylesheets = ext_style)
app.config['suppress_callback_exceptions'] = True

# Convert to data frame
df = pd.read_csv("~/Documents/HM/anonymisedData/Open_Univ_Data_Final.csv")
if list(df.columns)[0] == 'Unnamed: 0':
   df = pd.read_csv("~/Documents/HM/anonymisedData/Open_Univ_Data_Final.csv", index_col=0)

# Get a list of column names, and make a dictionary
col_names = list(df.columns)
data_options = [
    {"label": col_names[i], "value": col_names[i]}
    for i in range(len(col_names))
]

colors = {
    'background': '#111111',
    'text': '#ffffff'
}

# Have to make this modular
slider_options = df['highest_education'].unique().tolist()

# General app layout
app.layout = html.Div(children = 
[
    html.Div([html.H2("DASHBOARD DEMO")], className="banner"),
    html.Div(
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    children=html.Div(id="overview-tab"),
                ),
                dcc.Tab(
                    label="Data",
                    value="data",
                    children=html.Div(id="data-tab"),
                ),
                dcc.Tab(
                    label="Analysis",
                    value="analysis",
                    children=html.Div(id="analysis-tab"),
                ),
                dcc.Tab(
                    label="Model",
                    value="model",
                    children=html.Div(id="model-tab"),
                ),
            ],
            vertical=False,
            mobile_breakpoint=480,
        )
    ),
    html.Br(),
    html.Div(id='rest-of-the-page')
])

# Tab Selection
@app.callback(Output('rest-of-the-page', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    app.config['suppress_callback_exceptions'] = True
    if tab == 'overview':
        return html.Div([
            html.Div([html.H3('Overview Tab')], className = 'banner2'),
            html.Br(),
            html.Div([
                dcc.Dropdown(
                    className='drops',
                    id='my-dropdown-1',
                    options=data_options,
                    value='sum_click',
                ),
                dcc.Dropdown(
                    className='drops',
                    id='my-dropdown-2',
                    options=data_options,
                    value='code_module',
                ),
                dcc.RadioItems(
                    id='crossfilter-xaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Log',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Br(),
            html.H4("Here is a scatterplot describing the relationship between number of clicks and final result, categorised by code module and presentation:",id='intro1'),
            dcc.Graph(
                id = 'overview-graph1',
                figure = {
                    'data':[
                        go.Scatter(
                            x = df[df['final_result'] == i]['sum_click'],
                            y = df[df['final_result'] == i]['code_module'],
                            text = df[df['final_result'] == i]['id_student'],
                            mode = 'markers',
                            opacity = 0.6,
                            marker = {
                                'size': 10,
                                'line': {'width': 0.5, 'color': 'white'}
                            },
                            name = i
                        ) for i in df.final_result.unique()
                    ],
                    'layout': go.Layout(
                        # title = 'Dynamic Data Visualization',
                        xaxis = {'type': 'log', 'title': 'Sum of Clicks', 'color': colors['text']},
                        yaxis = {'title': 'Code Module', 'color': colors['text']},
                        margin = {'l': 40, 'b': 40, 't': 10, 'r': 10},
                        legend = {'x': 0, 'y': 1},
                        hovermode = 'closest',
                        clickmode='event+select',
                        plot_bgcolor = colors['background'],
                        paper_bgcolor = colors['background'],
                        font = {
                            'color' : colors['text']
                        }
                    )
                }
            ),
            html.Br(),
            html.Div([
                dcc.Slider(
                    className='slider1',
                    id='slider-update',
                    marks={i:'{}'.format(slider_options[i]) for i in range(len(slider_options))},
                    max=4,
                    value=2,
                    step=None,
                    updatemode='drag'
                )
            ]),
            html.Div(className='row', children=[
                html.Div([
                    dcc.Markdown(d("""
                        **Click on points in the graph.**
                    """)),
                    html.Pre(id='click-data', style=styles['pre']),
                ], className='three-columns')
            ])

        ])
    elif tab == 'data':
        return html.Div([
            html.Div([html.H3('Data Analysis')], className="banner2"),
            html.Br(),
            dcc.Graph(
                id='pie-char1',
                figure = go.Figure(
                    data=[
                        go.Pie(
                            labels=df["code_module"].unique().tolist(),
                            values=df.groupby('code_module')['id_student'].count().tolist(),
                            marker={'colors': ['#EF963B', '#C93277', '#349600', '#EF533B', '#57D4F1','#cc6c8a','#d3c613']}, 
                            textinfo='label',
                            domain={"column": 0},
                            hole=0.4,
                        ),
                        go.Pie(
                            labels=df["code_module"].unique().tolist(),
                            values=df[df["age_band"]=="0-35"].groupby('code_module')['id_student'].count().tolist(),
                            marker={'colors': ['#EF963B', '#C93277', '#349600', '#EF533B', '#57D4F1','#cc6c8a','#d3c613']}, 
                            textinfo='label',
                            domain={"column": 1},
                            hole=0.4,
                        ),
                        go.Pie(
                            labels=df["code_module"].unique().tolist(),
                            values=df[df["age_band"]=="35-55"].groupby('code_module')['id_student'].count().tolist(),
                            marker={'colors': ['#EF963B', '#C93277', '#349600', '#EF533B', '#57D4F1','#cc6c8a','#d3c613']}, 
                            textinfo='label',
                            domain={"column": 2},
                            hole=0.4,
                        ),
                        go.Pie(
                            labels=df["code_module"].unique().tolist(),
                            values=df[df["age_band"]=="55<="].groupby('code_module')['id_student'].count().tolist(),
                            marker={'colors': ['#EF963B', '#C93277', '#349600', '#EF533B', '#57D4F1','#cc6c8a','#d3c613']}, 
                            textinfo='label',
                            domain={"column": 3},
                            hole=0.4,
                        )
                    ],
                    layout=go.Layout(
                        title=f"Number of Students", 
                        grid= {"rows": 1, "columns": 4},
                        legend={"x": 1, "y": 0.7},
                        plot_bgcolor=colors['background'],
                        paper_bgcolor = colors['background'],
                        font = {
                            'color' : colors['text']
                        },
                        annotations=[
                            {
                                "font": {
                                    "size": 14
                                },
                                "showarrow": False,
                                "text": "Total Number",
                                "x": 0.084,
                                "y": 0.5
                            },
                            {
                                "font": {
                                    "size": 14
                                },
                                "showarrow": False,
                                "text": "0-35 years",
                                "x": 0.372,
                                "y": 0.5
                            },
                            {
                                "font": {
                                    "size": 14
                                },
                                "showarrow": False,
                                "text": "35-55 years",
                                "x": 0.63   ,
                                "y": 0.5
                            },
                            {
                                "font": {
                                    "size": 14
                                },
                                "showarrow": False,
                                "text": "55<= years",
                                "x": 0.91,
                                "y": 0.5
                            }
                        ]
                    )
                )
            ),
            html.Br(),
            dcc.Dropdown(
                className='drops',
                id='my-dropdown-3',
                options=data_options,
                value='sum_click',
            ),
            html.Br(),
            dcc.Graph(
                id='graph-2-tabs',
                figure=go.Figure(
                    data=[
                        go.Bar(
                            x=df.code_module.unique().tolist(),
                            y=df[df['final_result']=='Withdrawn'].groupby('code_module')['sum_click'].mean().tolist(),
                            name='Withdrawn',
                            marker=go.bar.Marker(
                                color='rgb(108, 109, 109)'
                            )
                        ),
                        go.Bar(
                            x=df.code_module.unique().tolist(),
                            y=df[df['final_result']=='Fail'].groupby('code_module')['sum_click'].mean().tolist(),
                            name='Fail',
                            marker=go.bar.Marker(
                                color='rgb(237, 0, 0)'
                            )
                        ),
                        go.Bar(
                            x=df.code_module.unique().tolist(),
                            y=df[df['final_result']=='Pass'].groupby('code_module')['sum_click'].mean().tolist(),
                            name='Pass',
                            marker=go.bar.Marker(
                                color='rgb(13, 216, 84)'
                            )
                        ),
                        go.Bar(
                            x=df.code_module.unique().tolist(),
                            y=df[df['final_result']=='Distinction'].groupby('code_module')['sum_click'].mean().tolist(),
                            name='Distinction',
                            marker=go.bar.Marker(
                                color='rgb(45, 166, 173)'
                            )
                        )
                    ],
                    layout=go.Layout(
                        title='Average Clicks Per Student',
                        showlegend=True,
                        legend=go.layout.Legend(
                            x=0,
                            y=1.0
                        ),
                        margin=go.layout.Margin(l=40, r=0, t=40, b=30),
                        plot_bgcolor = colors['background'],
                        paper_bgcolor = colors['background'],
                        font = {
                            'color' : colors['text']
                        }
                    )
                ),
            ),
            html.Br(),
            dcc.Graph(
                id='graph-3-tabs',
                figure=go.Figure(
                    data=[
                        go.Bar(
                            x=df.code_module.unique().tolist(),
                            y=df[df['final_result']=='Withdrawn'].groupby('code_module')['days_interacted'].median().tolist(),
                            name='Withdrawn',
                            marker=go.bar.Marker(
                                color='rgb(108, 109, 109)'
                            )
                        ),
                        go.Bar(
                            x=df.code_module.unique().tolist(),
                            y=df[df['final_result']=='Fail'].groupby('code_module')['days_interacted'].median().tolist(),
                            name='Fail',
                            marker=go.bar.Marker(
                                color='rgb(237, 0, 0)'
                            )
                        ),
                        go.Bar(
                            x=df.code_module.unique().tolist(),
                            y=df[df['final_result']=='Pass'].groupby('code_module')['days_interacted'].median().tolist(),
                            name='Pass',
                            marker=go.bar.Marker(
                                color='rgb(13, 216, 84)'
                            )
                        ),
                        go.Bar(
                            x=df.code_module.unique().tolist(),
                            y=df[df['final_result']=='Distinction'].groupby('code_module')['days_interacted'].median().tolist(),
                            name='Distinction',
                            marker=go.bar.Marker(
                                color='rgb(45, 166, 173)'
                            )
                        )
                    ],
                    layout=go.Layout(
                        title='Median of Days Interacted Per Student',
                        showlegend=True,
                        legend=go.layout.Legend(
                            x=0,
                            y=1.0
                        ),
                        margin=go.layout.Margin(l=40, r=0, t=40, b=30),
                        plot_bgcolor = colors['background'],
                        paper_bgcolor = colors['background'],
                        font = {
                            'color' : colors['text']
                        }
                    )
                    ),
            )
        ])


# Update scatterplot
@app.callback(
    [Output('overview-graph1', 'figure'), Output('intro1', 'children')],
    [Input('my-dropdown-1', 'value'), Input('my-dropdown-2', 'value'), Input('slider-update','value'), Input('crossfilter-xaxis-type', 'value')])
def update_output_intro(input_value1, input_value2, selected_slider, x_axistype):
    '''
        return back updated graph
    '''
    app.config['suppress_callback_exceptions'] = True
    filtered_df = df[df['highest_education'] == slider_options[selected_slider]]
    return {
        'data':[
            go.Scatter(
                x = filtered_df[filtered_df['final_result'] == i][input_value1],
                y = filtered_df[filtered_df['final_result'] == i][input_value2],
                text = filtered_df[filtered_df['final_result'] == i]['id_student'],
                mode = 'markers',
                opacity = 0.6,
                marker = {
                    'size': 10,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name = i
            ) for i in filtered_df.final_result.unique()
        ],
        'layout': go.Layout(
            # title = 'Dynamic Data Visualization',
            xaxis = {'type': 'linear' if x_axistype == 'Linear' else 'log', 'title': input_value1, 'color': colors['text']},
            yaxis = {'title': input_value2, 'color': colors['text']},
            margin = {'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend = {'x': 0, 'y': 1},
            hovermode = 'closest',
            plot_bgcolor = colors['background'],
            paper_bgcolor = colors['background'],
            font = {
                'color' : colors['text']
            }
        )
    }, 'Here is a scatterplot describing the relationship between "{}" and "{}", categorised by code module:'.format(input_value1, input_value2)

@app.callback(
    Output('click-data', 'children'),
    [Input('overview-graph1', 'clickData')])
def display_click_data(clickData):
    '''
        returns data of each clicked point
    '''
    if clickData == None:
        return json.dumps(None,indent=2)
    print(clickData)
    a = json.dumps(clickData, indent=2)
    return (df[df['id_student'] == clickData['points'][0]['text']].to_json())

@app.callback(
    Output('graph-2-tabs', 'figure'),
    [Input('my-dropdown-3', 'value')])
def update_bar_graph(x_option):
    '''
        returns updated bar graph
    '''
    app.config['suppress_callback_exceptions'] = True
    return go.Figure(
        data=[
            go.Bar(
                x=df.code_module.unique().tolist(),
                y=df[df['final_result']=='Withdrawn'].groupby('code_module')[x_option].mean().tolist(),
                name='Withdrawn',
                marker=go.bar.Marker(
                    color='rgb(108, 109, 109)'
                )
            ),
            go.Bar(
                x=df.code_module.unique().tolist(),
                y=df[df['final_result']=='Fail'].groupby('code_module')[x_option].mean().tolist(),
                name='Fail',
                marker=go.bar.Marker(
                    color='rgb(237, 0, 0)'
                )
            ),
            go.Bar(
                x=df.code_module.unique().tolist(),
                y=df[df['final_result']=='Pass'].groupby('code_module')[x_option].mean().tolist(),
                name='Pass',
                marker=go.bar.Marker(
                    color='rgb(13, 216, 84)'
                )
            ),
            go.Bar(
                x=df.code_module.unique().tolist(),
                y=df[df['final_result']=='Distinction'].groupby('code_module')[x_option].mean().tolist(),
                name='Distinction',
                marker=go.bar.Marker(
                    color='rgb(45, 166, 173)'
                )
            ),
                        
        ],
        layout=go.Layout(
            title='Average "{}" Per Student'.format(x_option),
            showlegend=True,
            legend=go.layout.Legend(
                x=0,
                y=1.0
            ),
            margin=go.layout.Margin(l=40, r=0, t=40, b=30),
            plot_bgcolor = colors['background'],
            paper_bgcolor = colors['background'],
            font = {
                'color' : colors['text']
            }
        )
    )

if __name__ == '__main__':
    app.run_server(debug=True)