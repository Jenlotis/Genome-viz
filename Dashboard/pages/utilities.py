import dash_bootstrap_components as dbc
from dash import html


def render_layout(page_name, contents):
    return dbc.Container([
        dbc.Row([
            dbc.Row([
                html.H1(page_name,
                        style={'fontsize': 50, 'textAlign': 'center'},
                        className='display-1')
            ]),
            html.Hr(),
        ], className='bg-light'),
        dbc.Row([
            contents
        ])
    ], fluid=True, className='bg-light')
