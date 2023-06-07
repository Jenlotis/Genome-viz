import dash
from dash import html, dcc
from .utilities import render_layout

dash.register_page(__name__, path='/')

contents = html.Div(children=[
    html.Div(children='''
        This is our Home page content.
    '''),

])


def layout():
    return render_layout('Home Page', contents)
