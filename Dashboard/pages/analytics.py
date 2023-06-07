import dash
from .utilities import render_layout
import os
import random
from dash import html, dcc, callback, Input, Output, State, ctx
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import matplotlib

matplotlib.use('agg')


def fig_to_uri(in_fig, close_all=True, **save_args):
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def full_range(df, chr_array):
    output = []
    for chrom in chr_array:
        output.append([(0, df[df['chromosome'] == chrom]['end'].max())])
    return output


def known_range(df, chr_array):
    output = []
    for chrom in chr_array:
        chr_collection = []
        for index, row in df[df['chromosome'] == chrom].iterrows():
            chr_collection.append((row['start'], row['end'] - row['start']))
        output.append(chr_collection)
    return output


def make_chr_data(raw_df, col, tal, cn, feature_type):
    if len(tal) > 1:
        # colors = plotly.colors.n_colors('rgb(0, 0, 0)', 'rgb(255, 255, 255)', len(tal), colortype='Rainbow')
        colors = [f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'
                  for _ in range(len(tal))]
    else:
        colors = ['red']
    pairs = {acc: colors[i] for i, acc in enumerate(tal)}
    df = raw_df[(raw_df['chromosome'] == cn) & (raw_df['# feature'] == feature_type)]
    data = [{'start': 0,
             'end': df['end'].max(),
             'color': 'black',
             'id': 'base',
             'name': 'base'}]
    for index, row in df[~df[col].isin(tal)].iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': 'grey',
                     'id': row[col],
                     'name': row['name']})
    for index, row in df[raw_df[col].isin(tal)].iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': pairs[row[col]],
                     'id': row[col],
                     'name': row['name']})
    return data


def broken_bars(data, ystart, yh):
    fig_data = []
    for square in data:
        fig_data.append(go.Scatter(x=[square['start'], square['end'], square['end'], square['start']],
                                   y=[ystart] * 2 + [ystart + yh] * 2,
                                   fill='toself',
                                   fillcolor=square['color'],
                                   mode='lines',
                                   line_color=square['color'],
                                   name=square['id'],
                                   text=square['name'],
                                   hoverinfo='text + name'))

    return fig_data


def plot_seq(seq):
    fill_colors = {'A': '#5A6CFF', 'B': '#FF5D5D', 'C': '#FFF557', 'D': '#6ECE52'}

    fig = go.Figure(data=go.Scatter())

    for i, letter in enumerate(seq):
        fig.add_annotation(
            x=i + 1, y=3,
            text=f'<b>{letter}</b>',
            showarrow=False,
            font=dict(
                family="Arial",
                size=16,
                color='black',
            ),
            bordercolor='rgba(0, 0, 0, 0)',
            borderwidth=2,
            borderpad=24,
            bgcolor=fill_colors[letter]
        )

    fig.update_layout(
        plot_bgcolor="white",  # Set the background color to white
        width=800,
        height=400,
        xaxis=dict(
            showgrid=False,  # Hide the x-axis grid
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,  # Hide the y-axis grid
            showticklabels=False
        )
    )
    return fig


dash.register_page(__name__, path='/analytics')

data_path = "D:\Studia\Python_projects\Genome-viz\Data"
specimen_dirs = [dir_names for (dir_path, dir_names, file_names) in os.walk(data_path) if dir_names]
with open("..\\code-store.txt") as f:
    codes = f.read().splitlines()
ft_ref = {'mRNA': 'related_accession', 'CDS': 'product_accession'}

contents = html.Div(children=[
    dcc.Store(id='raw-df'),
    dcc.Store(id='target-df'),
    dcc.Dropdown(
        options=[{'label': f'{specimen}', 'value': f'{data_path}\\{specimen}'} for specimen in specimen_dirs[0]],
        id='specimen-dropdown',
        clearable=False,
        placeholder='Select a specimen to analyze'
    ),
    html.Br(),
    dbc.Container(id='parameters', style={'display': 'none'}, children=[
        dbc.Row([
            html.Label('Select desired annotations:'),
            html.Br(),
            dcc.Dropdown(
                options=codes,
                multi=True,
                clearable=True,
                searchable=True,
                id='code-selection'
            ),
            html.Br(),
            html.Br(),
            html.Label('Select the feature type:'),
            html.Br(),
            dbc.RadioItems(
                options=[
                    {'label': 'CDS (default)', 'value': 'CDS'},
                    {'label': 'mRNA', 'value': 'mRNA'}
                ],
                id='feature-type',
                value='CDS',
                inline=True
            ),
        ]),
        dbc.Row([
            dbc.Col([
                html.Label('Minimum \'log score\':'),
                html.Br(),
                dbc.Input(
                    id='ls-threshold',
                    value=18,
                    required=True
                )
            ]),
            dbc.Col([
                html.Label('Minimum margin \'log score\' - \'log bias\':'),
                html.Br(),
                dbc.Input(
                    id='ls-bias-margin',
                    value=6,
                    required=True
                )
            ]),
            dbc.Col([
                html.Label('Maximum entropy (window size 10):'),
                html.Br(),
                dbc.Input(
                    id='entropy-threshold',
                    value=1.5,
                    required=True
                )
            ])
        ]),
        html.Br(),
        dbc.Row([
            dbc.Button(
                'Process data',
                id='go_button',
                n_clicks=0,
                class_name='col-2'
            )
        ])
    ]),
    html.Br(),
    dbc.RadioItems(
        ['Genome', 'Chromosome', 'Gene'],
        inline=True,
        style={'display': 'none'},
        id='viz-level'
    ),
    html.Div([
        html.Label('Select a chromosome:'),
        html.Br(),
        dcc.Dropdown(
            id='chromosome-select'
        ),
    ], style={'display': 'none'}, id='chromosome-select-container'),
    html.Br(),
    html.Div([
        html.Label('Select a gene:'),
        html.Br(),
        dcc.Dropdown(
            id='gene-select'
        ),
    ], style={'display': 'none'}, id='gene-select-container'),
    html.Br(),
    html.Div(id='results'),
    dcc.Graph(id='single-chromosome-graph',
              style={'display': 'none'})
])


@callback(
    Output('parameters', 'style'),
    Input('specimen-dropdown', 'value'),
    prevent_initial_call=True
)
def display_parameters(selected_specimen):
    return {'display': 'block'}


@callback(
    Output('viz-level', 'style'),
    Output('viz-level', 'value'),
    Input('go_button', 'n_clicks'),
    prevent_initial_call=True
)
def show_radio(n_clicks):
    return {'display': 'block'}, 'Genome'


@callback(
    Output('results', 'children', allow_duplicate=True),
    Output('chromosome-select-container', 'style'),
    Output('chromosome-select', 'options'),
    Output('single-chromosome-graph', 'style', allow_duplicate=True),
    Output('raw-df', 'memory'),
    Output('target-df', 'memory'),
    State('feature-type', 'value'),
    State('ls-threshold', 'value'),
    State('ls-bias-margin', 'value'),
    State('entropy-threshold', 'value'),
    State('specimen-dropdown', 'value'),
    State('code-selection', 'value'),
    Input('viz-level', 'value'),
    prevent_initial_call=True
)
def initial_results(ft, lst, lsbm, et, specimen_path, codes_list, viz_level):
    lst, lsbm, et = float(lst), float(lsbm), float(et)
    features = pd.read_csv(glob(f'{specimen_path}\\*.txt')[0],
                           sep="	",
                           low_memory=False)
    predictions = pd.read_csv(glob(f'{specimen_path}\\*.csv')[0],
                              sep="	",
                              header=None,
                              names=['protein_id', 'start', 'end', 'log_score', 'log_bias', 'seq'])
    annotations = pd.read_table(glob(f'{specimen_path}\\*.tsv')[0],
                                header=None,
                                names=['protein_id', 'protein_subid', 'protein_len', 'base', 'code', 'name', 'start',
                                       'end', 'e-value', 'mark', 'date', 'seq', 'decr']
                                )
    motive_complexity = pd.read_csv(glob(f'{specimen_path}\\*.ent')[0],
                                    sep='	',
                                    header=None,
                                    names=['protein_id', 'complexity'])
    target_prediction_series = predictions[(predictions['log_score'] >= lst) &
                                           (predictions['log_bias'] + lsbm < predictions['log_score'])]['protein_id']
    target_complexity_series = motive_complexity[motive_complexity['complexity'] <= 1.5]['protein_id']
    if codes_list:
        target_annotation_series = annotations[annotations['code'].isin(codes_list)]['protein_id']
        target_proteins_set = set(target_prediction_series) & set(
            target_complexity_series) & set(target_annotation_series)
    else:
        target_proteins_set = set(target_prediction_series) & set(
            target_complexity_series)
    filtered_features = features[(features['related_accession'].isin(target_proteins_set)) | (
        features['product_accession'].isin(target_proteins_set)) & (features['# feature'] == ft)]
    chromosomes = features['chromosome'].dropna().unique()
    full_ranges = full_range(features, chromosomes)
    known_ranges = known_range(features, chromosomes)
    desired_ranges = known_range(filtered_features, chromosomes)

    if viz_level == 'Genome':

        colors = ['black', 'grey', 'red']
        fig, ax = plt.subplots(1, 1)
        for i, values in enumerate([full_ranges, known_ranges, desired_ranges]):
            for chr_num in range(len(chromosomes)):
                ax.broken_barh(values[chr_num], (1 + chr_num, .5), color=colors[i])
        ax.set_yticks(ticks=[n + 1.25 for n in range(len(chromosomes))],
                      labels=chromosomes)
        fig.set_figwidth(12)

        out_url = fig_to_uri(fig)
        return [dbc.Button('Save image'), html.Br(), html.Br(), html.Img(id='cur_plot', src=out_url, style={'width': '100%'})], {'display': 'none'}, dash.no_update,\
               {'display': 'none'}, features.to_dict('records'), filtered_features.to_dict('records')

    elif viz_level == 'Chromosome':

        return html.Div(), {'display': 'block'}, chromosomes, dash.no_update, dash.no_update, dash.no_update

    elif viz_level == 'Gene':

        return html.Div(), {'display': 'block'}, chromosomes, {'display': 'none'}, dash.no_update, dash.no_update


@callback(
    Output('single-chromosome-graph', 'figure'),
    Output('single-chromosome-graph', 'style', allow_duplicate=True),
    State('raw-df', 'memory'),
    State('target-df', 'memory'),
    State('feature-type', 'value'),
    State('viz-level', 'value'),
    Input('chromosome-select', 'value'),
    prevent_initial_call=True
)
def chromosome_results(raw_df, target_df, ft, viz_lvl, chr_num):
    if chr_num and viz_lvl == 'Chromosome':
        features = pd.DataFrame(raw_df)
        filtered_features = pd.DataFrame(target_df)
        target_accession_lst = filtered_features[filtered_features['chromosome'] == chr_num][ft_ref[ft]].unique()
        target_accession_lst = target_accession_lst[target_accession_lst != np.array(None)]
        data = make_chr_data(features, ft_ref[ft], target_accession_lst, chr_num, ft)
        fig = go.Figure(broken_bars(data, 20, 9))
        fig.update_layout(showlegend=False)
        return fig, {'display': 'block'}
    else:
        return dash.no_update, dash.no_update


@callback(
    Output('single-chromosome-graph', 'style', allow_duplicate=True),
    Output('gene-select-container', 'style'),
    Output('gene-select', 'options'),
    Output('gene-select', 'value'),
    State('viz-level', 'value'),
    State('raw-df', 'memory'),
    State('target-df', 'memory'),
    State('feature-type', 'value'),
    State('chromosome-select', 'value'),
    Input('chromosome-select', 'value'),
    Input('single-chromosome-graph', 'clickData'),
    prevent_initial_call=True
)
def seq_display(viz_lvl, raw_data_dict, data_dict, ft, chr_num_state, chr_num, click_data):
    raw_df = pd.DataFrame(raw_data_dict)
    df = pd.DataFrame(data_dict)
    if chr_num is not None and viz_lvl == 'Gene':
        options = df[df['chromosome'] == chr_num][ft_ref[ft]].unique()
        return {'display': 'none'}, {'display': 'block'}, options, dash.no_update
    elif ctx.triggered_id == 'single-chromosome-graph':
        options = df[df['chromosome'] == chr_num_state][ft_ref[ft]].unique()
        print(click_data['points'][0]['curveNumber'])
        return {'display': 'none'}, {'display': 'block'}, options, options[0]
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output('results', 'children', allow_duplicate=True),
    Input('gene-select', 'value'),
    prevent_initial_call=True
)
def display_seq(gene):
    return html.Div('ok')


def layout():
    return render_layout('Analytics', contents)
