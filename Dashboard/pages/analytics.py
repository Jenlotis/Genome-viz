import dash
from .utilities import render_layout
import os
from dash import html, dcc, callback, Input, Output, State, ctx
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from Bio import SeqIO
from dna_features_viewer import GraphicFeature, GraphicRecord
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


def make_chr_data(base_df, param_df, ann_proteins):
    data = [{'start': 0,
             'end': base_df['end'].max(),
             'color': 'black',
             'id': ['base', ''],
             'name': 'No data'}]
    ann_df = base_df[base_df['protein_id'].isin(ann_proteins)]
    both_df = param_df[param_df['protein_id'].isin(ann_proteins)]
    grey_df = pd.merge(base_df,
                       param_df,
                       indicator=True,
                       how='outer').query('_merge=="left_only"').drop('_merge',
                                                                      axis=1)
    grey_df = pd.merge(grey_df,
                       ann_df,
                       indicator=True,
                       how='outer').query('_merge=="left_only"').drop('_merge',
                                                                      axis=1)
    for index, row in grey_df.iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': '#AFB0B0',
                     'id': [row['protein_id'], row['name']],
                     'name': 'Outside scope'})
    for index, row in pd.merge(param_df,
                               both_df,
                               indicator=True,
                               how='outer').query('_merge=="left_only"').drop('_merge',
                                                                              axis=1).iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': 'red',
                     'id': [row['protein_id'], row['name']],
                     'name': 'Filtered'})
    for index, row in pd.merge(ann_df,
                               both_df,
                               indicator=True,
                               how='outer').query('_merge=="left_only"').drop('_merge',
                                                                              axis=1).iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': 'blue',
                     'id': [row['protein_id'], row['name']],
                     'name': 'User\'s choice'})
    for index, row in both_df.iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': 'purple',
                     'id': [row['protein_id'], row['name']],
                     'name': 'Filtered and user\'s choice'})
    return data


def broken_bars(data, ystart, yh):
    fig_data = []
    trace_dict = {}
    trace_names = []
    for i, square in enumerate(data):
        if square['name'] not in trace_names:
            flag = True
            trace_names.append(square['name'])
        else:
            flag = False
        trace_dict[i] = square['id'][0]
        fig_data.append(go.Scatter(x=[square['start'], square['end'], square['end'], square['start']],
                                   y=[ystart] * 2 + [ystart + yh] * 2,
                                   fill='toself',
                                   hoveron='points+fills',
                                   fillcolor=square['color'],
                                   mode='lines',
                                   line_color=square['color'],
                                   name=square['name'],
                                   text=f"{square['id'][0]} ({square['id'][1]})",
                                   hoverinfo='text',
                                   showlegend=flag),
                        )
    return fig_data, trace_dict


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
# with open("..\\code-store.txt") as f:
#     codes = f.read().splitlines()
# ft_ref = {'mRNA': 'related_accession', 'CDS': 'product_accession'}

contents = html.Div(children=[
    dcc.Store(id='raw-df'),
    dcc.Store(id='target-df'),
    dcc.Store(id='trace-dict'),
    dcc.Store(id='annotations'),
    dcc.Dropdown(
        options=[{'label': f'{specimen}', 'value': f'{data_path}\\{specimen}'} for specimen in specimen_dirs[0]],
        id='specimen-dropdown',
        clearable=False,
        placeholder='Select a specimen to analyze'
    ),
    html.Br(),
    dbc.Container(id='parameters', fluid=True, style={'display': 'none'}, children=[
        dbc.Row([
            html.B('Select desired annotations:'),
            html.Br(),
            dcc.Dropdown(
                multi=True,
                clearable=True,
                searchable=True,
                id='code-selection'
            ),
            html.Br(),
            html.Br(),
            html.B('Select the feature type:'),
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
                html.B('Minimum \'log score\':'),
                html.Br(),
                dbc.Input(
                    id='ls-threshold',
                    value=18,
                    required=True
                )
            ]),
            dbc.Col([
                html.B('Minimum margin \'log score\' - \'log bias\':'),
                html.Br(),
                dbc.Input(
                    id='ls-bias-margin',
                    value=6,
                    required=True
                )
            ]),
            dbc.Col([
                html.B('Maximum entropy (window size 10):'),
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
    dcc.Tabs([
        dcc.Tab(children=[
            html.Br(),
            dbc.RadioItems(
                ['Genome', 'Chromosome', 'Protein'],
                inline=True,
                style={'display': 'none'},
                id='viz-level'
            ),
            html.Div([
                html.B('Select a chromosome:'),
                html.Br(),
                dcc.Dropdown(
                    id='chromosome-select'
                ),
                html.Br(),
            ], style={'display': 'none'}, id='chromosome-select-container'),
            html.Div([
                html.B('Select a protein:'),
                html.Br(),
                dcc.Dropdown(
                    id='gene-select'
                ),
            ], style={'display': 'none'}, id='gene-select-container'),
            html.Br(),
            html.Div(id='results'),
            dcc.Graph(id='single-chromosome-graph',
                      style={'display': 'none'})
        ], label='Visualization', value='viz'),
        dcc.Tab(children=[

        ], label='Statistics', value='stat')
    ], id='result-tabs', style={'display': 'none'}, value='viz'),
])


@callback(
    Output('parameters', 'style'),
    Output('code-selection', 'options'),
    Output('raw-df', 'memory', allow_duplicate=True),
    Output('annotations', 'memory'),
    Input('specimen-dropdown', 'value'),
    prevent_initial_call=True
)
def display_parameters(specimen_path):
    features = pd.read_csv(glob(f'{specimen_path}\\*.txt')[0],
                           sep="	",
                           low_memory=False)
    predictions = pd.read_csv(glob(f'{specimen_path}\\*.csv')[0],
                              sep="	",
                              header=None,
                              names=['protein_id', 'seq_start', 'seq_end', 'log_score', 'log_bias', 'seq'])
    annotations = pd.read_table(glob(f'{specimen_path}\\*.tsv')[0],
                                header=None,
                                names=['protein_id', 'protein_subid', 'protein_len', 'base', 'code', 'name', 'start',
                                       'end', 'e-value', 'mark', 'date', 'seq', 'decr']
                                )
    motive_complexity = pd.read_csv(glob(f'{specimen_path}\\*.ent')[0],
                                    sep='	',
                                    header=None,
                                    names=['protein_id', 'complexity'])
    features['protein_id'] = features.apply(lambda x: x['related_accession'] if x['# feature'] == 'mRNA'
    else (x['product_accession'] if x['# feature'] == 'CDS'
          else np.nan), axis=1)
    df = features[features['# feature'] != 'gene'][['# feature',
                                                    'chromosome',
                                                    'genomic_accession',
                                                    'start',
                                                    'end',
                                                    'strand',
                                                    'name',
                                                    'product_length',
                                                    'protein_id']]
    df = pd.merge(df, predictions, how='inner', on='protein_id')
    df = pd.merge(df, motive_complexity, how='inner', on='protein_id')
    code_options = [{'label': f'{row["base"]} - {row["code"]} ({row["name"]})',
                     'value': row['code']}
                    for index, row in annotations[['base', 'code', 'name']].drop_duplicates().iterrows()]

    return {'display': 'block'}, code_options, df.to_dict('records'), annotations.to_dict('records')


@callback(
    Output('result-tabs', 'style'),
    Output('viz-level', 'style'),
    Output('viz-level', 'value'),
    Input('go_button', 'n_clicks'),
    prevent_initial_call=True
)
def show_radio(n_clicks):
    return {'display': 'block'}, {'display': 'block'}, 'Genome'


@callback(
    Output('results', 'children', allow_duplicate=True),
    Output('chromosome-select-container', 'style'),
    Output('chromosome-select', 'options'),
    Output('single-chromosome-graph', 'style', allow_duplicate=True),
    Output('raw-df', 'memory'),
    Output('target-df', 'memory', allow_duplicate=True),
    Output('gene-select-container', 'style', allow_duplicate=True),
    Output('chromosome-select', 'value'),
    State('raw-df', 'memory'),
    State('feature-type', 'value'),
    State('ls-threshold', 'value'),
    State('ls-bias-margin', 'value'),
    State('entropy-threshold', 'value'),
    State('code-selection', 'value'),
    State('chromosome-select', 'value'),
    State('annotations', 'memory'),
    Input('viz-level', 'value'),
    prevent_initial_call=True
)
def initial_results(df_dict, ft, lst, lsbm, et, codes_list, chr_num_state, ann_dict, viz_level):
    lst, lsbm, et = float(lst), float(lsbm), float(et)
    if codes_list is None:
        codes_list = []
    annotations = pd.DataFrame(ann_dict)
    ann_proteins = annotations[annotations['code'].isin(codes_list)]['protein_id'].unique()
    df = pd.DataFrame(df_dict)
    base_df = df[df['# feature'] == ft]
    param_df = base_df[(base_df['log_score'] >= lst)
                       & (base_df['log_bias'] + lsbm < base_df['log_score'])
                       & (base_df['complexity'] <= et)]
    ann_df = base_df[base_df['protein_id'].isin(ann_proteins)]
    both_df = param_df[param_df['protein_id'].isin(codes_list)]
    chromosomes = base_df['chromosome'].dropna().unique()

    if viz_level == 'Genome':

        colors = ['black', '#AFB0B0', 'red', 'blue', 'magenta']
        labels = ['No data', 'Outside scope', 'Filtered', 'User\'s choice', 'Filtered and user\'s choice']
        fig, ax = plt.subplots(1, 1)
        for i, values in enumerate([full_range(base_df, chromosomes),
                                    known_range(base_df, chromosomes),
                                    known_range(param_df, chromosomes),
                                    known_range(ann_df, chromosomes),
                                    known_range(both_df, chromosomes)]):
            for chr_num in range(len(chromosomes)):
                if len(colors) - i <= len(labels):
                    label = labels.pop(0)
                else:
                    label = ""
                ax.broken_barh(values[chr_num], (-1 - chr_num, -0.5), color=colors[i], label=label)
        ax.set_yticks(ticks=[-n - 1.25 for n in range(len(chromosomes))],
                      labels=chromosomes)
        ax.set_xticks(ticks=[], labels=[])
        ax.legend(loc='lower right', prop={'size': 8})
        fig.set_figwidth(10)

        out_url = fig_to_uri(fig)
        return [dbc.Button('Save image'), html.Br(), html.Br(),
                html.Img(id='cur_plot', src=out_url, style={'width': '100%'})], {'display': 'none'}, dash.no_update, \
               {'display': 'none'}, base_df.to_dict('records'), param_df.to_dict('records'), {'display': 'none'}, \
               dash.no_update

    elif viz_level == 'Chromosome':

        return html.Div(), {'display': 'block'}, chromosomes, dash.no_update, dash.no_update, dash.no_update, \
               {'display': 'none'}, chr_num_state

    elif viz_level == 'Protein':

        return html.Div(), {'display': 'block'}, chromosomes, {'display': 'none'}, dash.no_update, dash.no_update, \
               dash.no_update, chr_num_state


@callback(
    Output('single-chromosome-graph', 'figure'),
    Output('single-chromosome-graph', 'style', allow_duplicate=True),
    Output('trace-dict', 'memory'),
    State('raw-df', 'memory'),
    State('target-df', 'memory'),
    State('code-selection', 'value'),
    State('viz-level', 'value'),
    State('annotations', 'memory'),
    Input('chromosome-select', 'value'),
    prevent_initial_call=True
)
def chromosome_results(base_dict, param_dict, codes_list, viz_lvl, ann_dict, chr_num):
    if chr_num and viz_lvl == 'Chromosome':
        if codes_list is None:
            codes_list = []
        annotations = pd.DataFrame(ann_dict)
        ann_proteins = annotations[annotations['code'].isin(codes_list)]['protein_id'].unique()
        base_df = pd.DataFrame(base_dict)
        param_df = pd.DataFrame(param_dict)
        base_df = base_df[base_df['chromosome'] == chr_num]
        param_df = param_df[param_df['chromosome'] == chr_num]
        data, trace_dict = broken_bars(make_chr_data(base_df, param_df, ann_proteins), 20, 9)
        fig = go.Figure(data)
        fig.update_layout(showlegend=True, plot_bgcolor='white', legend=dict(orientation="h",
                                                                             yanchor="bottom",
                                                                             y=1.02,
                                                                             xanchor="right",
                                                                             x=1)
                          )
        fig.update_xaxes({'showticklabels': False})
        fig.update_yaxes({'showticklabels': False})
        return fig, {'display': 'block'}, trace_dict
    else:
        return dash.no_update, dash.no_update, dash.no_update


@callback(
    Output('single-chromosome-graph', 'style', allow_duplicate=True),
    Output('gene-select-container', 'style', allow_duplicate=True),
    Output('gene-select', 'options'),
    Output('gene-select', 'value'),
    Output('viz-level', 'value', allow_duplicate=True),
    State('viz-level', 'value'),
    State('raw-df', 'memory'),
    State('chromosome-select', 'value'),
    State('trace-dict', 'memory'),
    State('gene-select', 'value'),
    Input('chromosome-select', 'value'),
    Input('single-chromosome-graph', 'clickData'),
    prevent_initial_call=True
)
def seq_display(viz_lvl, base_dict, chr_num_state, trace_dict, gene_state, chr_num, click_data):
    base_df = pd.DataFrame(base_dict)
    if chr_num is not None and viz_lvl == 'Protein':
        options = base_df[base_df['chromosome'] == chr_num]['protein_id'].unique()
        return {'display': 'none'}, {'display': 'block'}, options, gene_state, dash.no_update
    elif ctx.triggered_id == 'single-chromosome-graph' and click_data['points'][0]['curveNumber'] != 0:
        options = base_df[base_df['chromosome'] == chr_num_state]['protein_id'].unique()
        selected = trace_dict[str(click_data['points'][0]['curveNumber'])]
        return {'display': 'none'}, {'display': 'block'}, options, selected, 'Protein'
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output('results', 'children', allow_duplicate=True),
    State('specimen-dropdown', 'value'),
    Input('gene-select', 'value'),
    prevent_initial_call=True
)
def display_seq(specimen_path, annotation):
    if annotation is not None:
        seq_len_dict = {seq_rec.id: len(seq_rec.seq) for seq_rec in SeqIO.parse(glob(f'{specimen_path}\\*.faa')[0],
                                                                                'fasta')}
        annotations = pd.read_table(glob(f'{specimen_path}\\*.tsv')[0],
                                    header=None,
                                    names=['protein_id', 'protein_subid', 'protein_len', 'base', 'code', 'name',
                                           'start',
                                           'end', 'e-value', 'mark',
                                           'date', 'seq', 'decr']
                                    )
        colors = ["#ffd700", "#ffcccc", "#cffccc", "#ccccff", '#D3869A', '#F3B659', '#57E8E4']
        features = [GraphicFeature(start=row['start'], end=row['end'], label=f"{row['code']} - {row['name']}", color=colors[index]) for
                    index, row in annotations[annotations['protein_id'] == annotation].reset_index().iterrows()]
        record = GraphicRecord(sequence_length=seq_len_dict[annotation], features=features)
        ax, _ = record.plot(figure_width=5)
        out_url = fig_to_uri(ax.figure)
        return [dbc.Button('Save image'), html.Br(),
                html.Center(html.Img(id='cur_plot', src=out_url, style={'width': '80%'})), html.Br(), html.Br()]


def layout():
    return render_layout('Analytics', contents)
