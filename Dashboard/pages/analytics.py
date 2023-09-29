import dash
import dash_daq as daq
from .utilities import render_layout
import os
from dash import html, dcc, callback, Input, Output, State, ctx, dash_table
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from Bio import SeqIO
from Bio import Align
from Bio.Align import substitution_matrices
from dna_features_viewer import GraphicFeature, GraphicRecord
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.entropy import entropy
import matplotlib
import sklearn.preprocessing as pre
import html as internet
from html.parser import HTMLParser

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
                     'name': 'Other'})
    for index, row in pd.merge(param_df,
                               both_df,
                               indicator=True,
                               how='outer').query('_merge=="left_only"').drop('_merge',
                                                                              axis=1).iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': 'red',
                     'id': [row['protein_id'], row['name']],
                     'name': 'With amyloid signaling motifs'})
    for index, row in pd.merge(ann_df,
                               both_df,
                               indicator=True,
                               how='outer').query('_merge=="left_only"').drop('_merge',
                                                                              axis=1).iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': 'blue',
                     'id': [row['protein_id'], row['name']],
                     'name': 'With user selected annotations'})
    for index, row in both_df.iterrows():
        data.append({'start': row['start'],
                     'end': row['end'],
                     'color': 'magenta',
                     'id': [row['protein_id'], row['name']],
                     'name': 'With amyloid signaling motifs and user selected annotations'})
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
                                   showlegend=flag),
                        )
    return fig_data, trace_dict


def count_prot(df, annotations, cl):
    merged = pd.merge(df[['protein_id']],
                      annotations[['protein_id', 'code', 'name']],
                      how='left',
                      on='protein_id')
    merged = merged[merged['code'].isin(cl)]
    data = merged.groupby(['code'])[['name']].count().reset_index()
    non_zero = data['code'].unique()
    zero = [ann for ann in cl if ann not in non_zero]
    for ann in zero:
        data = pd.concat([data, pd.DataFrame([{'code': ann, 'name': 0}])], axis=0, ignore_index=True)
    return data.sort_values(by=['code']).rename(columns={'code': 'Annotation code', 'name': 'Number of proteins'})


def intervs_amyloid(df):
    chromosomes = df['chromosome'].unique()
    data = []
    for chr_num in chromosomes:
        temp_df = df[df['chromosome'] == chr_num].drop_duplicates(subset=['start', 'end'])
        start = temp_df['start'].tolist()
        end = temp_df['end'].tolist()
        for i in range(len(end) - 1):
            data.append(abs(start[i + 1] - end[i]))
    fig = go.Figure(go.Histogram(
        x=data,
        nbinsx=100,
        autobinx=False
    ))
    return fig


def calc_stats(df_list, codes_list):
    results = {'amyloid-prot': df_list[1].shape[0], 'ann-prot': count_prot(df_list[0], df_list[2], codes_list),
               'ann-amyloid': count_prot(df_list[1], df_list[2], codes_list),
               'percent-amyloid': df_list[1].shape[0] / df_list[0].shape[0] * 100,
               'intervs': intervs_amyloid(df_list[1])}
    return results


def make_complexity(sp,vista):
    seq = load_fasta_file(glob(f'{sp}/*.faa')[0])
    fs = FeatureSet("")
    fs.add(Feature(entropy, window=int(vista)).then(min))
    ent = [{'protein_id': s.identifier.split(" ", 1)[0], 'complexity': s.data[0]} for s in fs(seq)]
    pd.DataFrame(ent).to_csv(f'{sp}/complexity{vista}.csv')

dash.register_page(__name__, path='/analytics')

data_path = "./Data"
specimen_dirs = [dir_names for (dir_path, dir_names, file_names) in os.walk(data_path) if dir_names]

contents = html.Div(children=[
    dcc.Store(id='raw-df'),
    dcc.Store(id='target-df'),
    dcc.Store(id='trace-dict'),
    dcc.Store(id='annotations'),
    dcc.Dropdown(
        options=[{'label': f'{specimen}'.replace('_', ' '), 'value': f"{data_path}/{specimen}"} for specimen in specimen_dirs[0]],
        id='specimen-dropdown',
        clearable=False,
        placeholder='Select a specimen to analyze'
    ),

    html.Br(),
    dbc.Spinner(dbc.Container(id='parameters', fluid=True, style={'display': 'none'}, children=[
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
        html.Br(),
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
                html.B('Window size:'),
                html.Br(),
                dbc.Input(
                    id='vista',
                    value=10,
                    required=True
                )
            ]),
            dbc.Col([
                html.B('Maximum entropy for above specified window size(if 0 entropy will not be claculated):'),
                html.Br(),
                dbc.Input(
                    id='entropy-threshold',
                    value=0,
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
    ]), color='info'),
    html.Br(),
    dcc.Tabs([
        dcc.Tab(children=[
            html.Br(),
            dbc.RadioItems(
                ['Genome', 'Chromosome'],
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
            dcc.Graph(id='single-chromosome-graph',
                      style={'display': 'none'}),
            html.Div([
                html.B('Select a protein:'),
                html.Br(),
                dcc.Dropdown(multi=True,
                             clearable=True,
                             searchable=True,
                             id='gene-select'
                ),
                dcc.Dropdown(multi=True,
                             id='Aligment',
                             style={'display':'none'}
                ),
                dbc.Row([
                    dbc.Col(daq.BooleanSwitch(label="Alignment", labelPosition="top", id='aligment_switch', style={'display': 'none'}, on=False, disabled=False, color='#68CDA3')),
                    dbc.Col(daq.BooleanSwitch(label="Sort", labelPosition="top", id='sort_switch', style={'display': 'none'}, on=False, disabled=True, color='#68CDA3')),
                    dbc.Col(daq.BooleanSwitch(label="Scale", labelPosition="top", id='scale_switch', style={'display': 'none'}, on=False, disabled=True, color='#68CDA3'))
                ]),

                html.Br(),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.B('Score treshold:'),
                            dbc.Input(
                                id='score-threshold',
                                value=0,
                                required=True,
                                style={'width':125}
                            )
                        ])
                    ])
                ], style={'display': 'none'}, id='score-threshold-container'),
               html.Br(),

            ], style={'display': 'none'}, id='gene-select-container'),
            html.Div(id='results'),
            html.Div(id='align-res'),
            html.Br(),
        ], label='Visualization', value='viz'),
        dcc.Tab(children=[
            html.Br(),
            html.B('Select the scope of statistics:'),
            html.Br(),
            dbc.RadioItems(
                ['Full genome', 'Distinct chromosomes'],
                id='stats-source-select',
                inline=True
            ),
            html.Br(),
            html.B('Select chromosome(s) for statistics:'),
            html.Br(),
            dcc.Dropdown(
                id='stats-chromosome-select',
                disabled=True,
                multi=True,
                clearable=True
            ),
            html.Br(),
            dbc.Button(
                'Calculate statistics',
                id='gen-stats-button',
                n_clicks=0,
                class_name='col-2'
            ),
            html.Br(),
            html.Br(),
            html.Div(id='stats-results')
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
    features = pd.read_csv(glob(f'{specimen_path}/*.txt')[0],
                           sep="	",
                           low_memory=False)
    predictions = pd.read_csv(glob(f'{specimen_path}/*model123456.csv')[0],
                              sep="	",
                              header=None,
                              names=['protein_id', 'seq_start', 'seq_end', 'log_score', 'log_bias', 'seq'])
    annotations = pd.read_table(glob(f'{specimen_path}/*.tsv')[0],
                                header=None,
                                names=['protein_id', 'protein_subid', 'protein_len', 'base', 'code', 'name', 'start',
                                       'end', 'e-value', 'mark', 'date', 'seq', 'decr']
                                )
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
    Output('stats-chromosome-select', 'options'),
    Output('gene-select', 'value'),
    State('raw-df', 'memory'),
    State('feature-type', 'value'),
    State('ls-threshold', 'value'),
    State('ls-bias-margin', 'value'),
    State('entropy-threshold', 'value'),
    State('code-selection', 'value'),
    State('chromosome-select', 'value'),
    State('annotations', 'memory'),
    State('specimen-dropdown', 'value'),
    State('vista','value'),
    Input('viz-level', 'value'),
    prevent_initial_call=True
)
def initial_results(df_dict, ft, lst, lsbm, et, codes_list, chr_num_state, ann_dict, specimen_path, vista, viz_level):
    lst, lsbm, et = float(lst), float(lsbm), float(et)
    if codes_list is None:
        codes_list = []
    annotations = pd.DataFrame(ann_dict)
    ann_proteins = annotations[annotations['code'].isin(codes_list)]['protein_id'].unique()

    df = pd.DataFrame(df_dict)

    if et==0:
        base_df = df[df['# feature'] == ft]
        param_df = base_df[(base_df['log_score'] >= lst)
                         & (base_df['log_bias'] + lsbm < base_df['log_score'])]
    else:
        if not glob(f'{specimen_path}/complexity{vista}.csv'):
            make_complexity(specimen_path, vista)

        if 'complexity' not in df:
            motive_complexity = pd.read_csv(glob(f'{specimen_path}/complexity{vista}.csv')[0])
            df = pd.merge(df, motive_complexity, how='inner', on='protein_id')
        base_df = df[df['# feature'] == ft]
        param_df = base_df[(base_df['log_score'] >= lst)
                         & (base_df['log_bias'] + lsbm < base_df['log_score'])
                         & (base_df['complexity'] <= et)]
    ann_df = base_df[base_df['protein_id'].isin(ann_proteins)]
    both_df = param_df[param_df['protein_id'].isin(ann_proteins)]
    chromosomes = base_df['chromosome'].dropna().unique()

    if viz_level == 'Genome':

        colors = ['black', '#AFB0B0', 'red', 'blue', 'magenta']
        labels = ['No data',
                  'Other',
                  'With amyloid signaling motifs',
                  'With user selected annotations',
                  'With amyloid signaling motifs and user selected annotations']
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
        ax.tick_params(axis="y", length=0.75)
        ax.set_yticks(ticks=[-n - 1.25 for n in range(len(chromosomes))],
                      labels=chromosomes)
        ax.set_xticks(ticks=[], labels=[])
        ax.legend(loc='lower right', prop={'size': 8})
        fig.set_figwidth(15)
        fig.tight_layout()

        out_url = fig_to_uri(fig)
        return [dbc.Button('Save image'), html.Br(), html.Br(),
                html.Img(id='cur_plot', src=out_url, style={'width': '100%'})], {'display': 'none'}, dash.no_update, \
               {'display': 'none'}, base_df.to_dict('records'), param_df.to_dict('records'), {'display': 'none'}, \
               dash.no_update, chromosomes, dash.no_update

    elif viz_level == 'Chromosome':

        return html.Div(), {'display': 'block'}, chromosomes, dash.no_update, dash.no_update, dash.no_update, \
               {'display': 'none'}, chr_num_state, dash.no_update, []


@callback(
    Output('single-chromosome-graph', 'figure'),
    Output('single-chromosome-graph', 'style', allow_duplicate=True),
    Output('trace-dict', 'memory'),
    Output('gene-select-container', 'style', allow_duplicate=True),
    #Output('gene-select', 'value'),
    State('raw-df', 'memory'),
    State('target-df', 'memory'),
    State('code-selection', 'value'),
    State('viz-level', 'value'),
    State('annotations', 'memory'),
    Input('chromosome-select', 'value'),
    Input('gene-select', 'value'),
    Input('Aligment', 'value'),
    prevent_initial_call=True
)
def chromosome_results(base_dict, param_dict, codes_list, viz_lvl, ann_dict, chr_num, light, aligment):
    if chr_num and viz_lvl == 'Chromosome':
        if codes_list is None:
            codes_list = []

        annotations = pd.DataFrame(ann_dict)
        ann_proteins = annotations[annotations['code'].isin(codes_list)]['protein_id'].unique()
        base_df = pd.DataFrame(base_dict)
        param_df = pd.DataFrame(param_dict)
        base_df = base_df[base_df['chromosome'] == chr_num]
        param_df = param_df[param_df['chromosome'] == chr_num]
        data, trace_dict = broken_bars(make_chr_data(base_df, param_df, ann_proteins),  0, 4)
        fig = go.Figure(data)

        if light is not None:
            light = [light] if isinstance(light, str) else light
            lights=base_df.loc[base_df['protein_id'].isin(light)]
            for index, row in lights.iterrows():
                fig.add_annotation(x=((row['end']-row["start"])/2)+row["start"],
                                   y=4,
                                   showarrow=True,
                                   arrowhead=2,
                                   arrowsize=2,
                                   arrowwidth=1,
                                   arrowcolor="#26E70E",
                                   ax=0,
                                   hoverlabel_bgcolor="#26E70E",
                                   hovertext=row["protein_id"])
        if aligment is not None:
            aligments=base_df.loc[base_df['protein_id'].isin(aligment)]
            for index, row in aligments.iterrows():
                fig.add_annotation(x=((row['end']-row["start"])/2)+row["start"],
                                   y=0,
                                   showarrow=True,
                                   arrowhead=3,
                                   arrowsize=2,
                                   arrowwidth=1,
                                   arrowcolor="#e67300",
                                   ax=0,
                                   ay=25,
                                   hoverlabel_bgcolor="#e67300",
                                   hovertext=row["protein_id"])

        fig.update_layout(showlegend=True, plot_bgcolor='white',
                          legend=dict(orientation="h",
                                      yanchor="bottom",
                                      y=1.02,
                                      xanchor="left",
                                      x=0,
                                      itemclick=False,
                                      itemdoubleclick=False),
                          margin=dict(t = 2, b = 1, l = 3, r = 3),
                          height=250
                          )
        fig.update_xaxes({'showticklabels': False})
        fig.update_yaxes({'showticklabels': False})
        return fig, {'display': 'block'}, trace_dict, {'display': 'block'}#, clean
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update#, dash.no_update


@callback(
    Output('single-chromosome-graph', 'style', allow_duplicate=True),
    Output('gene-select', 'options'),
    Output('gene-select', 'value', allow_duplicate=True),
    Output('sort_switch', 'style'),
    Output('scale_switch', 'style'),
    Output('aligment_switch', 'style'),
    Output('score-threshold-container', 'style'),
    State('raw-df', 'memory'),
    State('chromosome-select', 'value'),
    State('trace-dict', 'memory'),
    State('gene-select', 'value'),
    Input('single-chromosome-graph', 'clickData'),
    prevent_initial_call=True
)
def seq_display(base_dict, chr_num_state, trace_dict, annotation, click_data):
    base_df = pd.DataFrame(base_dict)
    if ctx.triggered_id == 'single-chromosome-graph' and click_data['points'][0]['curveNumber'] != 0:
        options = base_df[base_df['chromosome'] == chr_num_state]['protein_id'].unique()
        selected = trace_dict[str(click_data['points'][0]['curveNumber'])]
        if annotation is not None and len(annotation) > 0:
            selected=appending(selected, annotation)
        return {'display': 'block'}, options, selected, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def appending(selected, annotation):
    annotation=[annotation] if len(annotation[0]) == 1 else annotation
    if selected not in annotation:
        annotation.append(selected)
    elif selected in annotation:
        annotation.remove(selected)
    else:
        annotation=[selected]
    return annotation


@callback(
    Output('results', 'children', allow_duplicate=True),
    Output('Aligment', 'value'),
    Output('sort_switch', 'disabled'),
    Output('scale_switch', 'disabled'),
    Output('aligment_switch', 'disabled'),
    Output('align-res', 'children'),
    State('specimen-dropdown', 'value'),
    State('target-df', 'memory'),
    State('annotations', 'memory'),
    State('gene-select', 'value'),
    State('raw-df', 'memory'),
    State('chromosome-select', 'value'),
    Input('sort_switch', 'on'),
    Input('scale_switch', 'on'),
    Input('aligment_switch', 'on'),
    Input('gene-select', 'value'),
    Input('score-threshold', 'value'),
    prevent_initial_call=True
)
def multi_viz(specimen_path, param_dict, ann_dict, annotation, raw_dict, chromosome, sorter, scaler, alignments, value, treshold):
    visuals=[]
    found=[]
    a=[]
    align_vis=[]
    if len(annotation) > 0:
        scale_list=[]
        lista = [annotation] if isinstance(annotation, str) else annotation
        if sorter:
            raw_df = pd.DataFrame(raw_dict)
            lista = raw_df.loc[raw_df['protein_id'].isin(lista)]['protein_id']

        if scaler:
            raw_df=pd.DataFrame(raw_dict)
            raw_df=raw_df.drop_duplicates(subset=['protein_id'])
            for i in lista:
                scale_list.append(raw_df.loc[raw_df['protein_id']==i]["product_length"].values)
            maxi=max(scale_list)
            ticks=round(float(maxi)/1000)*100
            temp=[]
            for i in scale_list:
                temp.append(i/maxi)
            scale_list=temp
        else:
            scale_list=[1]*len(lista)
            ticks=100

        if alignments and len(lista)==1:
            param_df=pd.DataFrame(param_dict)
            param_df=param_df.loc[param_df['chromosome']== chromosome]
            chosen=param_df.loc[param_df['protein_id'].isin(lista)]['seq'].values[0]
            for index, row in param_df.iterrows():
                aligner=Align.PairwiseAligner(mode="global", open_gap_score=-10, extend_gap_score=-0.5)
                aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
                aligns=aligner.align(chosen,row["seq"])
                for align in aligns:
                    if align.score >= int(treshold):
                        found.append(row["protein_id"])
                    break

                for align in aligns:
                    if align.score >= int(treshold):
                        a.append([row['protein_id'],align,align.score])
            for f in found:
                b=[]
                for j in a:
                    if j[0]==f:
                        b.append(j[1:])
                for i in b:
                    jjj=str(i[0]).split()
                    aaa=html.Div([html.Pre(f"{jjj[0]} {jjj[1]} {jjj[2]} {jjj[3]}\n       {jjj[4]} {jjj[5]} {jjj[6]}\n{jjj[7]}  {jjj[8]} {jjj[9]} {jjj[10]}")])    
                    # jjj.insert(1, " ")
                    # jjj.insert(3, " ")
                    # jjj.insert(5, " ")
                    # jjj.insert(7, "\n")
                    # jjj.insert(8, "       ")
                    # jjj.insert(10, " ")
                    # jjj.insert(12, " ")
                    # jjj.insert(14, "\n")
                    # jjj.insert(16, "  ")
                    # jjj.insert(18, " ")
                    # jjj.insert(20, " ")
                    # aaa="".join(jjj)
                    # print(aaa)

                align_vis.append(html.Details([html.Summary(f), html.Div([display_seq(specimen_path, param_dict, ann_dict, f, 0.9, ticks, alignments), aaa])]))

            found.remove(lista[0])

        for seq,scalers in zip(lista, scale_list):
                visuals.append(display_seq(specimen_path, param_dict, ann_dict, seq, scalers, ticks, False))
        if len(lista)>1:
            return visuals, found, False, False, True, []
        else:
            return visuals, found, True, True, False, align_vis
    return visuals, found, dash.no_update, dash.no_update, dash.no_update, dash.no_update


def display_seq(specimen_path, param_dict, ann_dict, seq, scaler, ticks, name):
    seq_len_dict = {seq_rec.id: len(seq_rec.seq) for seq_rec in SeqIO.parse(glob(f'{specimen_path}/*.faa')[0],
                                                                            'fasta')}
    param_df = pd.DataFrame(param_dict)
    annotations = pd.DataFrame(ann_dict)
    motifs = [{'protein_id': row['protein_id'],
               'code': row['seq'],
               'name': "amyloid signalig motif",
               'start': row['seq_start'],
               'end': row['seq_end']} for index, row in param_df.iterrows()]

    colors = ["#ffd700", "#00ff00", "#0000ff", "#ff0000", "#ff00ff", "#00ffff", "#800000", "#008000", "#000080",
              "#808000", "#800080", "#008080", "#808080", "#ffa500", "#a52a2a", "#ffff00", "#ff4500", "#da70d6",
              "#dc143c", "#00ced1"
              ]#zolty, jaskrawa zielen, mocny niebieski, czerwony, rozowy, cyjan, bordowy, zielony, ciemny niebieski,
               #brudny zolty, fiolet, ciemny cyjan, szary, blady pomarancz, brudna czewien(taka cegla), jasny zolty, pomarancz, lilia
               #rozowa czerwien, inny cyjan
    features = [GraphicFeature(start=row['start'], end=row['end'], label=f"{row['code']} - {row['name']}",
                                color=colors[index]) for
                index, row in annotations[annotations['protein_id'] == seq].reset_index().iterrows()]

    motif_features=[]
    for amyloid in motifs:
        if amyloid['protein_id'] == seq:
            label = GraphicRecord._format_label(None, label=f"{amyloid['code']} - {amyloid['name']}", max_label_length=200, max_line_length=100)
            motif_features = [GraphicFeature(start=amyloid['start'], end=amyloid['end'], label=label, color="#964201")]

    record = GraphicRecord(sequence_length=seq_len_dict[seq], features=features, ticks_resolution=ticks)
    record_amyloid = GraphicRecord(sequence_length=seq_len_dict[seq], features=motif_features, ticks_resolution=ticks)

    szer = float((16*scaler)+1)
    ax, _ = record.plot(figure_width=szer)
    ax, _ = record_amyloid.plot(ax=ax, figure_width=szer, max_label_length=67, max_line_length=30)
    plt.tight_layout()
    ploting = ax.figure
    out_url = fig_to_uri(ploting)
    w_size = float((95*scaler)+ 5)
    if name:
        return html.Div([html.Img(title=seq, id='cur_plot', src=out_url, style={'width': f'{w_size}%'}), html.Br(), html.Br()])
    else:
        return html.Div([seq, html.Br(), html.Img(title=seq, id='cur_plot', src=out_url, style={'width': f'{w_size}%'}), html.Br(), html.Br()])


@callback(
    Output('stats-chromosome-select', 'disabled'),
    Input('stats-source-select', 'value'),
    prevent_initial_call=True
)
def stats_scope(selection):
    if selection == 'Distinct chromosomes':
        return False
    elif selection == 'Full genome':
        return True
    else:
        return dash.no_update


@callback(
    Output('stats-results', 'children'),
    State('stats-source-select', 'value'),
    State('stats-chromosome-select', 'value'),
    State('raw-df', 'memory'),
    State('target-df', 'memory'),
    State('annotations', 'memory'),
    State('code-selection', 'value'),
    Input('gen-stats-button', 'n_clicks'),
    prevent_initial_call=True
)
def generate_stats(source, chromosomes, base_dict, param_dict, ann_dict, codes_list, n_clicks):
    if source is None:
        return dbc.Alert('No data to display. Please select the source!', color='warning')
    elif source == 'Distinct chromosomes' and chromosomes is None or chromosomes == []:
        return dbc.Alert('No data to display. Please select at least one chromosome!', color='warning')
    else:
        if codes_list is None:
            codes_list = []
        base_df = pd.DataFrame(base_dict)
        param_df = pd.DataFrame(param_dict)
        annotations = pd.DataFrame(ann_dict)
        if source == 'Full genome':
            data = [base_df, param_df, annotations]
        elif source == 'Distinct chromosomes':
            data = [base_df[base_df['chromosome'].isin(chromosomes)],
                    param_df[param_df['chromosome'].isin(chromosomes)],
                    annotations]
        else:
            data = []
        stats = calc_stats(data, codes_list)
        return [
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Center([
                            html.B('Total number of proteins with signalling amyloid motifs: '),
                            f"{stats['amyloid-prot']}"
                        ])
                    ]),
                    dbc.Col([
                        html.Center([
                            html.B(
                                'Percentage of proteins with signalling amyloid motifs across all encoded proteins: '),
                            f"{round(stats['percent-amyloid'], 3)}%"
                        ])
                    ])
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.B('Total number of proteins per annotation:'),
                        html.Br(),
                        dash_table.DataTable(
                            data=stats['ann-prot'].to_dict('records'),
                            columns=[{"name": i, "id": i} for i in stats['ann-prot'].columns]
                        )
                    ]),
                    dbc.Col([
                        html.B('Total number of proteins with signalling amyloid motifs per annotation: '),
                        html.Br(),
                        dash_table.DataTable(
                            data=stats['ann-amyloid'].to_dict('records'),
                            columns=[{"name": i, "id": i} for i in stats['ann-amyloid'].columns]
                        )
                    ])
                ]),
                html.Br(),
                html.B('Distribution of intervals between consecutive proteins with signalling amyloid motifs:'),
                html.Br(),
                dbc.Row([
                    dcc.Graph(figure=stats['intervs'])
                ])
            ])
        ]


def layout():
    return render_layout('Analytics', contents)
