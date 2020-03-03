# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import file_reader as fr
import json
import string
import pandas as pd
import numpy as np
import my_html_fncts as mhf
import text_handling as th
from os import listdir
from os.path import isfile, join
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

########################################################
# Data structures used to supply HTML with information #
########################################################

df = fr.read_file('../Bible_Comparison/data/')

df_text = df.copy()
for col in df_text.columns:
    try:
        df_text[col] = df_text[col].astype(int)
    except:
        continue


df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['word_density'] = df['char_count'] / (df['word_count']+1)
df['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
df['title_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
df['upper_case_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

# ylt_df = df_text.loc[((df_text.version == 'ylt') & (df_text.book == 'ruth'))]


####################################################
## Code to run the main part of the applicaitons ###
####################################################

# Below are the code blocks to properly setup the HTML page for bootstrap
# https://getbootstrap.com/docs/4.4/getting-started/introduction/#starter-template


# external JavaScript files for Bootstrap v4.4.1
external_scripts = [

    ## JQuery first, then Popper.js, then Bootstrap JS ##
    {
        'src': 'https://code.jquery.com/jquery-3.3.1.slim.min.js',
        'integrity': 'sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo',
        'crossorigin': 'anonymous'
    },
    'https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js',
    'https://cdn.datatables.net/1.10.20/js/dataTables.bootstrap4.min.js',
    'https://cdn.datatables.net/responsive/2.2.3/js/dataTables.responsive.min.js',
    'https://cdn.datatables.net/responsive/2.2.3/js/responsive.bootstrap4.min.js',
    {
        'src': "https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js",
        'integrity': 'sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49',
        'crossorigin': 'anonymous'
    },
    # 'https://code.jquery.com/jquery-3.3.1.js',
    # './other/custom-script.js',
    {
        'src': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js',
        'integrity': 'sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy',
        'crossorigin': 'anonymous' 
    },
    './other/custom-script.js',

]

# external CSS stylesheets for Bootstrap v4.4.1
external_stylesheets = [
    {
        'rel': 'stylesheet',
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    },
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css',
    'https://cdn.datatables.net/1.10.20/css/dataTables.bootstrap4.min.css',
    'https://cdn.datatables.net/responsive/2.2.3/css/responsive.bootstrap4.min.css'
]

# meta-tags for bootstrap to work as intended
meta_tags = [ 
    {
        'name': 'viewport',
        'content': "width=device-width, initial-scale=1, shrink-to-fit=no"
    },
]

app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets, 
                external_scripts=external_scripts,
                assets_ignore='.*?',
                # assets_ignore='\.json$' 
                meta_tags=meta_tags)

## Dataframes for general statistic graphs
sum_df = df.groupby(['version']).agg('sum')
mean_df = df.groupby(['version']).agg('mean')
median_df = df.groupby(['version']).agg('median')

metrics = sum_df.columns

list_of_books = df.book.unique()
version_list = df.version.unique()
stopwords = th.get_stop_words('../Bible_Comparison/resources/custom-stopwords.txt')

###########################
## Main Dash app layout ##
##########################
app.layout = html.Div([
    
    # Container hosting the Page Title
    html.Div([
        html.H1(children='Bible Comparison Dashboard'),
    ], className='row'),

    # Container holding tabs within Dash Board
    html.Div([

        # Tabs
        dcc.Tabs(id="tabs-styled-with-inline", value='sum', children=[
            dcc.Tab(label='Sum', value='sum'),
            dcc.Tab(label='Mean', value='mean'),
            dcc.Tab(label='Median', value='median'),
            # dcc.Tab(label='Tab 4', value='tab-4'),
        ]),

        # Radio buttons to select different metrics
        mhf.generate_radio_buttons(list_of_options=metrics, id='metric-radio-options'),

        # Container to hold the graphs to be displayed by tab & radio button selection
        html.Div(id='tabs-content-inline', className='col-xl')
    ], className='container'),

    # Page divider
    html.Hr(),

    # Dropdown selections and Word Cloud display
    html.Div([
        html.Div([

            # Dropdown selections for books and versions
            html.Div([
                html.H4('Book'),
                mhf.generate_dropdown(list_of_options=list_of_books, id='book-select'),
                html.H4('Version'),
                mhf.generate_dropdown(list_of_options=version_list, id='version-select'),
                mhf.generate_button()
            ], className='col-sm'),
            
            # Word cloud container for display
            html.Div(
                children=[
                    dcc.Graph(id='word-cloud-grph')
                ],
                className='col-8',
                id='word-cloud-div'
            )
        ], 
        className='row')
    ], 
    className='container-fluid'),

    # Page divider
    html.Hr(),

    # Container used to create and generate table
    mhf.generate_div_container(
        list_of_elements=[
            html.H2('Data-table of Scripture'),
            mhf.generate_data_table(df_text)
            # mhf.generate_table(
            #     ylt_df, 
            #     max_rows=int(ylt_df.shape[0]), 
            #     class_name='table table-hover table-bordered table-striped', 
            #     id='mydatatable'
            # )
        ],
        # class_name='table-responsive'
        class_name='container table-responsive'
    ),

    # Page divider
    html.Hr(),

    # Add DataTable script to the end of the tag
    html.Script(src="./assets/custom-script.js", type="text/javascript")

], className='container-fluid')

@app.callback(Output('tabs-content-inline', 'children'), 
                [Input('tabs-styled-with-inline', 'value'), Input('metric-radio-options', 'value')])
def render_content(tab, metric_options):

    dff = df.groupby(['version']).agg(tab)
    return html.Div([
        mhf.generate_bar_graph(df=dff, metric=metric_options, title=f'{metric_options.title()} {tab.title()}')
    ])

    ## Original code for tab display ##
    # if tab == 'sum':
    #     return html.Div([
    #         mhf.generate_bar_graph(df=sum_df, metric='char_count')
    #     ])
    # elif tab == 'mean':
    #     return html.Div([
    #         mhf.generate_bar_graph(df=mean_df, metric='char_count')
    #     ])
    # elif tab == 'median':
    #     return html.Div([
    #         mhf.generate_bar_graph(df=median_df, metric='char_count')
    #     ])
    # elif tab == 'tab-4':
    #     return html.Div([
    #         mhf.generate_bar_graph(df=sum_df, metric='char_count')
    #     ])

@app.callback(Output('word-cloud-grph', 'figure'),
                [Input('submit-button', 'n_clicks')],
                [State('book-select', 'value'), State('version-select', 'value')]
            )
def update_wordcloud(n_clicks, book_select, version_select):

    if not book_select or not version_select:
        raise PreventUpdate

    df_book = df.loc[( (df.book==book_select) & (df.version==version_select) )]
    wordcloud_image = th.generate_word_cloud(df=df_book, col='text', stopwords=stopwords, img_width=1300, img_height=1000)
    fig = mhf.display_wordcloud(img_path=wordcloud_image, width=1300, height=1000, scl_factor=0.5)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)