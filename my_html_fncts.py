import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import text_handling as th
from wordcloud import WordCloud


def generate_div_container(list_of_elements=None, class_name=''):
    return html.Div(children=list_of_elements, className=class_name)

def generate_data_table(df=None):
    return dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 25,
        # virtualization=True
    )

def generate_btstrp_table_header(list_of_headers):
       
    return html.Thead([
        html.Tr([
            html.Th(col, scope='col') for col in list_of_headers]
            )
    ])

def generate_tbl_row(list_of_items=None):
    
    size = len(list_of_items)
    lst = []

    row_head = html.Th([ list_of_items[0] ], scope='row')
    row_body = [ html.Td(list_of_items[i]) for i in range(1,size)]

    lst.append(row_head)
    lst += row_body
    
    return html.Tr(children=lst)

def generate_btsrp_table_body(df=None):
   
    rows = [ generate_tbl_row(df.iloc[i]) for i in range(0, df.shape[0]) ]
    return html.Tbody(children=rows)
    
def generate_btstrp_table(df, class_name='table', max_rows=10, id=''):
    return html.Table([
        generate_btstrp_table_header(df.columns),
        generate_btsrp_table_body(df=df)
    ], className=class_name, id=id)

def generate_bar_graph(df=None, metric='', title='Vizualization', id='example-graph'):
    return dcc.Graph(
        id='example-graph',
        figure={
            'data': [ 
                {'x': [item[0]], 'y': [item[1]], 'type':'bar', 'name': item[0] } 
                for item in df[metric].items() 
            ], 
            'layout': {'title': title}
        }
    )


def generate_dropdown(list_of_options=None, label='Dropdown Label', id='', multi_flag=False):
    opts = [ 
            {'label': opt.title().replace('_',' '), 'value':opt} 
            for opt in list_of_options
    ]

    return dcc.Dropdown(
        options = opts,
        value=opts[0]['value'],
        id=id,
        clearable=False,
        multi=multi_flag
    )

def generate_radio_buttons(list_of_options=None, id='radio-options'):

    opts = [ 
        {'label': opt.title().replace('_',' '), 'value':opt} 
        for opt in list_of_options
    ]
    
    return dcc.RadioItems(
        options = [{'label': option, 'value': option} for option in list_of_options],
        value=opts[0]['value'],
        labelClassName='font-italic',
        inputStyle={'display':'inline-block', 'margin-right': '5px'},
        labelStyle={'display': 'inline-block', 'margin-right': '10px'},
        id=id
    )

def display_wordcloud(img_path='', width=800, height=800, scl_factor=0.5, display_flag=False):

    fig = go.Figure()

    # Constants
    img_width = width
    img_height = height
    scale_factor = scl_factor

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img_path)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    if display_flag:
        # Disable the autosize on double click because it adds unwanted margins around the image
        # More detail: https://plot.ly/python/configuration-options/
        fig.show(config={'doubleClick': 'reset'})

    return fig.to_dict()

def generate_button(children='SUBMIT', type='button', id='submit-button', class_name='btn btn-primary btn-block', n_clicks=0):
    return html.Button( 
        children=children, 
        type=type, 
        id=id,
        className=class_name,
        n_clicks=n_clicks)