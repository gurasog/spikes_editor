#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 19:27:29 2020

@author: gurasog
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:18:30 2020

@author: gurasog
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:22:11 2020

@author: gurasog
"""

import json
from scipy.io import loadmat
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

####### вот это то, что я буду менять ############
# df = pd.DataFrame({
#     "x": [1,2,1,2],
#     "y": [1,2,3,4],
#     "customdata": [1,2,3,4],
#     "fruit": ["apple", "apple", "orange", "orange"]
# })

# fig = px.scatter(df, x="x", y="y", color="fruit", custom_data=["customdata"])

# fig.update_layout(clickmode='event+select')

# fig.update_traces(marker_size=20)


import numpy as np
# get data
# from mne.datasets import sample
# data_path = sample.data_path()
# raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
# raw = mne.io.Raw(raw_fname, preload=False)
# picks = mne.pick_types(raw.info, meg='mag', exclude=[])
# start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
# data, times = raw[picks[:10], start:stop]


import plotly.io as pio
import plotly.express as px

pio.renderers.default = 'browser'

# turn data into dataframe
import pandas as pd;

# chnm = [];
# for i in range(0,10):
#     chnm.append(raw.info.ch_names[picks[i]]);

# chstd = np.mean(np.std(data.T,axis = 0));

# for i in range(0,data.shape[0]):
#     data[i,:] = data[i,:] + chstd*i;


'''
place for data

splines_starts=np.sort(np.random.randint(1,900,10))
splines_ends=splines_starts+21

splines=np.random.rand(1000,21)

'''

'''

'''

# spikes_data_path='/Users/gurasog/Desktop/Master/2_BCI/16_Git/bci/Rat_classical_dataset.json'
# data_path='/Users/gurasog/Desktop/Master/2_BCI/16_Git/bci/Rat_classical_dataset.mat'


spikes_data_path='Rat_spikes_28_long_dataset.json'
data_path='Rat_spikes_28_long.mat'

############################# PART ONE ###############################################

with open(spikes_data_path) as json_file:
    spikes_data = json.load(json_file)

dataset_names = []

for i in range(len(spikes_data)):
    dataset_names.append(spikes_data[i]['dataset_name'])

if len(set(dataset_names)) == 1:
    print('There is only 1 dataset')

else:
    print('Not supported yet')

splines_ends = []
spikes_starts = []

for i in range(len(spikes_data)):
    splines_ends.append(spikes_data[i]['point_num'])
    time_window = len(spikes_data[i]['normal_spike'][0])
    spikes_starts.append(spikes_data[i]['point_num'] - time_window)

    # проверять, чтобы они были не заданной длины
    # передавать длину первого сегмента из спайка

splines_ends = np.array(splines_ends)
splines_starts = np.array(spikes_starts)

#####


initial_range = 4
Fs = 250
data_mat_file = loadmat(data_path)
data_mat_file = data_mat_file['X'][0:6, :]
data = data_mat_file
times = np.linspace(0, 1 / Fs * data.shape[1], data.shape[1])

chstd = np.mean(np.std(data.T, axis=0))  # +10*np.std(np.std(data_mat_file.T,axis = 0));
# можно специфицировать под каждый канал по-своему
# если что можно менять масшаб по y

for i in [0, 1, 2, 3, 4, 5]:
    data[i, :] = data[i, :] + 3 * chstd * i;

chnm = ['1', '2', '3', '4', '5', '6']

df = pd.DataFrame(data=data_mat_file.T, columns=chnm)

# fig.update_layout(xaxis_range=[1000,2000])

fig = px.line(df, y=df.columns);
fig.update_layout(xaxis_range=[1000, 2000], xaxis=dict(

    rangeslider=dict(visible=True),
    type="linear")
                  )

######

shapes = []

for i in range(len(splines_starts)):
    x0 = splines_starts[i]
    x1 = splines_ends[i]
    dicti = dict(
        fillcolor="rgba(180, 20, 20, 0.2)",
        line={"width": 0},
        type="rect",
        x0=x0,
        x1=x1,
        xref="x",
        y0=0.05,
        y1=0.95,
        yref="paper")

    shapes.append(dicti)

fig.update_layout(
    shapes=shapes
)

####### вот это то, что я буду менять ############

app.layout = html.Div([

    dcc.Graph(
        id='time-series-plot',
        figure=fig
    ),

    html.Div([

        html.Button(id='previous-a-lot-button', n_clicks=0, children='<<'
                    ),

        html.Button(id='previous-button', n_clicks=0, children='<'
                    ),

        html.Button(id='next-button', n_clicks=0, children='>'
                    ),

        html.Button(id='next-a-lot-button', n_clicks=0, children='>>'
                    ),

        html.Div(id='output-state'),

        # latent variables
        html.Div(id='start-value-container', children=0, style={'display': 'none'}),
        html.Div(id='end-value-container', children=str(initial_range * Fs), style={'display': 'none'}),

        html.Div(id='translate-time-into-samples', children=str(Fs * initial_range), style={'display': 'none'}),
        html.Div(id='translate-spike-into-correct-spile', children=str(Fs * initial_range), style={'display': 'none'}),
        html.Div(id='spike-number-2', children='0', style={'display': 'none'}),

    ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}

    ),

    html.Div([

        html.Div(id='time-input-label;', children='Specify width of window in seconds'),

    ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}

    ),

    html.Div([

        dcc.Input(id='input-1-state', type='text', value=str(initial_range), size='0.5'),
        html.Button(id='submit-button-state', n_clicks=0, children='Ok'),

    ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}

    ),

    html.Div([

        html.Div(id='spike-input-label', children='Specify spike number'),

    ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}

    ),

    html.Div([

        dcc.Input(id='spike-number-input', type='text', value='1'),

        html.Button(id='ok-button', n_clicks=0, children='OK'
                    ),

    ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
    ),

    html.Div([

        html.Div(id='spike-number-with-text', children=' '),

    ]),

    html.Div([

        html.Button(id='previous-spike-button', n_clicks=0, children='Prev Spike'
                    ),

        html.Button(id='next-spike-button', n_clicks=0, children='Next Spike'
                    ),

    ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
    )

])


def check_slider_boundaries(start, end, period):
    if end <= len(times) and start >= 0:
        return start, end
    elif end > len(times):
        return len(times) - period, len(times)
    elif start < 0:
        return 0, period


@app.callback(

    Output('translate-time-into-samples', 'children'),
    Input('input-1-state', 'value'),
    State('translate-time-into-samples', 'children'))
def update_graph(input_1_state_value, translate_time_into_samples_children_state):
    try:

        int_input = int(input_1_state_value)
        return int_input * Fs

    except Exception:

        print('You need to put ineger value')
        return translate_time_into_samples_children_state

    return sample_value


@app.callback(

    Output('spike-input-label', 'children'),
    Input('spike-number-2', 'children'),

    Input('previous-button', 'n_clicks'),
    Input('previous-a-lot-button', 'n_clicks'),

    Input('next-a-lot-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    Input('submit-button-state', 'n_clicks')

)
# State('translate-time-into-samples', 'children'))

def update_graph(spike_number_2_children_input
                 , n_1, n_2, n_3, n_4, n_5
                 ):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if changed_id in ('previous-button.n_clicks', 'previous-a-lot-button.n_clicks'
                      , 'next-a-lot-button.n_clicks', 'next-button.n_clicks', 'submit-button-state.n_clicks'):

        return 'Specify spike number'


    else:
        return 'It is spike № ' + str(spike_number_2_children_input + 1)


@app.callback(

    Output('start-value-container', 'children'),
    Output('end-value-container', 'children'),

    Input('previous-button', 'n_clicks'),
    Input('previous-a-lot-button', 'n_clicks'),

    Input('next-a-lot-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),

    Input('spike-number-2', 'children'),
    Input('ok-button', 'n_clicks'),
    Input('previous-spike-button', 'n_clicks'),
    Input('next-spike-button', 'n_clicks'),

    Input('submit-button-state', 'n_clicks'),
    State('translate-time-into-samples', 'children'),

    State('start-value-container', 'children'),
    State('end-value-container', 'children'))
def update_graph(n_clicks_prev, n_clicks_prev_prev, n_clicks_next, n_clicks_next_next,

                 spike_number_2_input, n_clicks_ok, n_clicks_previous_spike, n_clicks_next_spike,

                 n_clicks, input1, state_start, state_end):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # print(changed_id)
    state_start_int = int(state_start)
    state_end_int = int(state_end)

    if changed_id == 'submit-button-state.n_clicks':

        try:

            int_input = int(input1)
            return [state_start_int, state_start_int + int_input]


        except Exception:
            print('lol')
            return [state_start_int, state_end_int]


    elif changed_id in ('ok-button.n_clicks', 'previous-spike-button.n_clicks', 'next-spike-button.n_clicks'):

        try:
            int_input = int(input1)
            int_spike_number_input_input = int(spike_number_2_input)

            spline_start = splines_starts[int_spike_number_input_input]
            spline_end = splines_ends[int_spike_number_input_input]
            spline_length = spline_end - spline_start;

            if spline_start - int_input / 2 < 0:
                scroller_start = 0
                scroller_end = scroller_start + int_input

            elif spline_end + int_input / 2 - spline_length > len(times):

                scroller_end = len(times)
                scroller_start = scroller_end - int_input

            else:

                scroller_start = spline_start - int_input / 2
                scroller_end = scroller_start + int_input

            return [scroller_start, scroller_end]


        except Exception:
            print('lol')
            return [state_start_int, state_end_int]



    elif changed_id == 'next-button.n_clicks':

        try:

            int_input = int(input1)
            return check_slider_boundaries(state_start_int + int_input / 2, state_end_int + int_input / 2, int_input)

        except Exception:
            print('lol')
            return [state_start_int, state_end_int]

    elif changed_id == 'previous-button.n_clicks':

        try:

            int_input = int(input1)
            return check_slider_boundaries(state_start_int - int_input / 2, state_end_int - int_input / 2, int_input)

        except Exception:
            print('lol')
            return state_start_int, state_end_int


    elif changed_id == 'next-a-lot-button.n_clicks':
        try:

            int_input = int(input1)

            # print(str(check_slider_boundaries(state_start_int+int_input, state_end_int+int_input, int_input)))
            return check_slider_boundaries(state_start_int + int_input, state_end_int + int_input, int_input)

        except Exception:

            return state_start_int, state_end_int


    elif changed_id == 'previous-a-lot-button.n_clicks':
        try:

            int_input = int(input1)
            return check_slider_boundaries(state_start_int - int_input, state_end_int - int_input, int_input)

        except Exception:
            print('lol')
            return [state_start_int, state_end_int]

    else:

        return [state_start_int, state_end_int]

    # elif changed_id=='n_clicks_prev.n_clicks':

    #     n_clicks_prev=


def which_splines_show(int_start, int_end):
    splines_starts_short = splines_starts - int_start
    splines_ends_short = splines_ends - int_start
    start = 0
    end = int_end - int_start

    spikes_to_show = []

    for i in range(len(splines_starts)):

        if (splines_starts_short[i] >= 0) and (splines_ends_short[i] <= end):
            spikes_to_show.append(i)

    return spikes_to_show


@app.callback(

    Output('time-series-plot', 'figure'),
    Input('start-value-container', 'children'),
    # Input('time-series-plot', 'figure'),
    Input('end-value-container', 'children'),

    Input('ok-button', 'n_clicks'),
    Input('previous-spike-button', 'n_clicks'),
    Input('next-spike-button', 'n_clicks'),

    State('spike-number-2', 'children'),

)
def update_graph(input_start, input_end,
                 n_clicks, n_clicks_ok, n_clicks_previous_next,
                 spike_number_2_state):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)

    int_start = int(input_start)
    int_end = int(input_end)

    times_to_present = times[int_start:int_end]

    data_to_present = data[:, int_start:int_end]
    df = pd.DataFrame(data=data_to_present.T, columns=chnm)
    fig = px.line(df, y=df.columns, x=times_to_present);
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),
                                 type="linear"))

    shapes = []

    splines_starts_short = splines_starts - int_start

    splines_ends_short = splines_ends - int_start

    splines_array = which_splines_show(int_start, int_end)

    for i in splines_array:
        x0 = splines_starts_short[i]
        x1 = splines_ends_short[i]

        dicti = dict(fillcolor="rgba(180, 20, 20, 0.2)", line={"width": 0}, type="rect", x0=times_to_present[x0],
                     x1=times_to_present[x1], xref="x",
                     y0=0.05, y1=0.95, yref="paper")
        shapes.append(dicti)

    fig.update_layout(shapes=shapes)

    if int_start == 0:
        fig.add_shape(x0=times_to_present[0], y0=0, x1=times_to_present[3], y1=1, type="rect", fillcolor='Red',
                      line={"width": 0}, xref="x", yref="paper")

    if int_end == len(times):
        fig.add_shape(x0=times_to_present[-3], y0=0, x1=times_to_present[-1], y1=1, type="rect", fillcolor='Red',
                      line={"width": 0}, xref="x", yref="paper")

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    print(changed_id)

    if changed_id in ('ok-button.n_clicks', 'previous-spike-button.n_clicks', 'next-spike-button.n_clicks'):
        int_spike_number_2_state = int(spike_number_2_state)
        print(int_spike_number_2_state)

        x0 = splines_starts_short[int_spike_number_2_state]
        x1 = splines_ends_short[int_spike_number_2_state]

        print(str(x0) + ' ' + str(x1))

        fig.add_shape(fillcolor="Red", line={"width": 0}, type="rect", x0=times_to_present[x0],
                      x1=times_to_present[x0 + 2], xref="x",
                      y0=0, y1=1, yref="paper")

    return fig


@app.callback(

    Output('spike-number-2', 'children'),

    Input('ok-button', 'n_clicks'),
    Input('previous-spike-button', 'n_clicks'),
    Input('next-spike-button', 'n_clicks'),

    State('spike-number-input', 'value'),
    State('spike-number-2', 'children')

)
def controle_spike_number(n_clicks_ok, n_clicks_previous_spike, n_clicks_next_spike, spike_number_input,
                          spike_number_state):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # print(changed_id)

    spike_number_state_int = int(spike_number_state)

    if changed_id == 'ok-button.n_clicks':

        try:

            int_spike_number_input = int(spike_number_input) - 1

            if int_spike_number_input >= len(splines_starts) or int_spike_number_input < 0:

                print('There is no spike with such number')
                return spike_number_state_int

            else:
                return int_spike_number_input



        except Exception:
            print('Your value is incorrect')
            return spike_number_state_int




    elif changed_id == 'previous-spike-button.n_clicks':

        potential_spike_number_state_int = spike_number_state_int - 1

        if potential_spike_number_state_int >= len(splines_starts) or potential_spike_number_state_int < 0:

            print('There is no spike there')
            return spike_number_state_int

        else:
            return potential_spike_number_state_int




    elif changed_id == 'next-spike-button.n_clicks':

        potential_spike_number_state_int = spike_number_state_int + 1

        if potential_spike_number_state_int >= len(splines_starts) or potential_spike_number_state_int < 0:

            print('There is no spike there')
            return spike_number_state_int

        else:
            return potential_spike_number_state_int

    else:
        return spike_number_state_int


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8115, debug=True, use_reloader=False, dev_tools_hot_reload=False)






