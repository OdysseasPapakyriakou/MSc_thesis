from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import pandas as pd
import os

import plotly.graph_objects as go

import base64

# from src.utils import visualize_kde_polar_plot

from dash_app.src.utils import visualize_kde_polar_plot

app = Dash(__name__)

exp = "experiment1"
subs = [int(sub) for sub in os.listdir("assets")]
experiments = [
    {"label": "Experiment 1: Multiple 3d arrays", "value": "experiment1"},
    {"label": "Experiment 2: Individual vs. average 3d", "value": "experiment2"},
    {"label": "Experiment 3: Utah vs. 3d", "value": "experiment3"}
]


def get_n_arrays(subject: int, hem: str, experiment: str):
    all_files = os.listdir(f"assets/{subject}/{hem}/{experiment}/")
    phos_files = [file for file in all_files if file.endswith(".csv") and file.startswith("arr")]
    return [i+1 for i in range(len(phos_files)-1)]


"""
aliceblue, antiquewhite, aqua, aquamarine, azure,
            beige, bisque, black, blanchedalmond, blue,
            blueviolet, brown, burlywood, cadetblue,
            chartreuse, chocolate, coral, cornflowerblue,
            cornsilk, crimson, cyan, darkblue, darkcyan,
            darkgoldenrod, darkgray, darkgrey, darkgreen,
            darkkhaki, darkmagenta, darkolivegreen, darkorange,
            darkorchid, darkred, darksalmon, darkseagreen,
            darkslateblue, darkslategray, darkslategrey,
            darkturquoise, darkviolet, deeppink, deepskyblue,
            dimgray, dimgrey, dodgerblue, firebrick,
            floralwhite, forestgreen, fuchsia, gainsboro,
            ghostwhite, gold, goldenrod, gray, grey, green,
            greenyellow, honeydew, hotpink, indianred, indigo,
            ivory, khaki, lavender, lavenderblush, lawngreen,
            lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, lightgray, lightgrey,
            lightgreen, lightpink, lightsalmon, lightseagreen,
            lightskyblue, lightslategray, lightslategrey,
            lightsteelblue, lightyellow, lime, limegreen,
            linen, magenta, maroon, mediumaquamarine,
            mediumblue, mediumorchid, mediumpurple,
            mediumseagreen, mediumslateblue, mediumspringgreen,
            mediumturquoise, mediumvioletred, midnightblue,
            mintcream, mistyrose, moccasin, navajowhite, navy,
            oldlace, olive, olivedrab, orange, orangered,
            orchid, palegoldenrod, palegreen, paleturquoise,
            palevioletred, papayawhip, peachpuff, peru, pink,
            plum, powderblue, purple, red, rosybrown,
            royalblue, rebeccapurple, saddlebrown, salmon,
            sandybrown, seagreen, seashell, sienna, silver,
            skyblue, slateblue, slategray, slategrey, snow,
            springgreen, steelblue, tan, teal, thistle, tomato,
            turquoise, violet, wheat, white, whitesmoke,
            yellow, yellowgreen
"""

app.layout = html.Div(style={'textAlign': 'center'}, children=[
    html.H3("3D Arrays in LH and RH"),
    html.Div([
            html.P("Select experiment:"),
            dcc.Dropdown(
                id="experiments-dropdown",
                options=experiments,
                value=experiments[0]["value"],   # default value
            )
        ]),
    html.Div([
        html.P("Select subject:"),
        dcc.Dropdown(
            id="subjects-dropdown",
            options=subs,
            value=114823,   # default value
        )
    ]),
    html.Div(id="n-arrays", style={'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Div(style={'flex': '50%', 'padding-right': '10px'}, children=[
            html.P("Select the number of arrays for RH:"),
            dcc.Dropdown(
                id='rh-array-dropdown',
                options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3},
                    {'label': '4', 'value': 4},
                    {'label': '5', 'value': 5}
                ],
                value=2  # default value
            )
        ]),
        html.Div(style={'flex': '50%', 'padding-left': '10px'}, children=[
            html.P("Select the number of arrays for LH:"),
            dcc.Dropdown(
                id='lh-array-dropdown',
                options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3},
                    {'label': '4', 'value': 4},
                    {'label': '5', 'value': 5}
                ],
                value=2  # default value
            )
        ])
    ]),
    html.Div(id="hem-graphs", style={'width': '80%', 'margin': 'auto'}, children=[
        html.Div(style={'margin-bottom': '20px'}, children=[
            html.H3("Title for Figures 1 and 2", style={'text-align': 'center'}),
            html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center'}, children=[
                    html.H4("Right Hemisphere 1"),
                    dcc.Graph(id="lh-graph1")
                ]),
                html.Div(style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center'}, children=[
                    html.H4("Left Hemisphere 1"),
                    dcc.Graph(id="rh-graph1")
                ])
            ])
        ]),
        html.Div(style={'margin-bottom': '20px'}, children=[
            html.H3("Title for Figures 3 and 4", style={'text-align': 'center'}),
            html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center'}, children=[
                    html.H4("Right Hemisphere 2"),
                    dcc.Graph(id="lh-graph2")
                ]),
                html.Div(style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center'}, children=[
                    html.H4("Left Hemisphere 2"),
                    dcc.Graph(id="rh-graph2")
                ])
            ])
        ])
    ]),

    html.Div(id="map-graphs", style={'width': '80%', 'margin': 'auto'}, children=[
        html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
            html.H4("Left visual field map"),
            html.Img(id="lh-bin-map-img", style={'width': '100%'})
        ]),
        html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
            html.H4("Right visual field map"),
            html.Img(id="rh-bin-map-img", style={'width': '100%'})
        ])
    ]),

    html.Div(id="phosphene-graphs", style={'width': '70%', 'margin': 'auto'}, children=[
        html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
            html.H4("Left visual field map"),
            html.Img(id="lh-map-img", style={'width': '100%'})
        ]),
        html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
            html.H4("Right visual field map"),
            html.Img(id="rh-map-img", style={'width': '100%'})
        ])
    ])

])


def create_hem_graph(hem_brain_coords_df, hem_arrays_df, hem: str, sel_arrays: int):
    sel_arrays_df = pd.DataFrame()
    for i in range(1, sel_arrays+1):
        array = "array" + str(i)
        arr_df = hem_arrays_df[hem_arrays_df["array"] == array]
        sel_arrays_df = pd.concat([sel_arrays_df, arr_df], axis=0, ignore_index=True)

    colors = ["green", "purple", "deeppink", "yellow", "maroon", "darkorange"]
    color_map = {
        label: colors[i]
        for i, label in enumerate(sel_arrays_df["array"].unique())
    }

    v1_df = hem_brain_coords_df[hem_brain_coords_df["v1"] == 1]
    no_v1 = hem_brain_coords_df[hem_brain_coords_df["v1"] != 1]
    fig_hem = go.Figure()
    fig_hem.add_trace(go.Scatter3d(x=no_v1["x"], y=no_v1["y"], z=no_v1["z"],
                                   mode="markers",
                                   name="brain area",
                                   marker=dict(size=3, opacity=0.2, color="dimgrey")))

    fig_hem.add_trace(go.Scatter3d(x=v1_df["x"], y=v1_df["y"], z=v1_df["z"],
                                   mode='markers',
                                   name="v1 area",
                                   marker=dict(size=5, opacity=0.2, color="lightblue")))

    for arr, color in color_map.items():
        df_label = sel_arrays_df[sel_arrays_df["array"] == arr]
        fig_hem.add_trace(go.Scatter3d(
            x=df_label["x"],
            y=df_label["y"],
            z=df_label["z"],
            mode='markers',
            marker=dict(size=1, opacity=0.6, color=color),
            name=arr
        ))

    if hem == "LH":
        fig_hem.update_layout(margin={"l": 0, "r": 0, "b": 50, "t": 50},
                              title={"x": 0.5, "font": {"size": 20}},
                              legend=dict(x=0.2, y=1, xanchor='center'))
    elif hem == "RH":
        fig_hem.update_layout(margin={"l": 0, "r": 0, "b": 50, "t": 50},
                              title={"x": 0.5, "font": {"size": 20}},
                              legend=dict(x=0.8, y=1, xanchor='center'))

    return fig_hem


def read_phos(subject: int, sel_arrays: int, hem: str):
    cumulative_phos = pd.DataFrame()
    for arr in range(1, sel_arrays+1):
        csv_filename = f"assets/{subject}/{hem}/experiment1/arr{arr}_phosphenes.csv"
        phos_df = pd.read_csv(csv_filename)
        cumulative_phos = pd.concat([cumulative_phos, phos_df], axis=0, ignore_index=True)

    return cumulative_phos.values


def read_map(subject: int, hem: str, experiment: str):
    image_path = f"./assets/{subject}/{hem}/{experiment}/binary_maps.png"
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

    return encoded_image


def read_density_map(subject: int, hem: str, experiment: str):
    image_path = f"./assets/{subject}/{hem}/{experiment}/density_phosphenes.png"
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

    return encoded_image


def create_n_arrays_dropdown(lh_n_arrays, rh_n_arrays, sel_arrays_lh, sel_arrays_rh):
    return html.Div(id="n-arrays", style={'width': '100%', 'display': 'flex'}, children=[
        html.Div(style={'flex': '50%'}, children=[
            html.P("Select the number of arrays for RH:"),
            dcc.Dropdown(
                id='rh-array-dropdown',
                options=rh_n_arrays,
                value=sel_arrays_rh  # default value
            )
        ]),
        html.Div(style={'flex': '50%'}, children=[
            html.P("Select the number of arrays for LH:"),
            dcc.Dropdown(
                id='lh-array-dropdown',
                options=lh_n_arrays,
                value=sel_arrays_lh  # default value
            )
        ])
    ])


def create_hemisphere_graphs(fig_rh_exp1, fig_rh_exp2, fig_lh_exp1, fig_lh_exp2, experiment):
    if experiment == "experiment2":
        return html.Div(id="hem-graphs", style={'width': '100%', 'margin': 'auto'}, children=[
            html.Div(style={'margin-bottom': '20px', 'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Right hemisphere", style={'text-align': 'center'}),
                    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                        html.Div(style={'flex': '1', 'textAlign': 'center'}, children=[
                            html.H4("Right hemisphere: Individualized"),
                            dcc.Graph(id="rh-ind", figure=fig_rh_exp1)
                        ]),
                        html.Div(style={'flex': '1', 'textAlign': 'center'}, children=[
                            html.H4("Right hemisphere: Average"),
                            dcc.Graph(id="rh-avg", figure=fig_rh_exp2)
                        ])
                    ])
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Left hemisphere", style={'text-align': 'center'}),
                    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                        html.Div(style={'flex': '1', 'textAlign': 'center'}, children=[
                            html.H4("Left hemisphere: Individualized"),
                            dcc.Graph(id="lh-ind", figure=fig_lh_exp1)
                        ]),
                        html.Div(style={'flex': '1', 'textAlign': 'center'}, children=[
                            html.H4("Left hemisphere: Average"),
                            dcc.Graph(id="lh-avg", figure=fig_lh_exp2)
                        ])
                    ])
                ])
            ])
        ])
    elif experiment == "experiment1":
        return html.Div(id="hem-graphs", style={'width': '100%', 'margin': 'auto', 'display': 'flex'}, children=[
                         html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
                             html.H4("Right Hemisphere"),
                             dcc.Graph(id="lh-graph", figure=fig_rh_exp1, style={'width': '100%'})
                         ]),
                         html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
                             html.H4("Left Hemisphere"),
                             dcc.Graph(id="rh-graph", figure=fig_lh_exp1, style={'width': '100%'})
                         ])
                     ])


def create_map_graphs(map_lvf_exp1, map_lvf_exp2, map_rvf_exp1, map_rvf_exp2, experiment):
    if experiment == "experiment2":
        return html.Div(id="map-graphs", style={'margin': 'auto'}, children=[
            html.Div(style={'margin-bottom': '20px', 'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Left visual field binary maps", style={'text-align': 'center'}),
                    html.Div(style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center'}, children=[
                        html.H4("Individualized"),
                        html.Img(id="lh-map-img1", src=f"data:image/png;base64,{map_lvf_exp1}", style={'width': '100%'})
                    ]),
                    html.Div(style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center'}, children=[
                        html.H4("Average"),
                        html.Img(id="rh-map-img1", src=f"data:image/png;base64,{map_lvf_exp2}", style={'width': '100%'})
                    ])
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Right visual field binary maps", style={'text-align': 'center'}),
                    html.Div(style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center'}, children=[
                        html.H4("Individualized"),
                        html.Img(id="lh-map-img2", src=f"data:image/png;base64,{map_rvf_exp1}", style={'width': '100%'})
                    ]),
                    html.Div(style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center'}, children=[
                        html.H4("Average"),
                        html.Img(id="rh-map-img2", src=f"data:image/png;base64,{map_rvf_exp2}", style={'width': '100%'})
                    ])
                ])
            ])
        ])
    elif experiment == "experiment1":
        return html.Div(id="map-graphs", style={'margin': 'auto'}, children=[
                html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                    html.H4("Left visual field binary map"),
                    html.Img(id="lh-map-img", src=f"data:image/png;base64,{map_lvf_exp1}", style={'width': '100%'})
                ]),
                html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                    html.H4("Right visual field binary map"),
                    html.Img(id="rh-map-img", src=f"data:image/png;base64,{map_rvf_exp1}", style={'width': '100%'})
                ])
            ])


def create_phosphene_graphs(density_lvf_exp1, density_lvf_exp2, density_rvf_exp1, density_rvf_exp2, experiment):
    if experiment == "experiment2":
        return html.Div(id="phosphene-graphs", style={'margin': 'auto'}, children=[
            html.Div(style={'margin-bottom': '20px', 'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Left visual field density maps", style={'text-align': 'center'}),
                    html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                        html.H4("Individualized"),
                        html.Img(id="lh-map-img1", src=f"data:image/png;base64,{density_lvf_exp1}", style={'width': '100%'})
                    ]),
                    html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                        html.H4("Average"),
                        html.Img(id="rh-map-img1", src=f"data:image/png;base64,{density_lvf_exp2}", style={'width': '100%'})
                    ])
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Right visual field density maps", style={'text-align': 'center'}),
                    html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                        html.H4("Individualized"),
                        html.Img(id="lh-map-img2", src=f"data:image/png;base64,{density_rvf_exp1}", style={'width': '100%'})
                    ]),
                    html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                        html.H4("Average"),
                        html.Img(id="rh-map-img2", src=f"data:image/png;base64,{density_rvf_exp2}", style={'width': '100%'})
                    ])
                ])
            ])
        ])
    elif experiment == "experiment1":
        return html.Div(id="phosphene-graphs", style={'margin': 'auto'}, children=[
                html.Div(style={'display': 'inline-block', 'width': '40%', 'textAlign': 'center'}, children=[
                    html.H4("Left visual field map"),
                    html.Img(id="lh-map-img", src=f"data:image/png;base64,{density_lvf_exp1}", style={'width': '100%'})
                ]),
                html.Div(style={'display': 'inline-block', 'width': '40%', 'textAlign': 'center'}, children=[
                    html.H4("Right visual field map"),
                    html.Img(id="rh-map-img", src=f"data:image/png;base64,{density_rvf_exp1}", style={'width': '100%'})
                ])
            ])

@app.callback(
    [Output('n-arrays', 'children'),
     Output('hem-graphs', 'children'),
     Output('map-graphs', 'children'),
     Output('phosphene-graphs', 'children')],
    [Input('experiments-dropdown', 'value')],
    [Input('subjects-dropdown', 'value')],
    [Input('lh-array-dropdown', 'value')],
    [Input('rh-array-dropdown', 'value')]
)
def update_graph(sel_experiment, subject, sel_arrays_lh, sel_arrays_rh):
    print(sel_experiment)
    lh_n_arrays = get_n_arrays(subject, "LH", "experiment1")
    rh_n_arrays = get_n_arrays(subject, "RH", "experiment1")

    # general
    lh_brain = pd.read_csv(f"assets/{subject}/LH/brain_coords.csv")
    rh_brain = pd.read_csv(f"assets/{subject}/RH/brain_coords.csv")

    # per experiment
    lh_arrays_exp1 = pd.read_csv(f"assets/{subject}/LH/experiment1/arrays.csv")
    rh_arrays_exp1 = pd.read_csv(f"assets/{subject}/RH/experiment1/arrays.csv")

    lh_arrays_exp2 = pd.read_csv(f"assets/{subject}/LH/experiment2/arrays.csv")
    rh_arrays_exp2 = pd.read_csv(f"assets/{subject}/RH/experiment2/arrays.csv")

    fig_rh_epx1 = create_hem_graph(rh_brain, rh_arrays_exp1, hem="RH", sel_arrays=sel_arrays_rh)
    fig_lh_exp1 = create_hem_graph(lh_brain, lh_arrays_exp1, hem="LH", sel_arrays=sel_arrays_lh)

    fig_rh_epx2 = create_hem_graph(rh_brain, rh_arrays_exp2, hem="RH", sel_arrays=sel_arrays_rh)
    fig_lh_exp2 = create_hem_graph(lh_brain, lh_arrays_exp2, hem="LH", sel_arrays=sel_arrays_lh)

    map_lvf_exp1 = read_map(subject, "RH", "experiment1")
    map_rvf_exp1 = read_map(subject, "LH", "experiment1")

    map_lvf_exp2 = read_map(subject, "RH", "experiment2")
    map_rvf_exp2 = read_map(subject, "LH", "experiment2")

    density_lvf_exp1 = read_density_map(subject, "RH", "experiment1")
    density_rvf_exp1 = read_density_map(subject, "LH", "experiment1")

    density_lvf_exp2 = read_density_map(subject, "RH", "experiment2")
    density_rvf_exp2 = read_density_map(subject, "LH", "experiment2")

    return (
        create_n_arrays_dropdown(lh_n_arrays, rh_n_arrays, sel_arrays_lh, sel_arrays_rh),
        create_hemisphere_graphs(fig_rh_epx1, fig_rh_epx2, fig_lh_exp1, fig_lh_exp2, sel_experiment),
        create_map_graphs(map_lvf_exp1, map_lvf_exp2, map_rvf_exp1, map_rvf_exp2, sel_experiment),
        create_phosphene_graphs(density_lvf_exp1, density_lvf_exp2, density_rvf_exp1, density_rvf_exp2, sel_experiment)
    )


if __name__ == '__main__':
    app.run_server(debug=True)