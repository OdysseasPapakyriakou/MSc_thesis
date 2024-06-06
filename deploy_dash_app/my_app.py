from dash import Dash, dcc, html, Input, Output

import pandas as pd
import os

import plotly.graph_objects as go

import base64

app = Dash(__name__)

subs = sorted([int(file) for file in os.listdir("assets") if "pdf" not in file])
experiments = [
    {"label": "Experiment 1: Multiple 3d arrays", "value": "experiment1"},
    {"label": "Experiment 2: Individual vs. average 3d", "value": "experiment2"},
    {"label": "Experiment 3: 5x3d vs 16xUtah", "value": "experiment3"}
]


def read_image(path: str):
    with open(path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

    return encoded_image


uu_logo_path = "src/uu.png"
enc_logo_img = read_image(uu_logo_path)
abstract_text = ["Recent advances in neurotechnology and artificial intelligence have paved the way "
                 "for restoring a rudimentary form of vision in blind patients by electrically "
                 "stimulating the visual areas of the brain. Finding the proper location in the "
                 "brain in a non-surgical way poses a significant challenge for the planning "
                 "and implementation of visual cortical prostheses.",
                 html.Br(), html.Br(),
                 "This research presents a Bayesian optimization pipeline for the informed "
                 "placement of multi-array brain electrodes in advanced visual cortical prostheses. "
                 "The procedure is based on individual MRI scans and retinotopy data from the "
                 "Human Connectome Project, allowing to find the array parameters and the "
                 "exact brain location that provide the closest match to a predetermined "
                 "visual field coverage. The procedure is tested for multiple arrays of two "
                 "types in the V1 area of the brain: 5 arrays of a theoretical three-dimensional "
                 "type of 1000 electrodes in each array; and 16 two-dimensional arrays of "
                 "100 electrodes each, modelling the Utah array, which is widely used in "
                 "research and clinical practice. The pipeline can easily be configured for "
                 "an arbitrary number of arrays, to simulate different array designs, "
                 "to cover a specific part of the visual field, and to include placement "
                 "in different cortical visual areas.",
                 html.Br(), html.Br(),
                 "Results show that the inclusion of additional arrays substantially increases "
                 "the visual field coverage compared to using one array, and that using "
                 "individual anatomical scans offers a great advantage compared to average "
                 "brain solutions. Therefore, the pipeline can be a valuable tool in designing and planning "
                 "novel types of brain-interfacing visual cortical prostheses."]


def get_n_arrays(subject: int, hem: str, experiment: str):
    all_files = os.listdir(f"assets/{subject}/{hem}/{experiment}/")
    phos_files = [file for file in all_files if file.endswith(".csv") and file.startswith("arr")]
    return [i+1 for i in range(len(phos_files)-1)]


app.layout = html.Div(style={'textAlign': 'center'}, children=[
    html.Div(style={'textAlign': 'center', 'margin-top': '50px'}, children=[
        html.H1("MSc thesis in Artificial Intelligence"),
        html.H2("Odysseas Papakyriakou",
                style={'font-weight': 'bold', 'font-size': '24px'}),
        html.H2("Bayesian optimization for the informed placement of multi-array "
                "brain electrodes in advanced visual cortical prostheses",
                style={'font-weight': 'normal', 'font-style': 'italic',
                       'width': '40%', 'margin': '0 auto', 'text-align': 'center'}),
        html.Br(),
        html.Img(id="uu-logo", src=f"data:image/png;base64,{enc_logo_img}", style={'width': '20%'}),
        html.P("Supervisor: Chris Klink", style={'font-size': '18px'}),
        html.P(children=[
            "This is a page for visualizing my results. "
            "You can download a pdf copy of the thesis ", html.A(
                "here.",
                href="/assets/Odysseas_AI_thesis.pdf",
                download="Odysseas_AI_thesis.pdf"
            )
        ], style={'font-size': '18px'}),
        html.P(children=[
            "And you can see all the code on my GitHub repo ", html.A(
                "here.",
                href="https://github.com/OdysseasPapakyriakou/MSc_thesis",
                target="_blank"   # opens link in a new tab
            )
        ], style={'font-size': '18px', 'display': 'block'}),
        html.H3("Abstract:"),
        html.P(abstract_text, style={'width': '50%', 'margin': '0 auto', 'text-align': 'justify', 'font-size': '16px'}),
        html.Br()
    ]),
    html.Div([
            html.B("Select experiment:", style={'font-size': '18px'}),
            dcc.Dropdown(
                id="experiments-dropdown",
                options=experiments,
                value=experiments[2]["value"],   # default value
            )
        ]),
    html.Div([
        html.B("Select subject:", style={'font-size': '18px'}),
        dcc.Dropdown(
            id="subjects-dropdown",
            options=subs,
            value=subs[0],   # default value
        )
    ]),
    html.Div(id="n-arrays", style={'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Div(style={'flex': '50%', 'padding-right': '10px'}, children=[
            html.B("Select the number of arrays for RH:", style={'font-size': '18px'}),
            dcc.Dropdown(
                id='rh-array-dropdown',
                options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3},
                    {'label': '4', 'value': 4},
                    {'label': '5', 'value': 5}
                ],
                value=3  # default value
            )
        ]),
        html.Div(style={'flex': '50%', 'padding-left': '10px'}, children=[
            html.B("Select the number of arrays for LH:", style={'font-size': '18px'}),
            dcc.Dropdown(
                id='lh-array-dropdown',
                options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3},
                    {'label': '4', 'value': 4},
                    {'label': '5', 'value': 5}
                ],
                value=3  # default value
            )
        ])
    ]),
    html.Br(), html.Br(),
    html.P(children=[
        "This interactive graph shows the array configurations and their "
        "exact location in the primary visual cortex (V1) of the brain after using the Bayesian "
        "optimization pipeline. The grey part is the whole grey matter area of each hemisphere, "
        "while the light blue part signifies the target location, the primary visual cortex (V1). "
        "The left hemisphere corresponds to vision in the right part of the visual field, "
        "and vice versa for the right hemisphere. The Bayesian optimization algorithm "
        "learns to place the arrays without any overlap in the target brain location, "
        "in a way that maximizes the generated artificial vision. Most importantly, "
        "the pipeline can be configured to specify the minimum distance between the arrays, "
        "to simulate different array designs, to cover a specific part of the visual field, "
        "and to include placement in different cortical visual areas."
    ],
        style={'width': '80%', 'margin': '0 auto', 'text-align': 'justify', 'font-size': '16px'}),
    html.Br(),
    html.Div(id="hem-graphs", style={'width': '80%', 'margin': 'auto'}, children=[]),
    html.Br(), html.Br(),
    html.P(children=[
        "When electrically stimulated, each array is able to generate artificial vision "
        "that corresponds to a specific part of the visual field. The binary maps show the "
        "part of the visual field corresponding to each array. "
        "Because the algorithm maximizes the array placement serially, "
        "the visual field coverage of each array decreases progressively."
    ],
        style={'width': '80%', 'margin': '0 auto', 'text-align': 'justify', 'font-size': '16px'}),
    html.Br(),
    html.Div(id="map-graphs", style={'width': '80%', 'margin': 'auto'}, children=[]),
    html.Br(), html.Br(),
    html.P(children=[
        "The last plots shows the estimated density of the artificially "
        "generated vision. This is based on the density of neurons corresponding to that "
        "brain location, as modelled in the retinotopy data. "
        "Higher neural density allows for more detailed visual perception."
    ],
        style={'width': '80%', 'margin': '0 auto', 'text-align': 'justify', 'font-size': '16px'}),
    html.Br(),
    html.Div(id="phosphene-graphs", style={'width': '70%', 'margin': 'auto'}, children=[]),

])


def create_hem_graph(hem_brain_coords_df, hem_arrays_df, hem: str, sel_arrays: int):
    sel_arrays_df = pd.DataFrame()
    for i in range(1, sel_arrays+1):
        array = "array" + str(i)
        arr_df = hem_arrays_df[hem_arrays_df["array"] == array]
        sel_arrays_df = pd.concat([sel_arrays_df, arr_df], axis=0, ignore_index=True)

    colors = ["black", "white", "red", "green", "darkturquoise", "yellow", "cyan", "magenta",
              "orange", "purple", "lime", "pink", "teal", "lavender", "darkorange", "brown"]

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
                              legend=dict(x=1.02, y=1, xanchor='center'))
    elif hem == "RH":
        fig_hem.update_layout(margin={"l": 0, "r": 0, "b": 50, "t": 50},
                              title={"x": 0.5, "font": {"size": 20}},
                              legend=dict(x=0.02, y=1, xanchor='center'))
    return fig_hem


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
            html.B("Select the number of arrays for RH:", style={'font-size': '18px'}),
            dcc.Dropdown(
                id='rh-array-dropdown',
                options=rh_n_arrays,
                value=sel_arrays_rh  # default value
            )
        ]),
        html.Div(style={'flex': '50%'}, children=[
            html.B("Select the number of arrays for LH:", style={'font-size': '18px'}),
            dcc.Dropdown(
                id='lh-array-dropdown',
                options=lh_n_arrays,
                value=sel_arrays_lh  # default value
            )
        ])
    ])


def create_hemisphere_graphs(fig_rh_exp1, fig_rh_comparison, fig_lh_exp1, fig_lh_comparison, experiment):
    if experiment == "experiment1":
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
    else:
        if experiment == "experiment2":
            title_3d_exp1 = "Individualized"
            title_3d_comparison = "Average"
        else:
            title_3d_exp1 = "3d"
            title_3d_comparison = "Utah"
        return html.Div(id="hem-graphs", style={'width': '100%', 'margin': 'auto'}, children=[
            html.Div(style={'margin-bottom': '20px', 'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Right hemisphere", style={'text-align': 'center'}),
                    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                        html.Div(style={'flex': '1', 'textAlign': 'center'}, children=[
                            html.H4(title_3d_exp1),
                            dcc.Graph(id="rh-ind", figure=fig_rh_exp1)
                        ]),
                        html.Div(style={'flex': '1', 'textAlign': 'center'}, children=[
                            html.H4(title_3d_comparison),
                            dcc.Graph(id="rh-avg", figure=fig_rh_comparison)
                        ])
                    ])
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Left hemisphere", style={'text-align': 'center'}),
                    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                        html.Div(style={'flex': '1', 'textAlign': 'center'}, children=[
                            html.H4(title_3d_exp1),
                            dcc.Graph(id="lh-ind", figure=fig_lh_exp1)
                        ]),
                        html.Div(style={'flex': '1', 'textAlign': 'center'}, children=[
                            html.H4(title_3d_comparison),
                            dcc.Graph(id="lh-avg", figure=fig_lh_comparison)
                        ])
                    ])
                ])
            ])
        ])


def create_map_graphs(map_lvf_exp1, map_lvf_exp2, map_rvf_exp1, map_rvf_exp2, experiment):
    if experiment == "experiment1":
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
    else:
        if experiment == "experiment2":
            title_3d_exp1 = "Individualized"
            title_3d_comparison = "Average"
        else:
            title_3d_exp1 = "3d"
            title_3d_comparison = "Utah"
        return html.Div(id="map-graphs", style={'margin': 'auto'}, children=[
            html.Div(style={'margin-bottom': '20px', 'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Left visual field binary maps", style={'text-align': 'center'}),
                    html.Div(style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center'}, children=[
                        html.H4(title_3d_exp1),
                        html.Img(id="lh-map-img1", src=f"data:image/png;base64,{map_lvf_exp1}", style={'width': '100%'})
                    ]),
                    html.Div(style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center'}, children=[
                        html.H4(title_3d_comparison),
                        html.Img(id="rh-map-img1", src=f"data:image/png;base64,{map_lvf_exp2}", style={'width': '100%'})
                    ])
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Right visual field binary maps", style={'text-align': 'center'}),
                    html.Div(style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center'}, children=[
                        html.H4(title_3d_exp1),
                        html.Img(id="lh-map-img2", src=f"data:image/png;base64,{map_rvf_exp1}", style={'width': '100%'})
                    ]),
                    html.Div(style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center'}, children=[
                        html.H4(title_3d_comparison),
                        html.Img(id="rh-map-img2", src=f"data:image/png;base64,{map_rvf_exp2}", style={'width': '100%'})
                    ])
                ])
            ])
        ])


def create_phosphene_graphs(density_lvf_exp1, density_lvf_exp2, density_rvf_exp1, density_rvf_exp2, experiment):
    if experiment == "experiment1":
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
    else:
        if experiment == "experiment2":
            title_3d_exp1 = "Individualized"
            title_3d_comparison = "Average"
        else:
            title_3d_exp1 = "3d"
            title_3d_comparison = "Utah"
        return html.Div(id="phosphene-graphs", style={'margin': 'auto'}, children=[
            html.Div(style={'margin-bottom': '20px', 'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Left visual field density maps", style={'text-align': 'center'}),
                    html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                        html.H4(title_3d_exp1),
                        html.Img(id="lh-map-img1", src=f"data:image/png;base64,{density_lvf_exp1}", style={'width': '100%'})
                    ]),
                    html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                        html.H4(title_3d_comparison),
                        html.Img(id="rh-map-img1", src=f"data:image/png;base64,{density_lvf_exp2}", style={'width': '100%'})
                    ])
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.H3("Right visual field density maps", style={'text-align': 'center'}),
                    html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                        html.H4(title_3d_exp1),
                        html.Img(id="lh-map-img2", src=f"data:image/png;base64,{density_rvf_exp1}", style={'width': '100%'})
                    ]),
                    html.Div(style={'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}, children=[
                        html.H4(title_3d_comparison),
                        html.Img(id="rh-map-img2", src=f"data:image/png;base64,{density_rvf_exp2}", style={'width': '100%'})
                    ])
                ])
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
    if sel_experiment != "experiment3":
        lh_n_arrays = get_n_arrays(subject, "LH", "experiment1")
        rh_n_arrays = get_n_arrays(subject, "RH", "experiment1")
    else:
        lh_n_arrays = get_n_arrays(subject, "LH", "experiment3")
        rh_n_arrays = get_n_arrays(subject, "RH", "experiment3")

    # general
    lh_brain = pd.read_csv(f"assets/{subject}/LH/brain_coords.csv")
    rh_brain = pd.read_csv(f"assets/{subject}/RH/brain_coords.csv")

    # per experiment
    lh_arrays_exp1 = pd.read_csv(f"assets/{subject}/LH/experiment1/arrays.csv")
    rh_arrays_exp1 = pd.read_csv(f"assets/{subject}/RH/experiment1/arrays.csv")

    lh_arrays_comparison = pd.read_csv(f"assets/{subject}/LH/{sel_experiment}/arrays.csv")
    rh_arrays_comparison = pd.read_csv(f"assets/{subject}/RH/{sel_experiment}/arrays.csv")

    fig_rh_epx1 = create_hem_graph(rh_brain, rh_arrays_exp1, hem="RH", sel_arrays=sel_arrays_rh)
    fig_lh_exp1 = create_hem_graph(lh_brain, lh_arrays_exp1, hem="LH", sel_arrays=sel_arrays_lh)

    fig_rh_comparison = create_hem_graph(rh_brain, rh_arrays_comparison, hem="RH", sel_arrays=sel_arrays_rh)
    fig_lh_comparison = create_hem_graph(lh_brain, lh_arrays_comparison, hem="LH", sel_arrays=sel_arrays_lh)

    map_lvf_exp1 = read_map(subject, "RH", "experiment1")
    map_rvf_exp1 = read_map(subject, "LH", "experiment1")

    map_lvf_comparison = read_map(subject, "RH", sel_experiment)
    map_rvf_comparison = read_map(subject, "LH", sel_experiment)

    density_lvf_exp1 = read_density_map(subject, "RH", "experiment1")
    density_rvf_exp1 = read_density_map(subject, "LH", "experiment1")

    density_lvf_comparison = read_density_map(subject, "RH", sel_experiment)
    density_rvf_comparison = read_density_map(subject, "LH", sel_experiment)

    return (
        create_n_arrays_dropdown(lh_n_arrays, rh_n_arrays, sel_arrays_lh, sel_arrays_rh),
        create_hemisphere_graphs(fig_rh_epx1, fig_rh_comparison, fig_lh_exp1, fig_lh_comparison, sel_experiment),
        create_map_graphs(map_lvf_exp1, map_lvf_comparison, map_rvf_exp1, map_rvf_comparison, sel_experiment),
        create_phosphene_graphs(density_lvf_exp1, density_lvf_comparison, density_rvf_exp1, density_rvf_comparison, sel_experiment)
    )


server = app.server
if __name__ == '__main__':
    app.run()
