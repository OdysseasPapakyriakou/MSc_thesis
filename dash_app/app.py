from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import pandas as pd

import plotly.graph_objects as go
import matplotlib.pyplot as plt

app = Dash(__name__)

lh_brain = pd.read_csv('assets/114823/LH/brain_coords.csv')
lh_arrays = pd.read_csv('assets/114823/LH/arrays.csv')

rh_brain = pd.read_csv('assets/114823/RH/brain_coords.csv')
rh_arrays = pd.read_csv('assets/114823/RH/arrays.csv')

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
    html.Div(style={'width': '70%', 'margin': 'auto'}, children=[
        html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
            html.H4("Left Hemisphere"),
            dcc.Graph(id="lh-graph")
        ]),
        html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
            html.H4("Right Hemisphere"),
            dcc.Graph(id="rh-graph")
        ])
    ]),
    html.Div(style={'width': '50%', 'margin': 'auto'}, children=[
        html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
            html.H4("Left visual field map"),
            html.Img(src="./assets/114823/RH/density_phosphenes.png", style={'width': '100%'})
        ]),
        html.Div(style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'}, children=[
            html.H4("Right visual field map"),
            html.Img(src="./assets/114823/LH/density_phosphenes.png", style={'width': '100%'})
        ])
    ])
])


def create_hem_graph(hem_brain_coords_df, hem_arrays_df, hem: str):

    sel_arrays = 2

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

@app.callback(
    [Output("lh-graph", "figure"),
     Output("rh-graph", "figure")],
    [Input("lh-graph", "id")]  # You need to provide an input for the callback to trigger
)
def update_graph(_):
    fig_lh = create_hem_graph(lh_brain, lh_arrays, hem="LH")
    fig_rh = create_hem_graph(rh_brain, rh_arrays, hem="RH")

    return fig_lh, fig_rh


if __name__ == '__main__':
    app.run_server(debug=True)