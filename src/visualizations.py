import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import json

from src.utils import get_date, get_density


def visualize_array_map(arr_phosphene_map: np.ndarray, total_phosphene_map: np.ndarray, hem: str):
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.suptitle("Array and total phosphene maps")
    axs[0].imshow(arr_phosphene_map, cmap="seismic", vmin=0, vmax=np.max(arr_phosphene_map) / 100)
    axs[1].imshow(total_phosphene_map, cmap="seismic", vmin=0, vmax=np.max(total_phosphene_map) / 100)
    if hem == 'RH':
        [axs[i].invert_xaxis() for i in range(2)]
    plt.tight_layout()
    plt.show()
    plt.close()


def share_axes(axes, sharex=True, sharey=True):
    """Determines how the axes are shared in the phosphene map visualization.
    This function is called within create_sub_plots()."""
    if isinstance(axes, np.ndarray):
        axes = axes.flat
    elif isinstance(axes, dict):
        axes = list(axes.values())
    else:
        axes = list(axes)
    ax0 = axes[0]
    for ax in axes:
        if sharex:
            ax.sharex(ax0)
            if not ax.get_subplotspec().is_last_row():
                ax.tick_params(labelbottom=False)
        if sharey:
            ax.sharey(ax0)
            if not ax.get_subplotspec().is_first_col():
                ax.tick_params(labelleft=False)


def create_sub_plots(total_maps: int, all_phosphene_maps: dict, fig: plt.figure,
                     grid: plt.GridSpec, axes: list, n_cols: int, hem: str):
    """Creates the subplots corresponding to each phosphene map.
    This function is called within visualize_phosphene_maps().

    Parameters
    ----------
    total_maps : int
        The total number of phosphene maps
    all_phosphene_maps : dict
        The dictionary with the phosphene maps per array
    plt : plt.figure
    grid : plt.GridSpec
        This is necessary to visualize the different maps nicely
    axes : list
        A list of axes created from fig.add_subplot()
    n_cols : int
        The number of columns of the grid
    hem : str
        The subject's hemisphere. Necessary to determine the x-axis
    """
    r, c = 0, 0
    y_ticks = [0, 250, 500, 750]
    if total_maps < 5:
        x_ticks = [0, 250, 500, 750]
    elif total_maps < 13:
        x_ticks = [0, 200, 500, 800]
    else:  # 13-16
        x_ticks = [0, 500, 1000]
    for i in range(1, total_maps+1):
        ax = fig.add_subplot(grid[r, c])
        axes.append(ax)
        ax.imshow(all_phosphene_maps[i], cmap="seismic", vmin=0, vmax=np.max(all_phosphene_maps[i]) / 100)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        if hem == "RH":
            ax.invert_xaxis()
        plt.title("Map " + str(i))
        if total_maps == 2:
            c += 1
        else:
            c += 1
            if c % (n_cols - 2) == 0:
                r += 1
                c = 0
        share_axes(axes, sharex=False, sharey=True)


def visualize_phosphene_maps(all_phosphene_maps: dict, total_phosphene_map: np.ndarray,
                             out_path: str, sub: str, hem: str, config: json, best: bool = False,
                             show: bool = False, save: bool = True):
    """Creates and saves a visualization of the phosphene maps for each array,
    as well as the total phosphene map from all arrays combined.
    The visualization is configured specifically to the number of phosphene maps.

    Parameters
    ----------
    all_phosphene_maps : dict
        The dictionary with the phosphene maps per array
    total_phosphene_map : np.ndarray
        The resulting phosphene map from combining all array-specific phosphene maps
    out_path : str
        The path to save the visualization
    sub : str
        The subject number
    hem : str
        The subject's hemisphere
    config : json
        The configuration file
    best : bool
        Whether the phosphenes represent the best possible phosphenes that can be obtained.
        This is used to create the appropriate title, filename, and visualization.
    show : bool
        If true, it will display the plot
    save : bool
        If true, it will save the plot in the specified path created from the parameters
    """
    sub_hem_path = out_path + sub + "/" + hem + "/"
    name = "_phosphene_maps_" if not best else "_best_possible_phosphene_map_"
    filename = sub + "_" + hem + name + get_date()
    if best:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(total_phosphene_map, cmap="seismic", vmin=0, vmax=np.max(total_phosphene_map / 100))
        if hem == "RH":
            ax.invert_xaxis()
        plt.title(f"Best possible phosphene map for {sub}, {hem}")
        if save:
            plt.savefig(sub_hem_path + filename + ".png")
        if show:
            plt.show()
        plt.close()
        return

    fig = plt.figure(figsize=(12, 6))
    total_maps = len(all_phosphene_maps)
    if total_maps > 16:
        print(f"Function {visualize_phosphene_maps.__name__} is configured for a max of 16 arrays.")
        return

    axes = []
    if total_maps == 1:
        grid = plt.GridSpec(1, 1)
        ax = fig.add_subplot(grid[0, 0])
        ax.imshow(all_phosphene_maps[1], cmap="seismic", vmin=0, vmax=np.max(all_phosphene_maps[1]) / 100)
        if hem == "RH":
            ax.invert_xaxis()
        plt.title("Phosphene map from one array")
        if save:
            plt.savefig(sub_hem_path + filename + ".png")
        if show:
            plt.show()
        plt.close()
        return

    elif total_maps == 2:
        n_cols = 4 if total_maps == 2 else 3
        n_rows = 1
        grid = plt.GridSpec(n_rows, n_cols, wspace=0.5, hspace=0.2)
        create_sub_plots(total_maps, all_phosphene_maps, fig, grid, axes, n_cols, hem)

    elif total_maps < 5:
        n_cols = int(np.ceil(total_maps / 2) + 2)
        n_rows = 2
        grid = plt.GridSpec(n_rows, n_cols, wspace=0.5, hspace=0.2)
        create_sub_plots(total_maps, all_phosphene_maps, fig, grid, axes, n_cols, hem)

    elif total_maps < 9:
        n_rows = 2
        n_cols = int(np.ceil(total_maps / 2) + 2)
        grid = plt.GridSpec(n_rows, n_cols, wspace=0.5, hspace=0.2)
        create_sub_plots(total_maps, all_phosphene_maps, fig, grid, axes, n_cols, hem)

    elif total_maps < 13:
        n_rows = 3
        n_cols = 5 if total_maps == 9 else 6
        grid = plt.GridSpec(n_rows, n_cols, wspace=0.5, hspace=0.5)
        create_sub_plots(total_maps, all_phosphene_maps, fig, grid, axes, n_cols, hem)

    else:  # 13 - 16
        n_rows = 4
        n_cols = 6
        grid = plt.GridSpec(n_rows, n_cols, wspace=0.5, hspace=0.7)
        create_sub_plots(total_maps, all_phosphene_maps, fig, grid, axes, n_cols, hem)

    ax = fig.add_subplot(grid[:n_rows, n_cols - 2:])
    ax.imshow(total_phosphene_map, cmap="seismic", vmin=0, vmax=np.max(total_phosphene_map / 100))
    if hem == "RH":
        ax.invert_xaxis()
    plt.title("Total phosphene map")
    plt.suptitle(f"Phosphene maps for {sub}, {hem}, for {total_maps} arrays of shape: "
                 f"{config['N_COMBS']}x{config['N_SHANKS_PER_COMB']}x{config['N_CONTACTPOINTS_SHANK']}",
                 fontsize=12)
    if save:
        plt.savefig(sub_hem_path + filename + ".png")
    if show:
        plt.show()
    plt.close()


def visualize_polar_plot(all_phosphenes: np.ndarray, out_path: str,
                         sub: str, hem: str, config: json, best: bool = False,
                         show: bool = False, save: bool = True):
    """Creates and saves the polar plot from the given phosphenes.

    Parameters
    ----------
    all_phosphenes : np.ndarray
        The numpy array with all phosphenes
    out_path : str
        The path to save the visualization
    sub : str
        The subject number
    hem : str
        The subject's hemisphere
    config : json
        The configuration file
    best : bool
        Whether the phosphenes represent the best possible phosphenes that can be obtained.
        This is used to create the appropriate title and filename.
    show : bool
        If true, it will display the plot
    save : bool
        If true, it will save the plot in the specified path created from the parameters
    """
    sub_hem_path = out_path + sub + "/" + hem + "/"
    name = "_polar_phosphenes_" if not best else "_best_polar_phospenes_"
    filename = sub_hem_path + sub + "_" + hem + name + get_date()
    phos = all_phosphenes.copy()

    polar_angles = phos[:, 0]
    eccentricities = phos[:, 1]
    rfsizes = phos[:, 2]
    theta = np.deg2rad(polar_angles)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(projection="polar")
    scatter = ax.scatter(theta, eccentricities, c=eccentricities, s=rfsizes*3, cmap="jet")
    ax.set_theta_zero_location("N")
    ax.set_thetalim(0, np.pi)
    ax.set_xticks(np.pi / 180. * np.linspace(180, 0, 5, endpoint=True))
    y_ticks = np.linspace(0, 90, 4, endpoint=True)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(["%.0f" % x for x in y_ticks])
    ax.tick_params(labelsize=10)
    ax.set_ylim(0, 90)
    cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label("Eccentricity")
    if hem == "RH":
        ax.set_theta_direction(1)
    else:
        ax.set_theta_direction(-1)
    title = f"Best possible phosphenes for {sub}, {hem}" if best else \
            (f"Phosphenes for {sub}, {hem}, \n "
             f"for {config['N_ARRAYS']} arrays of shape: "
             f"{config['N_COMBS']}x{config['N_SHANKS_PER_COMB']}x{config['N_CONTACTPOINTS_SHANK']}")

    plt.title(title, fontsize=12)
    if save:
        plt.savefig(filename + ".png")
    if show:
        plt.show()
    plt.close()


def visualize_kde_polar_plot(all_phosphenes: np.ndarray, out_path: str,
                             sub: str, hem: str, config: json, best: bool = False,
                             show: bool = False, save: bool = True):
    """Plots and saves a polar plot from the given phosphenes.
    The phosphenes are modeled with a Gaussian KDE, where the color indicates the density.

    Parameters
    ----------
    all_phosphenes : np.ndarray
        The numpy array with all phosphenes
    out_path : str
        The path to save the visualization
    sub : str
        The subject number
    hem : str
        The subject's hemisphere
    config : json
        The configuration file
    best : bool
        Whether the phosphenes represent the best possible phosphenes that can be obtained.
        This is used to create the appropriate title and filename.
    show : bool
        If true, it will display the plot
    save : bool
        If true, it will save the plot in the specified path created from the parameters
    """
    from math import radians
    from scipy.stats.kde import gaussian_kde

    sub_hem_path = out_path + sub + "/" + hem + "/"

    name = "_Gaussian_KDE_polar_phosphenes_" if not best else "_best_Gaussian_KDE_polar_phosphenes_"
    filename = sub_hem_path + sub + "_" + hem + name + get_date()
    colors_list = []
    phos_angle_in_radians = []
    for i in range(all_phosphenes.shape[0]):
        phos_angle_in_radians.append(radians(all_phosphenes[i, 0]))  # radians of polar angles
    colors_list.append(all_phosphenes[:, 1])

    # transform phosphenes to a 2d image
    ph_map = np.zeros((config["WINDOWSIZE"], config["WINDOWSIZE"], 1), "float32")
    if np.asarray(all_phosphenes).shape[0] > 0:
        ph_map, ph_density = get_density(all_phosphenes, "dens", config["WINDOWSIZE"], 3)
    ph_map = np.squeeze(ph_map)

    if ph_map.max() > 0:
        ph_map /= ph_map.max()
        ph_map /= ph_map.sum()

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(projection="polar")

    if len(phos_angle_in_radians) > 2:  # kde needs at least 3 points
        # combine x, y
        samples = np.vstack((phos_angle_in_radians, all_phosphenes[:, 1]))
        # create pdf
        dense_obj = gaussian_kde(samples)
        vals = dense_obj.evaluate(samples)
        cvals = vals * samples.shape[1]
    else:
        cvals = []

    ax.set_theta_zero_location("N")
    indices = np.argsort(cvals)
    cvals.sort()
    cvals = cvals[::-1]
    indices = indices[::-1]
    phos_angle_in_radians = [phos_angle_in_radians[i] for i in indices]
    phos = all_phosphenes.copy()  # not needed if all correct
    phos = phos[indices, :]
    scatter = ax.scatter(phos_angle_in_radians, phos[:, 1], s=phos[:, 2]*3,
                         c=cvals, cmap="jet", alpha=1)

    title = f"Best possible Gaussian KDE phosphenes for {sub}, {hem}" if best else \
            (f"Phosphenes for {sub}, {hem}, \n "
             f"for {config['N_ARRAYS']} arrays of shape: "
             f"{config['N_COMBS']}x{config['N_SHANKS_PER_COMB']}x{config['N_CONTACTPOINTS_SHANK']}")

    plt.title(title, fontsize=12)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rlim(0, 90)
    ax.set_xticks(np.pi / 180. * np.linspace(180, 0, 5, endpoint=True))
    y_ticks = np.linspace(0, 90, 4, endpoint=True)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(["%.0f" % x for x in y_ticks])
    cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label("Gaussian kernel density")
    if hem == "RH":
        ax.set_theta_direction(1)
    else:
        ax.set_theta_direction(-1)
    if save:
        plt.savefig(filename + ".png")
    if show:
        plt.show()
    plt.close()

def visualize_arrays_within_v1(v1_hem_coords: np.ndarray, arrays_df: pd.DataFrame, hem: str):
    v1_hem_df = pd.DataFrame(data={"x": v1_hem_coords[0, :],
                                   "y": v1_hem_coords[1, :],
                                   "z": v1_hem_coords[2, :]})

    fig = px.scatter_3d(v1_hem_df, x="x", y="y", z="z",
                        opacity=0.35,
                        title=f"V1 area of {hem} hemisphere")

    trace2 = go.Scatter3d(x=arrays_df["x"], y=arrays_df["y"], z=arrays_df["z"],
                          marker={"size": 5},
                          opacity=0.6,
                          name="Coordinates of optimized arrays")
    fig.add_trace(trace2)
    # tight layout
    fig.update_layout(margin={"l": 0, "r": 0, "b": 50, "t": 50},
                      title={"x": 0.5, "font": {"size": 20}})

    fig.update_traces(marker={"size": 5})
    fig.show()


def make_3d_plot(opt_arrays: dict):
    total_points = opt_arrays[1].shape[1]

    df = pd.DataFrame(data={"array": ["array1"] * total_points,
                            "x": opt_arrays[1][0, :],
                            "y": opt_arrays[1][1, :],
                            "z": opt_arrays[1][2, :]})

    for key, val in opt_arrays.items():
        if key == 1:
            continue
        df_next = pd.DataFrame(data={"array": [f"array{key}"] * total_points,
                                     "x": opt_arrays[key][0, :],
                                     "y": opt_arrays[key][1, :],
                                     "z": opt_arrays[key][2, :]})

        df = pd.concat([df, df_next], axis=0, ignore_index=True)

    fig = px.scatter_3d(df, x="x", y="y", z="z",
                        color="array",
                        opacity=0.6,
                        title="Coordinates of optimized arrays")
    # tight layout
    fig.update_layout(margin={"l": 0, "r": 0, "b": 50, "t": 50},
                      title={"x": 0.5, "font": {"size": 20}})

    fig.update_traces(marker={"size": 5})
    fig.show()