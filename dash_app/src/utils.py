import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

matplotlib.use('agg')

import base64
from io import BytesIO


def get_density(phosphenes, phos_or_dens="dens", window_size=1000, ph_size_scale=3):
    from .ninimplant import pol2cart
    from .phos_elect import make_gaussian_v1
    import copy
    # Creating sizes
    # SIZE_POWER = 2 # original
    SIZE_COEFF = 0.01
    SIZE_BIAS = 1.5
    sizes = phosphenes[:, 1] * SIZE_COEFF + SIZE_BIAS  # size function of ecc
    sizes /= sizes.max()
    sizes *= 3

    # Important: we need the cartesian coordinates
    c_x, c_y = pol2cart(phosphenes[:, 0], phosphenes[:, 1])

    # Creating phosphene map (we already have it in the main code)
    phosphene_map = np.zeros((window_size, window_size, 1), 'float32')

    # create phosphene map
    max_eccentricity = np.max(phosphenes[:, 1])
    scaled_ecc = (window_size / 2) / max_eccentricity
    for i in range(phosphenes.shape[0]):
        s = int(sizes[i] * ph_size_scale)
        x = int(c_x[i] * scaled_ecc + window_size / 2)
        y = int(c_y[i] * scaled_ecc + window_size / 2)
        if s < 2:  # make super small phosphenes into size 2
            s = 2
        elif (s % 2) != 0:
            s = s + 1
        halfs = s // 2

        g = make_gaussian_v1(size=s, fwhm=s, center=None)
        g /= g.max()
        g /= g.sum()
        g = np.expand_dims(g, -1)
        try:
            phosphene_map[y - halfs:y + halfs, x - halfs:x + halfs] = phosphene_map[y - halfs:y + halfs,
                                                                      x - halfs:x + halfs] + g
        except ValueError:
            continue

    # Calculating a density map with a window that will smooth the density values
    density_window_size = 1  # 10
    density_map = np.zeros((phosphene_map.shape[0], phosphene_map.shape[1]))
    for i in range(phosphene_map.shape[0]):
        for j in range(phosphene_map.shape[1]):
            cuadradito = copy.deepcopy(phosphene_map[i - density_window_size:i + density_window_size,
                                       j - density_window_size:j + density_window_size])
            density = cuadradito.sum()
            density_map[i, j] = density

    # Getting the dense values corresponding to the dense map value at each phosphene's location
    # Choose to pick values from the phosphene map itself or from the smoothed density map
    density_values = []
    density = 0
    for i in range(c_x.shape[0]):
        x = int(c_x[i] * scaled_ecc + window_size / 2)
        y = int(c_y[i] * scaled_ecc + window_size / 2)
        if phos_or_dens == "dens":
            density = density_map[y, x]
        elif phos_or_dens == "phos":
            density = phosphene_map[y, x]
        else:
            print('wrong phos_or_dens')
        density_values.append(density)

    density_values = np.asarray(density_values)
    phosphene_map = np.rot90(phosphene_map, 1)

    return phosphene_map, density_values


def visualize_kde_polar_plot(all_phosphenes: np.ndarray,
                             sub: str, hem: str, sel_arrays: int):
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

    colors_list = []
    phos_angle_in_radians = []
    for i in range(all_phosphenes.shape[0]):
        phos_angle_in_radians.append(radians(all_phosphenes[i, 0]))  # radians of polar angles
    colors_list.append(all_phosphenes[:, 1])

    # transform phosphenes to a 2d image
    ph_map = np.zeros((1000, 1000, 1), "float32")
    if np.asarray(all_phosphenes).shape[0] > 0:
        ph_map, ph_density = get_density(all_phosphenes, "dens", 1000, 3)
    ph_map = np.squeeze(ph_map)

    if ph_map.max() > 0:
        ph_map /= ph_map.max()
        ph_map /= ph_map.sum()

    fig = plt.figure(figsize=(8, 8))
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

    if cvals.max() > 0:
        cvals /= cvals.max()

    indices = indices[::-1]
    phos_angle_in_radians = [phos_angle_in_radians[i] for i in indices]
    phos = all_phosphenes.copy()  # not needed if all correct
    phos = phos[indices, :]
    scatter = ax.scatter(phos_angle_in_radians, phos[:, 1], s=phos[:, 2]*3,
                         c=cvals, cmap="jet", alpha=1)

    title = (f"Cumulative phosphenes for {sub}, {hem}, \n "
             f"for {sel_arrays} array(s)")

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

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")

    # Reset the buffer position to the start
    buf.seek(0)
    # Convert the plot to a base64 encoded string
    encoded_plot = base64.b64encode(buf.getvalue()).decode()
    # Close the plot to release memory
    plt.close()

    return encoded_plot


def create_df_from_opt_dict(optimized_arrays: dict):
    """Creates a dataframe from the optimized arrays.
    Used to visualize the results.

    Parameters
    ----------
    optimized_arrays : dict
        The dictionary with the optimized arrays

    Returns
    -------
    pd.DataFrame
        A pandas dataframe to be used as input to the visualization function.
    """
    total_points = optimized_arrays[1].shape[1]
    df = pd.DataFrame(data={"array": ["array1"] * total_points,
                            "x": optimized_arrays[1][0, :],
                            "y": optimized_arrays[1][1, :],
                            "z": optimized_arrays[1][2, :]})

    for key, val in optimized_arrays.items():
        if key == 1:
            continue
        df_next = pd.DataFrame(data={"array": [f"array{key}"] * total_points,
                                     "x": optimized_arrays[key][0, :],
                                     "y": optimized_arrays[key][1, :],
                                     "z": optimized_arrays[key][2, :]})

        df = pd.concat([df, df_next], axis=0, ignore_index=True)
    return df


def create_dirs(out_path: str, sub: str, hem: str):
    """Creates the output directories, including the subject and hemisphere

    Parameters
    ----------
    out_path : str
        The path to the output directory
    sub : str
        The subject number
    hem : str
        The subject's hemisphere
    """
    sub_hem_path = out_path + sub + "/" + hem + "/"
    if not os.path.exists(sub_hem_path):
        os.makedirs(sub_hem_path, exist_ok=True)