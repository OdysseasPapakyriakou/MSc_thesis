import json
import os

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.distance import cdist

from src.phos_elect import get_cortical_magnification, cortical_spread, prf_to_phos
np.seterr(divide='ignore', invalid='ignore')


def get_phosphene_map(phosphenes, config):
    # the inverse cortical magnification in degrees (visual angle)/mm tissue
    M = 1 / get_cortical_magnification(phosphenes[:, 1], config["CORT_MAG_MODEL"])
    spread = cortical_spread(config["AMP"])  # radius of current spread in the tissue, in mm
    sizes = spread * M  # radius of current spread * cortical magnification = rf radius in degrees
    sigmas = sizes / 2  # radius to sigma of gaussian

    # phosphene size based on CMF + stim amp
    phosphenes[:, 2] = sigmas

    # generate map using Gaussians
    # transforming obtained phosphenes to a 2d image
    phosphene_map = np.zeros((config["WINDOWSIZE"], config["WINDOWSIZE"]), 'float32')
    phosphene_map = prf_to_phos(phosphene_map, phosphenes, view_angle=config["VIEW_ANGLE"], ph_size_scale=1)
    phosphene_map /= phosphene_map.max()
    phosphene_map /= phosphene_map.sum()

    return phosphene_map


def get_overlap_validity(optimized_arrays_from_f_manual, cur_contacts_xyz_moved, config):
    grid_valid_overlap = True
    if config["OVERLAP_METHOD"] == "distance":
        for key, arr in optimized_arrays_from_f_manual.items():
            min_distance = get_min_distance(arr, cur_contacts_xyz_moved)
            if min_distance < config["MINIMUM_DISTANCE"]:
                grid_valid_overlap = False
                break
        return grid_valid_overlap
    elif config["OVERLAP_METHOD"] == "convex hull":
        for key, arr in optimized_arrays_from_f_manual.items():
            mesh = trimesh.points.PointCloud(arr.T)
            mesh = mesh.convex_hull
            if any(mesh.contains(cur_contacts_xyz_moved.T)):
                grid_valid_overlap = False
                break   # if overlap with at least one array no need to go through all
        return grid_valid_overlap
    else:
        print("OVERLAP_METHOD needs to be either 'distance' or 'convex hull'. Returning False")
        return False


def read_config(file_path="config.json"):
    try:
        with open(file_path, "r") as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        # provide defaults
        print(f"Config file not found: {file_path}. Setting default parameters.")
        config = {
            "NUM_CALLS": 150,
            "INIT_ALPHA": 0,
            "INIT_BETA": 0,
            "INIT_OFFSET_FROM_BASE": 20,
            "SHANK_LENGTH": 9,
            "N_CONTACTPOINTS_SHANK": 8,
            "N_COMBS": 3,
            "N_SHANKS_PER_COMB": 8,
            "N_ARRAYS": 5,
            "SPACING_ALONG_XY": 1,
            "NUM_INITIAL_POINTS": 10,
            "DC_PERCENTILE": 50,
            "WINDOWSIZE": 1000,
            "CORT_MAG_MODEL": "wedge-dipole",
            "VIEW_ANGLE": 90,
            "AMP": 100,
            "A": 1,
            "B": 0.05,
            "C": 1,
            "OVERLAP_METHOD": "distance",
            "MINIMUM_DISTANCE": 1,
            "A_MAX": 1,
            "B_MAX": 1,
            "C_MAX": 1000,
            "N": 5,
            "DELTA": 0.2,
            "THRESHOLD": 0.05
        }
        return config


def get_date():
    import time

    date = time.localtime()
    month = str(date[1])
    day = str(date[2])
    hour = str(date[3])
    minutes = str(date[4])
    date = "2024" + "_" + month + "_" + day + "_" + hour + "_" + minutes
    return date


def get_arr_df(arr_current, arr_dices, total_dices, arr_yields, total_yields,
               arr_HDs, total_HDs, arr_costs, total_costs, num_calls):

    return pd.DataFrame(data={"iteration": np.arange(start=1, stop=num_calls + 1),
                              "array": [arr_current] * num_calls,
                              "array_dices": arr_dices,
                              "total_dices": total_dices,
                              "array_yields": arr_yields,
                              "total_yields": total_yields,
                              "array_HD": arr_HDs,
                              "total_HD": total_HDs,
                              "array_cost": arr_costs,
                              "total_cost": total_costs}
                        )


def get_best_df(arr_current, dice, total_dice, prop_total_dice, grid_yield, grid_yield_total,
                hell_d, hell_d_total, prop_total_hd, cost, prop_cost, res):

    return pd.DataFrame(data={"array": arr_current,
                              "best_alpha": res.x[0],
                              "best_beta": res.x[1],
                              "best_offset_from_base": res.x[2],
                              "array_dice": dice,
                              "total_dice": total_dice,
                              "prop_total_dice": prop_total_dice,
                              "array_yield": grid_yield,
                              "total_yield": grid_yield_total,
                              "array_HD": hell_d,
                              "total_HD": hell_d_total,
                              "prop_total_hd": prop_total_hd,
                              "cost": cost,
                              "prop_cost": prop_cost},
                        index=[0]
                        )


def get_best_df_10x10x10(arr_current, dice, total_dice, prop_total_dice, grid_yield, grid_yield_total,
                         hell_d, hell_d_total, prop_total_hd, cost, prop_cost, res):

    return pd.DataFrame(data={"array": arr_current,
                              "best_alpha": res[0],
                              "best_beta": res[1],
                              "best_offset_from_base": res[2],
                              "best_shank_length": res[3],
                              "array_dice": dice,
                              "total_dice": total_dice,
                              "prop_total_dice": prop_total_dice,
                              "array_yield": grid_yield,
                              "total_yield": grid_yield_total,
                              "array_HD": hell_d,
                              "total_HD": hell_d_total,
                              "prop_total_hd": prop_total_hd,
                              "cost": cost,
                              "prop_cost": prop_cost},
                        index=[0]
                        )


def get_empty_df():
    return pd.DataFrame(data={"notes": "no valid array for this hem and sub"})


def check_all_files(results_path):
    """Checks the results directory for empty files
    Unfortunately a lot of files from 5_arrays_10x10x10 had 0 size :(

    Parameters
    ----------
    results_path : str
        One of:
        - "/home/odysseas/Desktop/UU/thesis/BayesianOpt/5_arrays_10x10x10/results/"
        - "/home/odysseas/Desktop/UU/thesis/BayesianOpt/16_arrays_1x10x10/results/"
        - "/home/odysseas/Desktop/UU/thesis/BayesianOpt/fsaverage_5_arrays_10x10x10/results/"

    Returns
    -------
    (all_files, file_sizes, empty_files) : tuple
    """
    all_files = []
    file_sizes = []
    empty_files = []
    for root, dirs, files in os.walk(results_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            all_files.append(file_path)
            file_sizes.append(file_size)
            if file_size == 0:
                empty_files.append((file_path, file_size))

    if len(all_files) == 0:
        print("Couldn't find any files, check the results path")
    else:
        if len(empty_files) > 0:
            print(f"Oops, there are {len(empty_files)} empty files")
        else:
            print(f"All good, no empty files")
    return all_files, file_sizes, empty_files


def get_final_df(n_specified_arrays, n_valid_arrays, total_dice, prop_total_dice, grid_yield_total,
                hell_d_total, prop_total_hd, cost, prop_cost, res):
    """Writes one row with the final results after having optimized all arrays."""

    return pd.DataFrame(data={"n_specified_arrays": n_specified_arrays,
                              "n_valid_arrays": n_valid_arrays,
                              "best_alpha": res.x[0],
                              "best_beta": res.x[1],
                              "best_offset_from_base": res.x[2],
                              "total_dice": total_dice,
                              "prop_total_dice": prop_total_dice,
                              "total_yield": grid_yield_total,
                              "total_HD": hell_d_total,
                              "prop_total_hd": prop_total_hd,
                              "cost": cost,
                              "prop_cost": prop_cost},
                        index=[0]
                        )

def get_final_df_10x10x10(n_specified_arrays, n_valid_arrays, total_dice, prop_total_dice, grid_yield_total,
                hell_d_total, prop_total_hd, cost, prop_cost, res):
    """Writes one row with the final results after having optimized all arrays."""

    return pd.DataFrame(data={"n_specified_arrays": n_specified_arrays,
                              "n_valid_arrays": n_valid_arrays,
                              "best_alpha": res[0],
                              "best_beta": res[1],
                              "best_offset_from_base": res[2],
                              "best_shank_length": res[3],
                              "total_dice": total_dice,
                              "prop_total_dice": prop_total_dice,
                              "total_yield": grid_yield_total,
                              "total_HD": hell_d_total,
                              "prop_total_hd": prop_total_hd,
                              "cost": cost,
                              "prop_cost": prop_cost},
                        index=[0]
                        )


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


def write_results_pickle(results_path, s, hem, data):
    """Saves optimized arrays and maps in a pickled list

    Parameters
    ----------
    results_path : str
        The output path
    s : str
        The subject number
    hem : str
        The subject's hemisphere
    data : list
        [optimized_arrays_from_f_manual, phosphenes_per_arr, phosphene_map_per_arr,
         total_contacts_xyz_moved, all_phosphenes, total_phosphene_map]
    """
    import pickle

    sub_hem_path = results_path + s + "/" + hem + "/"
    filename = sub_hem_path + s + "_" + hem + "_" + get_date() + ".pkl"
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=-1)


def write_results(df, out_path, sub, hem, results_type):
    sub_hem_path = out_path + sub + "/" + hem + "/"
    filename = sub_hem_path + sub + "_" + hem + "_" + results_type + "_" + get_date()
    df.to_csv(filename + ".csv")


def write_params(out_path, sub, hem, config):
    sub_hem_path = out_path + sub + "/" + hem + "/"
    filename = sub + "_" + hem + "_" + get_date()
    params = {"n_calls": config["NUM_CALLS"],
              "dice_weight_a": config["A"],
              "yield_weight_b": config["B"],
              "HD_weight_c": config["C"],
              "n_arrays": config["N_ARRAYS"],
              "n_contactpoints_shank": config["N_CONTACTPOINTS_SHANK"],
              "n_combs": config["N_COMBS"],
              "n_shanks_per_comb": config["N_SHANKS_PER_COMB"],
              "spacing_along_xy": config["SPACING_ALONG_XY"],
              "overlap_method": config["OVERLAP_METHOD"],
              "minimum_distance": config["MINIMUM_DISTANCE"],
              "hemisphere": hem,
              "brain_area": "V1",
              "num_initial_points": config["NUM_INITIAL_POINTS"],
              "dc_percentile": config["DC_PERCENTILE"],
              "windowsize": config["WINDOWSIZE"],
              "cort_mag_model": config["CORT_MAG_MODEL"],
              "view_angle": config["VIEW_ANGLE"],
              "amp": config["AMP"]
              }

    # Convert and write JSON object to file
    with open(sub_hem_path + filename + "_params.json", "w") as outfile:
        json.dump(params, outfile)


def read_pickle_file(sub: str, hem: str) -> list:
    """Reads the saved pickle file.

    Parameters
    ----------
    hem : str
        The hemisphere from which we want to get the pickle file

    Returns
    -------
    data : list
        [optimized_arrays_from_f_manual, phosphenes_per_arr, phosphene_map_per_arr,
        total_contacts_xyz_moved, all_phosphenes, total_phosphene_map]
    """
    import glob, pickle
    RESULTS_PATH = "/home/odysseas/Desktop/UU/thesis/BayesianOpt/fsaverage_5_arrays_10x10x10/results/"
    dir = RESULTS_PATH + sub + "/" + hem + "/"
    filenames = glob.glob(os.path.join(dir, "*.pkl"))
    data = []
    if filenames:
        # Assuming there's only one file in the directory, you can take the first one
        filename = filenames[0]
        try:
            with open(filename, "rb") as file:
                data = pickle.load(file)
        except FileNotFoundError as e:
            print(e)
        return data


def get_density(phosphenes, phos_or_dens="dens", window_size=1000, ph_size_scale=3):
    from src.ninimplant import pol2cart
    from src.phos_elect import make_gaussian_v1
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


def get_min_distance(arr1: np.ndarray, arr2: np.ndarray):
    """Gets the minimum distance between two arrays.
    This function is used within get_overlap_validity() to determine
    whether the array configuration overlaps with an already existing array.

    Parameters
    ----------
    arr1 : np.ndarray
    arr2 : np.ndarray
    Returns
    ----------
    np.float
        The minimum distance between the two arrays.
    """
    arr1 = arr1.T if arr1.shape[1] != 3 else arr1
    arr2 = arr2.T if arr2.shape[1] != 3 else arr2
    distances = cdist(arr1, arr2)
    return np.min(distances)


def custom_stopper(res, n=5, delta=0.2, thresh=0.05):
    """Returns True (stops the optimization) when the difference between
    best and worst of the best N are below delta AND the best is below thresh

    N = last number of cost values to track
    delta = ratio best and worst
    """
    if len(res.func_vals) >= n:
        func_vals = np.sort(res.func_vals)
        worst = func_vals[n - 1]
        best = func_vals[0]
        return (abs((best - worst) / worst) < delta) & (best < thresh)
    else:
        return None


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
