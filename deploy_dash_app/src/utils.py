import pandas as pd
import os


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