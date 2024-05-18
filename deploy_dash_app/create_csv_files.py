import pandas as pd
import numpy as np
import nibabel as nib
import shutil
import pickle
import os
import glob

from src.utils import create_df_from_opt_dict, create_dirs

EXP1_RESULTS_PATH = "/home/odysseas/Desktop/UU/thesis/MSc_thesis/5_arrays_10x10x10/results/"
EXP2_RESULTS_PATH = "/home/odysseas/Desktop/UU/thesis/MSc_thesis/fsaverage_5_arrays_10x10x10/results/"
EXP3_RESULTS_PATH = "/home/odysseas/Desktop/UU/thesis/MSc_thesis/16_arrays_1x10x10/results/"

SUBS = os.listdir(EXP2_RESULTS_PATH)[:20]


def get_experiment(res_path: str):
    if "5_arrays_10x10x10" in res_path and not "fsaverage" in res_path:
        exp = "experiment1"
    elif "fsaverage" in res_path:
        exp = "experiment2"
    elif "16_arrays_1x10x10" in res_path:
        exp = "experiment3"
    else:
        print(f"res_path is wrong: {res_path}")
        return None
    return exp


def read_pickle_and_create_arrays_csv(res_path: str, sub: str, hem: str):
    """Reads the saved pickle file and creates a csv with the x, y, z
    coordinates of the optimized arrays for the given sub and hemisphere.

    Parameters
    ----------
    res_path : str
        The path with the saved results for each experiment
    sub: str
        The subject from which we want to read the pickle file.
    hem : str
        The hemisphere from which we want to read the pickle file
    """
    exp = get_experiment(res_path)
    pickle_dir = res_path + sub + "/" + hem + "/"
    create_dirs("assets/", sub, hem + f"/{exp}")
    filenames = glob.glob(os.path.join(pickle_dir, "*.pkl"))
    data = []
    if filenames:
        # Assuming there's only one file in the directory, you can take the first one
        filename = filenames[0]
        try:
            with open(filename, "rb") as file:
                data = pickle.load(file)
        except FileNotFoundError as e:
            print(e)
        optimized_arrays_from_f_manual = data[0]
        arrays_df = create_df_from_opt_dict(optimized_arrays_from_f_manual)
        arrays_df.to_csv(f"./assets/{sub}/{hem}/{exp}/arrays.csv", index=False)
        print(f"Created arrays csv for {sub}/{hem}/{exp}")


def read_scans_and_create_brains_csv(sub: str):
    data_dir = f"/home/odysseas/Desktop/UU/thesis/BayesianOpt/input_processed_data_HCP/{sub}/T1w/mri/"
    ang_img = nib.load(data_dir + "inferred_angle.mgz")
    polar_map = ang_img.get_fdata()
    ecc_img = nib.load(data_dir + "inferred_eccen.mgz")
    ecc_map = ecc_img.get_fdata()
    aparc_img = nib.load(data_dir + "aparc+aseg.mgz")
    aparc_roi = aparc_img.get_fdata()

    # compute valid voxels
    dot = (ecc_map * polar_map)
    good_coords = np.asarray(np.where(dot != 0.0))

    gm_coords_rh = np.vstack(np.where((aparc_roi >= 1000) & (aparc_roi < 2000)))
    gm_coords_lh = np.vstack(np.where(aparc_roi > 2000))

    set_rounded_good_coords = set(map(tuple, good_coords.T))
    set_rounded_gm_coords_rh = set(map(tuple, gm_coords_rh.T))
    set_rounded_gm_coords_lh = set(map(tuple, gm_coords_lh.T))
    good_coords_lh = np.array(list(set(set_rounded_good_coords) & set(set_rounded_gm_coords_lh))).T
    good_coords_rh = np.array(list(set(set_rounded_good_coords) & set(set_rounded_gm_coords_rh))).T

    lh_brain_df = pd.DataFrame(data={"x": good_coords_lh[0, :],
                                     "y": good_coords_lh[1, :],
                                     "z": good_coords_lh[2, :]})

    rh_brain_df = pd.DataFrame(data={"x": good_coords_rh[0, :],
                                     "y": good_coords_rh[1, :],
                                     "z": good_coords_rh[2, :]})

    ###### V1, in case I need them ######
    label_img = nib.load(data_dir + "inferred_varea.mgz")
    label_map = label_img.get_fdata()
    v1_coords_rh = np.asarray(np.where(label_map == 1))
    v1_coords_lh = np.asarray(np.where(label_map == 1))
    set_rounded_v1_coords_lh = set(map(tuple, v1_coords_lh.T))
    set_rounded_v1_coords_rh = set(map(tuple, v1_coords_rh.T))
    v1_coords_lh = np.array(list(set(set_rounded_v1_coords_lh) & set(set_rounded_gm_coords_lh))).T
    v1_coords_rh = np.array(list(set(set_rounded_v1_coords_rh) & set(set_rounded_gm_coords_rh))).T

    lh_v1_df = pd.DataFrame(data={"x": v1_coords_lh[0, :],
                                  "y": v1_coords_lh[1, :],
                                  "z": v1_coords_lh[2, :]})

    rh_v1_df = pd.DataFrame(data={"x": v1_coords_rh[0, :],
                                  "y": v1_coords_rh[1, :],
                                  "z": v1_coords_rh[2, :]})

    for hem, brain_df, v1_df in zip(["LH", "RH"], [lh_brain_df, rh_brain_df], [lh_v1_df, rh_v1_df]):
        create_dirs("assets/", sub, hem)
        merged_df = pd.merge(brain_df, v1_df, on=["x", "y", "z"], how="left", indicator=True)
        merged_df["v1"] = (merged_df["_merge"] == "both").astype(int)
        merged_df.drop("_merge", axis=1, inplace=True)

        merged_df.to_csv(f"./assets/{sub}/{hem}/brain_coords.csv", index=False)
        print(f"Created brain coords csv for {sub}/{hem}")


def read_and_save_phosphenes_per_array(res_path: str, sub: str, hem: str):
    """Aiming to read the phosphenes per array (dict),
    which is the second element in the pickled list.
    The shape of the phosphenes for a given array is (x, 3),
    where x is the number of phosphenes for that array.

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
    exp = get_experiment(res_path)

    pickle_dir = res_path + sub + "/" + hem + "/"
    create_dirs("assets/", sub, hem + f"/{exp}")
    filenames = glob.glob(os.path.join(pickle_dir, "*.pkl"))
    data = []
    if filenames:
        # Assuming there's only one file in the directory, you can take the first one
        filename = filenames[0]
        try:
            with open(filename, "rb") as file:
                data = pickle.load(file)
        except FileNotFoundError as e:
            print(e)
        phosphenes_per_arr = data[1]
        for arr, phos in phosphenes_per_arr.items():
            arr_phosphenes_df = pd.DataFrame(data={"x": phos.T[0, :],
                                                   "y": phos.T[1, :],
                                                   "z": phos.T[2, :]})
            arr_phosphenes_df.to_csv(f"./assets/{sub}/{hem}/{exp}/arr{arr}_phosphenes.csv", index=False)
            print(f"Saved phospheness for {sub}/{hem}/{exp} and array {arr}")


def copy_density_picture(res_path: str, sub: str, hem: str):
    exp = get_experiment(res_path)
    picture_path = res_path + sub + "/" + hem + "/"
    create_dirs("assets/", sub, hem + f"/{exp}")
    filenames = glob.glob(os.path.join(picture_path, "*.png"))
    if filenames:
        try:
            src_density_filename = [filename for filename in filenames if "H_Gaussian_KDE" in filename][0]
            dest_path = f"./assets/{sub}/{hem}/{exp}/density_phosphenes.png"
            shutil.copyfile(src_density_filename, dest_path)
            print(f"Saved density picture for {sub}/{hem}/{exp}")

        except Exception as e:
            print(e)


def copy_bin_map_picture(res_path: str, sub: str, hem: str):
    exp = get_experiment(res_path)
    picture_path = res_path + sub + "/" + hem + "/"
    create_dirs("assets/", sub, hem + f"/{exp}")
    filenames = glob.glob(os.path.join(picture_path, "*.png"))
    if filenames:
        try:
            src_density_filename = [filename for filename in filenames if "H_phosphene_maps" in filename][0]
            dest_path = f"./assets/{sub}/{hem}/{exp}/binary_maps.png"
            shutil.copyfile(src_density_filename, dest_path)
            print(f"Copied binary maps picture for {sub}/{hem}/{exp}")

        except Exception as e:
            print(e)



def main():
    for sub in SUBS:
        read_scans_and_create_brains_csv(sub)
        for hem in ["LH", "RH"]:
            for exp in [EXP1_RESULTS_PATH, EXP2_RESULTS_PATH, EXP3_RESULTS_PATH]:
                read_pickle_and_create_arrays_csv(exp, sub, hem)
                read_and_save_phosphenes_per_array(exp, sub, hem)
                copy_density_picture(exp, sub, hem)
                copy_bin_map_picture(res_path=exp, sub=sub, hem=hem)


if __name__ == "__main__":
    main()
