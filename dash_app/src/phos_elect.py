'''
Created on Wed Jul 13 08:20:28 2021

@author: van Hoof & Lozano


phos_elect.create_grid
phos_elect.reposition_grid
phos_elect.implant_grid
phos_elect.get_phosphenes
phos_elect.prf_to_phos
phos_elect.normalized_uv

'''

import numpy as np
import math
from math import radians
import trimesh  # needed for convex hull

### needed for matrix rotation/translation ect
from src.ninimplant import pol2cart, get_xyz, transform, cube_from_points, translate_cube


def normalized_uv(a, axis=-1, order=2):
    """Used in the grid implantation procedure implant_grid()"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def create_grid(desired_center, shank_length=9, n_contactpoints_shank=10,
                n_combs=1, n_shanks_per_comb=10, spacing_along_xy=1, offset_from_origin=0):
    """Creates the electrode array.
    The total number of contact points = n_contactpoints_shank * n_shanks_per_comb * n_combs

    Modified version of the original create_grid by Rick.
    - Removed unused variables like base_width, etc.
    - Changed n_combs and n_shanks_per_comb to different values than n_contactpoints_shank
    - Removed the part with contacts_position_length because it wasn't used
    - Removed s part that used ELECTRODE_ABS_ANGLE because it had no effect since it was 0

    Parameters
    ----------
    desired_center : list (global)
        A list of three numbers indicating the start location in the brain in 3d space,
        which is the center of the calcarine suclus of each hemisphere and is a global variable
    shank_length : int
        The length of the electrode shank in mm along grid z-axis
    n_contactpoints_shank : int (global)
        The number of contact points per shank, which is a global variable
    n_combs : int (global)
        The number of combs. One comb has several shanks of several contact points
    n_shanks_per_comb : int (global)
        The number of shanks per comb
    spacing_along_xy : int or float (global)
    offset_from_origin : int or float
        The offset from the middle contact points to the base origin in mm

    Returns
    -------
    np.ndarray
        An array of (3, x) representing the coordinates of the x contact points in 3d space
    """
    HEIGHT_DIFFERENCE = 0  # mm base pitch/offset
    spacing_between_combs = spacing_along_xy  # mm along y-axis = HORIZONTAL DIFFERENCE

    # dimensions v1/v2 80 x 100 x 60 mm
    BRAIN_ANGLE = 0  # angle of the brain with respect to the horizontal line

    # Creating vectors with the position of the contacts relative to the shank origin
    # This is one shank, and the contacts_perShank contact points are evenly spaced
    # from: offset_from_origin (e.g. 0) to: shank_length (e.g. 12)
    # the space between them is shank_length/(contacts_perShank - 1)
    contacts_position_length_singleShank = np.linspace(offset_from_origin, shank_length, num=n_contactpoints_shank)

    # *** First, define the shanks vertically:
    shankList = []

    # for each shank
    for sh in range(n_shanks_per_comb):
        # initialize 3D coordinates [x, y, z]
        coords_shank = np.zeros((n_contactpoints_shank, 3))
        # fill z axis
        for contact in range(n_contactpoints_shank):
            coords_shank[contact, 2] = contacts_position_length_singleShank[contact]
        shankList.append(coords_shank)

    # *** Second, translate each shank so they match the inter shank spacing
    shank_spacing = spacing_along_xy

    for i in range(n_shanks_per_comb):
        shankList[i][:, 0] += i * shank_spacing

    # In addition, we will add some OFFSET FROM ORIGIN
    comb_horizontal_angle = math.radians(BRAIN_ANGLE)  # 0 since BRAIN_ANGLE = 0
    tan = math.tan(comb_horizontal_angle)  # 0 since comb_horizontal_angle = 0

    shankOriginList = []
    for i in range(n_shanks_per_comb):
        hor_distance_origin = i * shank_spacing
        vert_distance_origin = hor_distance_origin * tan  # 0 since tan = 0
        shankOrigin = np.array([i * shank_spacing, 0, vert_distance_origin])

        shankList[i][:, 2] += vert_distance_origin + offset_from_origin  # 0 since vert.. and offset.. are 0
        shankOriginList.append(shankOrigin)

    # Now we have a comb of electrodes ready to translate and rotate
    comb = np.asarray(shankList).reshape(n_shanks_per_comb * n_contactpoints_shank, 3)
    comb = np.hstack((comb, np.ones((comb.shape[0], 1)).astype('float32'))).T

    # First, center our original comb at map center/desired location
    rotation_angles = (0, 0, 0)
    comb_center = np.mean(comb, axis=1)
    x_new_comb, y_new_comb, z_new_comb = translate_cube(comb_center,
                                                        desired_center,  # [150, 150, 60], #[0,0,0],
                                                        rotation_angles,
                                                        comb)
    comb = cube_from_points(x_new_comb, y_new_comb, z_new_comb)
    contacts_xyz = np.asarray(comb)[0:3, :]

    # Now, copy and translate comb
    for i in range(n_combs - 1):
        # Translating to obtain a parallel COMB
        rotation_angles = (0, 0, 0)
        x_new_comb, y_new_comb, z_new_comb = translate_cube(comb_center,
                                                            comb_center[0:3] +
                                                            [0, spacing_between_combs, HEIGHT_DIFFERENCE],
                                                            rotation_angles,
                                                            comb)
        comb = cube_from_points(x_new_comb, y_new_comb, z_new_comb)
        contacts_xyz = np.hstack((contacts_xyz, np.asarray(comb)[0:3, :]))

    return contacts_xyz


def reposition_grid(orig_grid, new_location=None, new_angle=None):
    """Moves the reference line according to new angle.
    Used within implant_grid()

    Parameters
    ----------
    orig_grid : np.ndarray
        The array returned by create_grid, with shape (3, x) of x contact points in 3d space
    new_location : list
        A list of three numbers indicating the location in the brain in 3d space
    new_angle : tuple
        A tuple of three values used to calculate the reference line based on the new angle.
        It's rotating only around x and y angles (alpha and beta, or pitch and yaw), the other one, z, is always 0.

    Returns
    -------
    np.ndarray
        An array of (3, x) representing the coordinates of the x contact points in 3d space
        now after having moved the reference line according to the new angle."""

    points = np.vstack((orig_grid, np.ones((1, orig_grid.shape[1]))))
    x_c, y_c, z_c = np.mean(orig_grid, axis=1)

    # ROTATE #
    # translation - before rotating, first move grid to rotation axis/location
    points = transform(points, -x_c, -y_c, -z_c, 0, 0, 0)
    # rotation
    points = transform(points, 0, 0, 0, new_angle[0], new_angle[1], new_angle[2])
    # translation - move back to desired_center
    points = transform(points, x_c, y_c, z_c, 0, 0, 0)

    # TRANSLATE #
    # to center
    points = transform(points, -x_c, -y_c, -z_c, 0, 0, 0)

    # to new location
    points = transform(points, new_location[0], new_location[1], new_location[2], 0, 0, 0)

    x_new_cube, y_new_cube, z_new_cube = get_xyz(points)
    contacts_xyz = np.asarray([x_new_cube, y_new_cube, z_new_cube])

    return contacts_xyz


def implant_grid(gm_mask, orig_grid, start_location, new_angle, offset_from_base):
    """Determines the insertion point of the center of the electrode grid, based on angle and target point.

    Parameters
    ----------
    gm_mask : np.ndarray (global)
        The grey matter estimated from the mri images
    orig_grid : np.ndarray
        The electrode grid as returned from create_grid()
    start_location : list (global)
        A list of three numbers indicating the start location in the brain in 3d space,
        which is the center of the calcarine suclus of each hemisphere and is a global variable
    new_angle : tuple
        A tuple of three values used to calculate the reference line based on the new angle.
        It's rotating only around x and y angles (alpha and beta, or pitch and yaw), the other one, z, is always 0.
    offset_from_base : dim3 (global)
        The third parameter to be optimized

    Returns
    -------
    ref_contacts_xyz : np.ndarray -> not used

    contacts_xyz_moved : np.ndarray

    refline, refline_moved, projection, ref_orig, ray_visualize, new_location -> not used

    grid_valid : bool
        Whether the resulting grid is valid (invalid if it has points outside the convex hull)

    """
    # start location
    ref_orig = np.array([start_location[0], start_location[1], start_location[2]])  # ref-line vector
    ref_orig_targ = np.array([start_location[0], start_location[1], 0.0])  # ref-line vector

    # move reference line according to new_angle             
    refline = np.transpose(np.vstack((ref_orig, ref_orig_targ)))
    refline_moved = reposition_grid(refline, start_location, new_angle)

    # find direction between ref start and endpoint
    ab = refline_moved[:, 1] - refline_moved[:, 0]
    ref_direction = normalized_uv(ab)

    # convert greymatter to pointcloud and compute convex hull
    mesh = trimesh.points.PointCloud(gm_mask)
    mesh = mesh.convex_hull

    # create unit vector that describes direction along z-axis of the electrode grid
    ray_origins = np.array([[start_location[0], start_location[1], start_location[2]],
                            [start_location[0], start_location[1], start_location[2]]])
    ray_directions = np.array([[ref_direction[0][0], ref_direction[0][1], ref_direction[0][2]],
                               [ref_direction[0][0], ref_direction[0][1], ref_direction[0][2]]])
    ray_visualize = trimesh.load_path(np.hstack((ray_origins, ray_origins + 30 * ray_directions)).reshape(-1, 2, 3))

    # compute base offset
    zdist_center_to_base = ((np.max(orig_grid[2, :]) - np.min(orig_grid[2, :])) / 2) + offset_from_base
    base_offset = [ref_direction[0][0] * zdist_center_to_base, ref_direction[0][1] * zdist_center_to_base,
                   ref_direction[0][2] * zdist_center_to_base]

    # compute intersection direction unit and convex hull
    locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                   ray_directions=ray_directions)
    projection = locations[0]

    # position grid at point where line vector touches convex hull and move center of grid into the brain by zdist_center_to_base        
    new_location = [projection[0] - base_offset[0], projection[1] - base_offset[1], projection[2] - base_offset[2]]

    # move original electrode grid according to new_angle
    ref_contacts_xyz = reposition_grid(orig_grid, start_location, new_angle)

    # move electrode grid according to new_angle
    contacts_xyz_moved = reposition_grid(orig_grid, new_location, new_angle)

    # check whether grid contains points outside of convex hull    
    if np.sum(mesh.contains(contacts_xyz_moved.T)) < contacts_xyz_moved.shape[1]:
        grid_valid = False
    else:
        grid_valid = True

    return ref_contacts_xyz, contacts_xyz_moved, refline, refline_moved, projection, ref_orig, ray_visualize, new_location, grid_valid


def get_phosphenes(contacts_xyz, good_coords, polar_map, ecc_map, sigma_map):
    """Creates the generated phosphenes from the given contact points.

    Parameters
    ----------
    contacts_xyz : np.ndarray
        The contact points after calling implant_grid()
    good_coords : np.ndarray
        The brain coordiantes of the specified regions (here v1)
    polar_map : np.ndarray
        Defined by the brain scan files "inferred_angle.mgz"
    ecc_map : np.ndarray
        Defined by the brain scan files "inferred_ecc.mgz"
    sigma_map : np.ndarray
        Defined by the brain scan files "inferred_sigma.mgz"

    Returns
    -------
    phosphenes : np.ndarray
        The generated phosphenes of shape(N, 3), where 3 is angle, ecc, rf size
    """

    # filter good coords
    b1 = np.round(np.transpose(np.array(contacts_xyz)))
    b2 = np.transpose(np.array(good_coords))
    indices_prf = []
    for i in range(b1.shape[0]):
        tmp = np.where(np.array(b2 == b1[i, :]).all(axis=1))
        if tmp[0].shape[0] != 0:
            indices_prf.append(tmp[0][0])  # len(indices_prf) is the n of functional electrodes

    s_list, p_list, e_list = [], [], []
    for i in indices_prf:
        xp, yp, zp = good_coords[0][i], good_coords[1][i], good_coords[2][i]
        pol = polar_map[xp, yp, zp]
        ecc = ecc_map[xp, yp, zp]
        sigma = sigma_map[xp, yp, zp]

        p_list.append(pol)
        e_list.append(ecc)
        s_list.append(sigma)

    # normalize to range(0, 2*pi)
    eccentricities = np.asarray(e_list)
    polar_angles = np.asarray(p_list)
    rf_sizes = np.asarray(s_list)

    # angle x ecc x rfsize
    phosphenes = np.vstack((polar_angles, eccentricities))
    phosphenes = np.vstack((phosphenes, rf_sizes))
    phosphenes = phosphenes.T

    return phosphenes


def get_cortical_magnification(ecc, mapping_model='wedge-dipole'):
    """Gets the cortical magnification given an eccentricity and a mapping model.
    The parameters a, b, and k are specified in Horten & Hoyt (1991)."""
    a = 0.75
    b = 120
    k = 17.3
    if mapping_model == 'monopole':
        return k / (ecc + a)
    if mapping_model in ['dipole', 'wedge-dipole']:
        return k * (1 / (ecc + a) - 1 / (ecc + b))
    raise NotImplementedError


def cortical_spread(amplitude):
    """Returns the radius of the cortical spread in mm given a stimulation amplitude.
    The stimulation amplitude is defined in the config as 100.
    The formula is given from Tehovnik et al. 2007"""
    return np.sqrt(amplitude / 675)


def make_gaussian(sigma, center=None):
    """Makes a square gaussian kernel, given the specified scale (sigma).

    Parameters
    ----------
    sigma : int or float
        The scale factor of the gaussian
    center : list or np.ndarray
        The center of the phosphene

    Returns
    -------
    The square gaussian kernel
    """
    size = sigma * 5
    x = np.arange(0, size, 1, 'float32')
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def make_gaussian_v1(size, fwhm=3, center=None):
    """
    Makes a square gaussian kernel.

    Parameters
    ----------
    size : int or float
        The length of one side of the square
    fwhm : int or float
        The full-width-half-maximum, which
        can be thought of as an effective radius.
    center : list or np.ndarray
        The center of the phosphene

    Returns
    -------
    The square gaussian kernel
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def prf_to_phos(phosphene_map, phosphenes, view_angle=90, ph_size_scale=1):
    """Creates a phosphene map (a 2d image) given the phosphenes generated by get_phosphenes().
    The phosphene map is placed in an initial empty 2d square numpy array.

    Parameters
    ----------
    phosphene_map : np.ndarray
        A square array of 0's of arbitrary size (default 1000x1000)
    phosphenes : np.ndarray
        The phosphenes generated by get_phosphenes(), with shape (N, 3) where 3 is angle, ecc, rf size
    view_angle : int
        The degree of eccentricity, where 90deg means 90% of window size.
        Scales the eccentricity and size of phosphenes in degrees to the window size of the phosphene map in pixels.
    ph_size_scale : int
        Scales the phosphenes for easier visualization

    Returns
    -------
    phosphene_map : np.ndarray
        A square array showing the phosphene map in a 2d image.
    """
    # windowsize is defined as the diameter of the plotting window in pixels
    window_size = phosphene_map.shape[1]

    # eccentricity is a radius value assuming (0,0) to be in the center
    # degrees to pixels
    scaled_ecc = (window_size / 2) / view_angle  # pixels per degree of visual angle

    for i in range(0, phosphenes.shape[0]):
        s = int(phosphenes[i, 2] * scaled_ecc)  # phosphene size in pixels
        c_x, c_y = pol2cart(radians(phosphenes[i, 0]), phosphenes[i, 1])
        x = int(c_x * scaled_ecc + window_size / 2)
        y = int(c_y * scaled_ecc + window_size / 2)

        if s < 2:
            s = 2  # print('Tiny phosphene: artificially making size == 2')

        elif (s % 2) != 0:
            s = s + 1
        else:
            None

        g = make_gaussian(sigma=s, center=None)
        g /= g.max()
        half_gauss = g.shape[0] // 2

        try:
            phosphene_map[y - half_gauss:y + half_gauss, x - half_gauss:x + half_gauss] += g
        except:
            None  # print('error... (probably a phosphene on the edge of the image')

    # rotate by 90 degrees to match orientation visual field
    phosphene_map = np.rot90(phosphene_map, 1)
    return phosphene_map
