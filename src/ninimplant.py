import mathutils
import numpy as np
from math import radians
from PIL import Image, ImageDraw


def cart2pol(x, y):
    """Converts cartesian to polar coordinates.

    Parameters
    ----------
    x : np.ndarray | float
    y : np.ndarray | float

    Returns
    -------
    rho : np.ndarray | float
        This is the eccentricity
    y : np.ndarray | float
        This is the theta or the angle
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(theta, rho):
    """Converts polar to cartesian coordinates.

    Parameters
    ----------
    theta : np.ndarray | float
        This is the angle in radians

    rho : np.ndarray | float
        This is the eccentricity

    Returns
    -------
    x : np.ndarray | float
    y : np.ndarray | float
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def get_xyz(data, under_sampling=1):
    """Gets the xyz coordinates specified by under sampling"""
    if under_sampling == 1:
        x = data[0]
        y = data[1]
        z = data[2]
      
    else:
        x = data[0][::under_sampling]
        y = data[1][::under_sampling]
        z = data[2][::under_sampling]
        
    return x, y, z


def transform(points,
              TRANSLATION_X=0, TRANSLATION_Y=0, TRANSLATION_Z=0,
              ROTATION_ANGLE_X=0, ROTATION_ANGLE_Y=0, ROTATION_ANGLE_Z=0):
    """
    The input is a matrix like this:
    (n_points, 3 + 1)
      [[x1, x2]
       [y1, y2]
       [z1, z2]
       [1,  1]]
    
    The output is a matrix like this:
      [[x1', x2']
       [y1', y2']
       [z1', z2']]
       [1,   1]]
    """
    mat_rot_x = mathutils.Matrix.Rotation(radians(ROTATION_ANGLE_X), 4, 'X')
    mat_rot_y = mathutils.Matrix.Rotation(radians(ROTATION_ANGLE_Y), 4, 'Y')
    mat_rot_z = mathutils.Matrix.Rotation(radians(ROTATION_ANGLE_Z), 4, 'Z')
    
    mat_trans_x = mathutils.Matrix.Translation(mathutils.Vector((TRANSLATION_X, 0, 0)))
    mat_trans_y = mathutils.Matrix.Translation(mathutils.Vector((0, TRANSLATION_Y, 0)))
    mat_trans_z = mathutils.Matrix.Translation(mathutils.Vector((0, 0, TRANSLATION_Z)))
    
    trans_x = np.array(mat_trans_x)
    trans_y = np.array(mat_trans_y)
    trans_z = np.array(mat_trans_z)
    
    rot_x = np.array(mat_rot_x)
    rot_y = np.array(mat_rot_y)
    rot_z = np.array(mat_rot_z)

    # Join transformation matrices
    # print('Warning: in function transform, we are first rotating, then translating')
    transform = rot_x @ rot_y @ rot_z @ trans_x @ trans_y @ trans_z

    # Apply transformations and return
    return (transform @ points).astype('float32')


def create_cube(x_base, y_base, height, xmax, ymax, zmax):
    """Creates a cube given the specified parameters"""
    
    mask_result = np.zeros((xmax, ymax, zmax))
    img = Image.new('L', (xmax, ymax), 0)
    base = [(0, 0), (y_base, x_base)]

    for i in range(height):
        ImageDraw.Draw(img).rectangle(base, outline=1, fill=1)
        mask = np.array(img)
        mask_result[:,:,i] = mask
    
    coords = np.where(mask_result)
    points = np.array([coords[0], coords[1], coords[2]]).T.astype('float32')
    aux_ones = np.ones((points.shape[0], 1)).astype('float32')
    points = np.hstack((points, aux_ones))

    return points.T, mask_result.astype(int)


def cube_from_points(x, y, z):
    """Creates a cube given x, y, z points"""
    points = np.array([x, y, z]).T.astype('float32')
    aux_ones = np.ones((x.shape[0], 1)).astype('float32')
    points = np.hstack((points, aux_ones))

    return points.T


def get_translation(cube_center, desired_center):
    """Returns TRANSLATION_X, TRANSLATION_Y, TRANSLATION_Z given a desired cube center"""
    return desired_center - cube_center[0:3]


def translate_cube(cube_center, desired_center, rotation_angles, points):
    """Used to create the electrode grid

    Parameters
    ----------
    cube_center : np.ndarray
        x, y, z coordinates of the comb center
    desired_center : list
        Three elements representing the desired center of the
        electrode grid (x, y, z), which is the center of the caclarine sulcus
    rotation_angles : tuple
        The rotation angles for each dimension (x, y, z)
    points : np.ndarray
        The electrode points on the comb (4, N) where the rows represent x, y, z, 1

    Returns
    -------
    x, y, z of translated cube
    """
    needed_translation = desired_center - cube_center[0:3]
    ROTATION_ANGLE_X, ROTATION_ANGLE_Y, ROTATION_ANGLE_Z = rotation_angles
    TRANSLATION_X, TRANSLATION_Y, TRANSLATION_Z = get_translation(cube_center, desired_center)
    
    # transform = trans @ rot
    new_cube = transform(points,
                  TRANSLATION_X,TRANSLATION_Y,TRANSLATION_Z,
                  ROTATION_ANGLE_X,ROTATION_ANGLE_Y,ROTATION_ANGLE_Z)
    x_new_cube, y_new_cube, z_new_cube = get_xyz(new_cube)
    
    return x_new_cube, y_new_cube, z_new_cube