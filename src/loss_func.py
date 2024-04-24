import numpy as np
import sys

from scipy.special import rel_entr


def hellinger_distance(p, q):
    """
    Computes the Hellinger distance between two probability distributions.
    
    Args:
        p (array-like): A probability distribution.
        q (array-like): Another probability distribution.
        
    Returns:
        float: The Hellinger distance between p and q.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
        
    # replace zeros with smallest value
    a[a == 0] = sys.float_info.min
    b[b == 0] = sys.float_info.min
    
    x = rel_entr(a, b)

    return np.abs(np.sum(x))


def get_yield(contacts_xyz, good_coords, empty_score=1.0):
    """
    Computes the yield between the arrays and the gray tissue

    Parameters
    ----------
    contacts_xyz : np.ndarray
        Electrode grid for which we calculate the yield
    good_coords : np.ndarray
        All GM voxels which contain pRF data

    Returns
    -------
    contacts_yield : float
    """
    # filter good coords
    b1 = np.round(np.transpose(np.array(contacts_xyz)))
    b2 = np.transpose(np.array(good_coords))
    indices_prf = []
    for i in range(b1.shape[0]):
        tmp = np.where(np.array(b2 == b1[i, :]).all(axis=1))
        if tmp[0].shape[0] != 0:
            indices_prf.append(tmp[0][0])

    contact_hits = len(indices_prf)
    total_points = contacts_xyz.shape[1]

    contacts_yield = (contact_hits / total_points)

    return contacts_yield


def DC(im1, im2, bin_thresh, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    
    im1 = (im1 > bin_thresh).astype(bool)
    im2 = (im2 > bin_thresh).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score, None, None

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() / im_sum), im1, im2
