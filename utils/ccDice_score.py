# credits to: Rougé, P., Merveille, O., & Passat, N. (2024). ccDice: A topology-aware Dice score based on connected components. International Workshop on Topology-and Graph-Informed Imaging Informatics (pp. 11–21). Springer.
# https://github.com/PierreRouge/ccDice

# modified to support Numpy 2.0.1   (only works with binary labels)

import numpy as np
from scipy import ndimage
from skimage.measure import label
from scipy import ndimage as ndi


#def ccDice(y_pred, y_true, alpha=0.5):
def ccDice(y_pred_label, cc_pred, y_true_label, cc_true, alpha=0.5):    # modified (bug occurred with default)
    
    #y_pred_label, cc_pred = label(y_pred, return_num=True)
    #y_true_label, cc_true = label(y_true, return_num=True)
    
    y_true_label[y_true_label != 0] = y_true_label[y_true_label != 0] + cc_pred

    list_s = []
    indices_cc = []
    for a in range(1, cc_pred + 1):
        for b in range(cc_pred + 1, cc_pred + cc_true + 1):
            
            y1 = np.zeros(y_pred_label.shape)
            y1[y_pred_label == a] = 1
            
            y2 = np.zeros(y_true_label.shape)
            y2[y_true_label == b] = 1
            
            s_ab = S(y1, y2)
            s_ba = S(y2, y1)
            
            list_s.append(s_ab)
            list_s.append(s_ba)
            
            indices_cc.append((a, b))
            indices_cc.append((b, a))
        
    if alpha <= 0.5:
        # Sort the list
        list_s = np.array(list_s)
        indices = np.argsort(-list_s)
        indices_cc = np.array(indices_cc)
        
        list_s = np.array(list_s)
        list_s = list_s[indices]
        indices_cc = indices_cc[indices]
    
    left_list = []
    right_list = []
    tp = 0
    i = 0
    s = list_s[0]
    coor = indices_cc[0]
    while s >= alpha and i < len(list_s):
        
        if (coor[0] not in left_list) and (coor[1] not in right_list):
        
            left_list.append(coor[0])
            right_list.append(coor[1])
            tp += 1
            
        i += 1
        if i < len(list_s):
            s = list_s[i]
            coor = indices_cc[i]
          
    ccdice = tp / (cc_pred + cc_true)
    
    return ccdice



def S(y1, y2):
    return np.sum(y1 * y2) / np.sum(y1)



def _resolve_neighborhood(footprint, connectivity, ndim, enforce_adjacency=True):
    """Validate or create a footprint (structuring element).

    Depending on the values of `connectivity` and `footprint` this function
    either creates a new footprint (`footprint` is None) using `connectivity`
    or validates the given footprint (`footprint` is not None).

    Parameters
    ----------
    footprint : ndarray
        The footprint (structuring) element used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    ndim : int
        Number of dimensions `footprint` ought to have.
    enforce_adjacency : bool
        A boolean that determines whether footprint must only specify direct
        neighbors.

    Returns
    -------
    footprint : ndarray
        Validated or new footprint specifying the neighborhood.

    Examples
    --------
    >>> _resolve_neighborhood(None, 1, 2)
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]])
    >>> _resolve_neighborhood(None, None, 3).shape
    (3, 3, 3)
    """
    if footprint is None:
        if connectivity is None:
            connectivity = ndim
        footprint = ndi.generate_binary_structure(ndim, connectivity)
    else:
        # Validate custom structured element
        footprint = np.asarray(footprint, dtype=bool)
        # Must specify neighbors for all dimensions
        if footprint.ndim != ndim:
            raise ValueError(
                "number of dimensions in image and footprint do not" "match"
            )
        # Must only specify direct neighbors
        if enforce_adjacency and any(s != 3 for s in footprint.shape):
            raise ValueError("dimension size in footprint is not 3")
        elif any((s % 2 != 1) for s in footprint.shape):
            raise ValueError("footprint size must be odd along all dimensions")

    return footprint



def _label_bool(image, background=None, return_num=False, connectivity=None):
    """Faster implementation of clabel for boolean input.

    See context: https://github.com/scikit-image/scikit-image/issues/4833
    """
    if background == 1:
        image = ~image

    if connectivity is None:
        connectivity = image.ndim

    if not 1 <= connectivity <= image.ndim:
        raise ValueError(
            f'Connectivity for {image.ndim}D image should '
            f'be in [1, ..., {image.ndim}]. Got {connectivity}.'
        )

    footprint = _resolve_neighborhood(None, connectivity, image.ndim)
    result = ndimage.label(image, structure=footprint)

    if return_num:
        return result
    else:
        return result[0]


def label(label_image, background=None, return_num=False, connectivity=None):
    r"""Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor::

      1-connectivity     2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]             [ ]
            |               \  |  /                 |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
            |               /  |  \             hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    label_image : ndarray of dtype int
        Image to label.
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as 0. By default, 0-valued pixels are considered as background
        pixels.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    See Also
    --------
    skimage.measure.regionprops
    skimage.measure.regionprops_table

    References
    ----------
    .. [1] Christophe Fiorio and Jens Gustedt, "Two linear time Union-Find
           strategies for image processing", Theoretical Computer Science
           154 (1996), pp. 165-181.
    .. [2] Kensheng Wu, Ekow Otoo and Arie Shoshani, "Optimizing connected
           component labeling algorithms", Paper LBNL-56864, 2005,
           Lawrence Berkeley National Laboratory (University of California),
           http://repositories.cdlib.org/lbnl/LBNL-56864

    Examples
    --------
    >>> import numpy as np
    >>> x = np.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, connectivity=1))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, connectivity=2))
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, background=-1))
    [[1 2 2]
     [2 1 2]
     [2 2 1]]
    >>> x = np.array([[1, 0, 0],
    ...               [1, 1, 5],
    ...               [0, 0, 0]])
    >>> print(label(x))
    [[1 0 0]
     [1 1 2]
     [0 0 0]]
    """
    if label_image.dtype == bool:
        return _label_bool(
            label_image,
            background=background,
            return_num=return_num,
            connectivity=connectivity,
        )
    else:
        return "Not supported because of Numpy 2.0.1 and scikit-image 0.25.0 incompatibility"