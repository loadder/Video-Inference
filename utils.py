from collections import deque
import numpy as np

def center_crop_images(images, crop_dims):
    """
    Crop images into center.
    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (height x width x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    crops_ix = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = crops_ix[0]
    crops_ix = crops_ix.astype(int)
    # Extract crops
    crop_images = deque([], len(images))
    
    for im in images:
        crop_image = im[crops_ix[0]:crops_ix[2], crops_ix[1]:crops_ix[3], :]
        crop_images.append(crop_image)

    return crop_images
