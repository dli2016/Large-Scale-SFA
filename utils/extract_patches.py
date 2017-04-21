# File   : extract_patches.py
# Brief  : Extract patches from input images based on some constrains.
# Version: 2.2
# Author : by da.li on 2017/04/20

import sys
import Image
import numpy as np

#from pyspark.rdd import RDD
#from pyspark import SparkContext

from file_operations import getPathAndKey, loadImages

def doExtraction(data, patch_w, patch_h, w, h, stride):
    """
    Extract local patches from image based on specific stride and size.

    Args:
      data    - image data.
      patch_w - width of the local patches.
      patch_h - heigth of the local patches.
      w       - width of the input image.
      h       - heigth of the input image.
      stride  - space interval of extraction (unit: pixels).

    Returns:
      patches with key.
    """
    src_dims = patch_w * patch_h
    # Result.
    patches = []
    gray_max_val = 255.0
    # Columns.
    ys = range(0, w - patch_w + 1, stride)
    # Rows.
    xs = range(0, h - patch_h + 1, stride)
    # Extraction.
    for x in xs:
        bb_lower = x + patch_h
        for y in ys:
            bb_right = y + patch_w
            bb = (y, x, bb_right, bb_lower)
            patch_obj = data.crop(bb)
            # Convert data format to double
            patch_data_arr = np.asarray(patch_obj.getdata())
            patch_ddata = patch_data_arr / gray_max_val
            patch_ddata_max = np.amax(patch_ddata, axis=1)
            patches.append(patch_ddata_max)
    # Returns.
    return patches

def extractPatches(image_filenames, patch_w, patch_h, w, h, stride, with_key=0):

    """
    Extract patches from training images.
    
    Args:
      image_filenames - Filenames of input images.
      patch_w         - Patch width.
      patch_h         - Patch height.
      w               - Image width after resizing.
      h               - Image height after resizing.
      stride          - The distance between two conjunctive patches.
      with_key        - 0 no, 1 yes.
    
    Returns:
      Patches data.
    """
    if with_key == 0:
        image = loadImages(image_filenames, w, h)
        patches = doExtraction(image, patch_w, patch_h, w, h, stride)
        return patches
    elif with_key == 1:
        path, key = getPathAndKey(image_filenames)
        image = loadImages(path, w, h)
        patches = doExtraction(key, image, patch_w, patch_h, w, h, stride)
        # the key indicates the image index that the patch comes from.
        return (key, patches)
    else:
        print "LaS-SFA: BAD value of with_key!"
        return None

