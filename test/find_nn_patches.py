
# File  : find_nn_patches.py
# Brief : Study the influence of "Patches are in the SAME image" on kNN result.
# Author: by da.li on 2017/04/20

import sys
sys.path.append("../utils")

import numpy as np
import time

from extract_patches import extractPatches
from file_operations import *
from pyflann import *

def studyInfluences(path_root, imgs_path, is_random, params):

    """
    We mainly talk about the effect caused by following parameters.

    params:
      pw         - width of patches.
      ph         - height of patches.
      w          - width of images.
      h          - height of images.
      k          - top k.
      strides    - window step.
      images_num - number of images in gallery set.
    """
    # Params
    w = params['w']
    h = params['h']
    patch_w = params['pw']
    patch_h = params['ph']
    stride = params['strides']
    k = params['k']
    images_num = params['num_images']

    filenames_total = getImgsNameFromFile(imgs_path)
    filenames = []
    if is_random:
        np.random.shuffle(filenames_total)
        filenames = filenames_total[0:images_num]
    else:
        filenames = filenames_total[0:images_num:2]

    # Extract patches
    patches = []
    patches_num_per_image = 0
    time_stamp0 = time.time()
    for filename in filenames:
        filename = path_root + filename
        patches_temp = extractPatches(filename, patch_w, patch_h, w, h, stride)
        patches.extend(patches_temp)
        patches_num_per_image = len(patches_temp)

    patches = np.asarray(patches)
    time_stamp1 = time.time()
    print "Cost time of extracting patches is", (time_stamp1-time_stamp0)
    # print "size :", patches.shape

    # probes_num = int(images_num * 0.1)
    patches_num= patches.shape[0]
    probes_num = int(patches_num * 0.01)
    patches_idx= range(patches_num)
    np.random.shuffle(patches_idx)
    patches_idx_selected = patches_idx[0:probes_num]
    #print "Patch Index of the probe patches:", patches_idx_selected
    probe_patches = patches[patches_idx_selected]
    #probe_patches, gallery_patches, indexes
    gallery_patches = patches
    flann = FLANN()
    time_stamp2 = time.time()
    results, dists = flann.nn(gallery_patches, probe_patches, k, algorithm="kdtree")
    time_stamp3 = time.time()
    print "Cost time of flann is", (time_stamp3 - time_stamp2)
    #print "Result Patch Index of the probe patches:", results
    image_index_probe  = np.asarray(patches_idx_selected) / patches_num_per_image
    image_index_results= results / patches_num_per_image
    #print "Image Index of the probe patches:", image_index_probe
    #print "Result Image Index of the probe patches:"
    #print image_index_results

    per = []
    for ki in xrange(k):
        res_i = image_index_results[:,ki] - image_index_probe
        num = np.sum(res_i == 0)
        per_i = float(num) / len(image_index_probe)
        per.append(per_i)
    
    return per, np.asarray(patches_idx_selected), results, patches_num_per_image

# Test
if __name__ == "__main__":

    params = {\
        'w': 64,\
        'h': 128,\
        'pw': 16,\
        'ph': 16,\
        'strides': 4,\
        'k': 4,\
        'num_images': 100\
    }
    if len(sys.argv) == 4:
        patch_size = sys.argv[1]
        strides = sys.argv[2]
        num_images = sys.argv[3]

        params['pw'] = int(patch_size)
        params['ph'] = int(patch_size)
        params['strides'] = int(strides)
        params['num_images'] = int(num_images)
    
    is_random = 1
    path_root  = "/mnt/disk1/video_7T/inria_data/train/"
    image_path = "../test/data/inria_train_pos.txt"
    start = time.time()
    per, res1, res2, res3 = \
        studyInfluences(path_root, image_path, is_random, params)
    end = time.time()
    print "Total cost time is", (end - start)
    print "Result is", per
