
# File   : knn.py
# Brief  : Calculate the top k nearest neighbours of specfic inputs (probe
#          data and gallery data).
# Version: 0.2
# Author : by da.li on 2017/04/24

import numpy as np

from pyflann import *

def knn(probe_data, gallery_data, k, alg='linear'):
    """
    probe_data    - The top k Nearest Neighbours of this dataset will be found.
                  - <type 'numpy.ndarray'>
    gallery_data  - Find the top k Nearest Neighbours from this dataset.
                  - <type 'numpy.ndarray'>
    alg           - The chosen algoritm to do Approximate Nearest Neighbour(ANN).
    k             - Top k.
    """
    # Get data size.
    probe_data_num = 0
    probe_data_dim = 0
    if len(probe_data.shape) == 1:
        probe_data_num = 1
        probe_data_dim = probe_data.shape[0]
    else:
        probe_data_num = probe_data.shape[0]
        probe_data_dim = probe_data.shape[1]

    # Get knn
    flann = FLANN()
    results, dists = flann.nn(gallery_data, probe_data, k, algorithm=alg)

    # Reshape the final results
    knn = gallery_data[results].reshape(probe_data_num, probe_data_dim*k)

    return (knn, dists)

# Test
if __name__ == "__main__":
    probe = np.asarray([[1.0,1.0,1.0], [1.0,2.0,2.0], [1.5,1.2,1.0]])
    #probe = np.asarray([1.0,1.0,1.0])
    print len(probe.shape)
    gallery = np.asarray([[1.1,2.1,3.1], [2.1,3.1,4.1], [3.1,4.1,5.1], [4.1,5.1,6.1], [5.1,6.1,7.1], [6.1,7.1,8.1],\
                          [8.1,9.1,10.1]]);
    print probe
    print "==============="
    res = knn(probe, gallery, 3)
    print res
