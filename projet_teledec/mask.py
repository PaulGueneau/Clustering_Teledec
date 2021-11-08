from functools import partial

import numpy as np
from pyrasta.raster import Raster


def set_no_data(arrays, label, no_data):
    r1, labels = arrays
    r1[labels == label] = no_data

    return r1


label_raster = Raster("/home/gueneau/Documents/labels_t16_2_NaN=0.tif")
# sentinel2 = Raster("/home/gueneau/Documents/clip_sentinel2.tif")
sentinel2 = Raster("/home/gueneau/Documents/FOTO_Clustering_T16_/masked_sentinel2_t16_w19.tif")
result_raster = Raster.raster_calculation([sentinel2, label_raster], partial(set_no_data, label=1, no_data=-999), no_data=-999)
result_raster.to_file("/home/gueneau/Documents/masked_sentinel2_t16_NaN=0.tif")
