import cv2 as cv
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

def clustering(img, method,K):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 70, 0.1)
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    if method =='k_means':
        _, labels, (centers) = cv.kmeans(pixels, K, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        return labels, centers
    else:

        connectivity = kneighbors_graph(pixels, 7)
        ward = AgglomerativeClustering(n_clusters=K, connectivity=connectivity).fit(pixels)
        return ward


    return 0;


