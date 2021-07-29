import cv2 as cv
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

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


def elbow(pixels,criteria):
    inerties = []
    nb_clusters = 15
    for K in range(2,nb_clusters):
        compactness, labels, (centers) = cv.kmeans(pixels, K, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        inerties.append(compactness)
    X = np.arange(2,K+1,1)

    plt.figure()
    plt.title("Elbow's method")
    plt.plot(X,inerties)
    plt.xlabel('K')
    plt.ylabel('Compactness')
    plt.xlim(2,nb_clusters)

    return 0;


def filtering(ker,type,img):
    if type == 'Gaussian':
        img = cv.GaussianBlur(img,ker,0)
    else:
        img = cv.medianBlur(img,len(ker))
    return(img)


