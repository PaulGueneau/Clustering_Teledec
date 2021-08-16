import cv2 as cv
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import math

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
    ratio = []
    vecs = []
    dists = []
    angle = []
    nb_clusters = 17
    for K in range(2,nb_clusters):
        compactness, labels, (centers) = cv.kmeans(pixels, K, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        inerties.append(compactness)
    X = np.arange(2,K+1,1)

    #for k in range(2,len(inerties)):
     #   ratio.append((inerties[k-1]-inerties[k])/inerties[k-1])
    for i in range(len(inerties)-1):
        vecs = np.append(vecs, [1, inerties[i+1]-inerties[i]])
        dists = np.append(dists,np.sqrt(inerties[i]**2+1)*np.sqrt(inerties[i+1]**2+1))

    vecs = vecs.reshape(len(inerties)-1,2)
    for j in range(len(vecs)-1):
        angle = (np.dot(vecs[i],vecs[i+1]))/(dists[i+1]*dists[i])

    angle = np.arccos(angle)
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


