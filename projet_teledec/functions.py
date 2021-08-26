import cv2 as cv
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def cluster_discrimination(K,labels,img):
    masked_image = np.copy(img)
    masked_image = masked_image.reshape((-1,3))
    colors = np.random.rand(K,3)


    for i in range(K):
         masked_image[labels==i] = colors[i]


    #masked_image[ward_labels==1] = [1,0,0]
    #masked_image[ward_labels==3] = [1,0,0]
    masked_image = masked_image.reshape(img.shape)
    #ward_labels = ward_labels.reshape(844,844)

    #### Legend ####
    ### def legend(labels,colors,K) à faire
    mycmap = plt.cm.jet
    values= np.unique(labels.ravel())
    patches = [ mpatches.Patch(color = colors[i], label= "Cluster {c}".format(c=values[i]) ) for i in range(K)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.title('Cluster separation')
    plt.imshow(masked_image)
    return(masked_image, patches)

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


def histo_cluster(labels,foto,K):
    dict = {}
    for i in range(K):
        dict[i] = foto[:, labels == i]
    return(dict)

def stats_clusters(dict,K,color):
    mean = [];
    min = [];
    max = [];
    median = [];
    std = []
    if color == 'red':
        j = 0
    elif color == 'green':
        j = 1
    else:
        j=2


    for k in range(K):
        mean.append(np.mean(dict[k][j]))
        median.append(np.median(dict[k][j]))
        max.append(np.max(dict[k][j]))
        min.append(np.min(dict[k][j]))
        std.append(np.std(dict[k][j]))

    print("means on "+color+"=", mean)
    print("medians on "+color+"=", median)

    print("mins on "+color+"=", min)
    print("maxs on "+color+"=", max)

    print("stds on" +color+"=", std)

