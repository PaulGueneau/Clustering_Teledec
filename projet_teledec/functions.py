import cv2 as cv
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler


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
    scaler = MinMaxScaler()
    X = scaler.fit_transform(pixels)
    mask = np.random.choice([False,True], len(pixels), p=[0.95,0.05])
    img_sample = X[mask]
    inertia = [np.nan]
    dists = []
    km_score = []
    km_silhouette = []
    db_score = []
    nb_clusters = 6
    for n in range(nb_clusters):
        kmeans = KMeans(n_clusters=n+1, n_init=10, max_iter=300).fit(img_sample)
        labels = kmeans.predict(img_sample)
        inertia.append(kmeans.inertia_)
        km_score.append(-kmeans.score(img_sample)/100)
        if n == 0:
            km_silhouette.append(np.nan)
            db_score.append(np.nan)
        else:
            km_silhouette.append(silhouette_score(img_sample,labels))
            db_score.append(davies_bouldin_score(img_sample,labels))
    inertia = np.asarray(inertia)
    delta_r  = inertia[1:-1]/ inertia[0:-2] - inertia[2:] / inertia[1:-1]



    return km_silhouette,db_score, delta_r;






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
    return mean

