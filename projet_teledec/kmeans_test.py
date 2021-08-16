import gdal
from fototex.foto import Foto, FotoSector, FotoBatch
from skimage import io
from fototex.foto_tools import degrees_to_cardinal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

