# Clustering_Teledec
A project in which we detect the urban zones on satellite images with a semi-supervised classification based on FOTOTEX and clustering algorithms.

- functions.py contains some of the main functions used in the main like filtering, clustering.. 
- indices_pyrasta computes the spectral indices (NDVI, NDWI and BI) and resamples them but others might be added.
- mask.py masks a cluster affecting no-data to it to keep the cluster containing the built-up areas.
- masked_foto.py applies the FOTOTEX algorithm on the mask result of mask.py 
