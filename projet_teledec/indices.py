import gdal
import numpy as np
from pyrasta.raster import Raster


#def ndvi(arrays):
#    return (arrays[0] - arrays[1])/(arrays[0] + arrays[1])


#def ib(*channels):
#    return np.sqrt(np.sum([c**2 for c in channels]))


nir = Raster("/home/gueneau/Documents/T16SBC_20210321T163951_B08_20m.tif")
r = Raster("/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20200722T130251_N0209_R095_T24MVU_20200722T143759.SAFE/GRANULE/L1C_T24MVU_A026547_20200722T130253/IMG_DATA/T24MVU_20200722T130251_B04.jp2")
b = Raster("/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20200722T130251_N0209_R095_T24MVU_20200722T143759.SAFE/GRANULE/L1C_T24MVU_A026547_20200722T130253/IMG_DATA/T24MVU_20200722T130251_B02.jp2")
g = Raster("/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20200722T130251_N0209_R095_T24MVU_20200722T143759.SAFE/GRANULE/L1C_T24MVU_A026547_20200722T130253/IMG_DATA/T24MVU_20200722T130251_B03.jp2")
swir = Raster("/home/gueneau/Documents/S2A_MSIL1C_20210321T163951_N0209_R126_T16SBC_20210321T203030.SAFE/GRANULE/L1C_T16SBC_A030010_20210321T164459/IMG_DATA/T16SBC_20210321T163951_B11.jp2")


# ndvi = (nir - r)/(nir + r)
# ib = (b**2+g**2+nir**2+r**2)**0.5
# ndwi = (nir -g)/(nir + g)
ndbi = (swir - nir)/(swir + nir)

# ndvi.to_file("/home/gueneau/Documents/NDVI_T24.tif")
# ib.to_file("/home/gueneau/Documents/IB_T24.tif")
# ndwi.to_file("/home/gueneau/Documents/NDWI_T24.tif")
ndbi.to_file("/home/gueneau/Documents/NDBI_T16.tif")
#ndvi_raster = Raster.raster_calculation(rasters=[nir, r], fhandle=ndvi,
#                                        output_type=gdal.GetDataTypeByName("float32"))
