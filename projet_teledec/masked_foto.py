from fototex.foto import Foto

foto = Foto("/home/gueneau/Documents/masked_sentinel2_t34kbg_2.tif",method="block")

foto.run(19)

foto.save_rgb()
