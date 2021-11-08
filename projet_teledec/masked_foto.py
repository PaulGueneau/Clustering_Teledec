from fototex.foto import Foto

foto = Foto("",method="block")

foto.run(19,keep_dc_component=True)

foto.save_rgb()
