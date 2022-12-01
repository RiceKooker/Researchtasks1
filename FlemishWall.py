from Classes import BlockWall, ThreeDECScript
import Const
import numpy as np

print(Const.brick_dim_num)
def create_geometry(dimensions=Const.brick_dim_num, block_num=True):
    if block_num:
        wall_dim = np.multiply(np.array(dimensions), np.array(Const.brick_dims_UK))
    else:
        wall_dim = dimensions
    wall = BlockWall(wall_dim)
    return wall.three_DEC_create()


brick_commands = create_geometry()
model_creation = f"""
model new
model config dynamic
model large-strain on
""" + brick_commands + """
block create brick -1.8 4.4 -1 1.4 -0.4 0
"""

material_properties = """
block property density 2400

block contact generate-subcontacts
block contact jmodel assign mohr
block contact property stiffness-normal 2e9 stiffness-shear 2e9 friction 40
block contact material-table default property stiffness-normal 2e9 stiffness-shear 2e9 friction 40
"""

boundary_conditions = """
model gravity 0 0 -9.81
block fix range pos-z -0.4 0
"""

loadings = """
block mech damp local 0.8
model solve

model save 'initial'

block gridpoint ini disp 0 0 0
block contact reset disp

model dynamic active on
block mech damp rayleigh 0 0 

[freq = 5.0]
fish def cos_
  cos_ = math.cos(2.0*math.pi*mech.time*freq)
end
fish def sin_
  sin_ = math.sin(2.0*math.pi*mech.time*freq)
end

block apply vel-x 1 fish cos_ range pos-z -0.4 0
block apply vel-y 1 fish sin_ range pos-z -0.4 0
block apply vel-z 0.1 fish cos_ range pos-z -0.4 0

 ;source
block history vel-x pos -1.8 -1 -0.4
block history vel-z pos -1.8 -1 -0.4

block history vel-x pos 1.4 0 1
block history vel-z pos 1.4 0 1
block history disp-x pos 1.4 0 1
block history disp-z pos 1.4 0 1
model solve time 0.5

model save 'wall'
"""

script = ThreeDECScript(model_creation, boundary_conditions, loadings, material_properties).final_commands()
