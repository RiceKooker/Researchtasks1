import itasca as it
import numpy as np
it.command("python-reset-state false")

# Physical dimensions of the wall
Wall_dimensions = np.array([3, 0.4, 2])
Height = Wall_dimensions[2]
Length = Wall_dimensions[0]
Thickness = Wall_dimensions[1]
# Percentage displacement relative to the dimension in each direction
Displacement_ratio = np.array([0.2, 0.2, 0])
# Absolute displacements in 3 directions
Dis = np.multiply(Wall_dimensions, Displacement_ratio)

n_cycle = 1000
Vel = Dis/n_cycle


load_scrip1 = """
[freq = 5.0]
fish def cos_
  cos_ = math.cos(2.0*math.pi*mech.time*freq)
end
fish def sin_
  sin_ = math.sin(2.0*math.pi*mech.time*freq)
end

block apply vel-x 1 fish cos_ range pos-z -0.4 0
block apply vel-y 1 fish sin_ range pos-z -0.4 0
block apply vel-z 0.1 fish cos_ range pos-z -0.4
"""

load_scrip2 = f"""
block apply vel-x {Vel[0]} range pos-z 1.947 1.997
block apply vel-y {Vel[1]} range pos-z 1.947 1.997
"""

command_script = """
model new
model config dynamic
model large-strain on

block generate from-vrml filename 'wall.wrl'

block create brick -1.8 4.4 -1 1.4 -0.4 0
block property density 2400

block contact generate-subcontacts
block contact jmodel assign mohr
block contact property stiffness-normal 2e9 stiffness-shear 2e9 friction 40
block contact material-table default property stiffness-normal 2e9 stiffness-shear 2e9 friction 40

model gravity 0 0 -9.81

block fix range pos-z 1.947 1.997  ; Yilong edited - fix the top of the wall
block fix range pos-z -0.4 0

block mech damp local 0.8
model solve

model save 'initial'

block gridpoint ini disp 0 0 0
block contact reset disp

model dynamic active on
block mech damp rayleigh 0 0 

""" + load_scrip2 + f"""

 ;source
block history vel-x pos -1.8 -1 -0.4
block history vel-z pos -1.8 -1 -0.4

block history vel-x pos 1.4 0 1
block history vel-z pos 1.4 0 1
block history disp-x pos 1.4 0 1
block history disp-z pos 1.4 0 1

;model solve time 0.5
model cycle {n_cycle}

model save 'wall'

"""
it.command(command_script)

# import numpy as np
#
#
# class Brick:
#     def __init__(self, location, length):
#         # Location and length are both 3-dimensional vectors representing values in x, y, and z directions.
#         # At the front face, point 1 is the left bottom corner and points 2,3 and 4 are assigned in a clockwise manner.
#         # Points 5, 6, 7, and 8 are assigned similarly
#         self.location = location
#         self.length = length
#
#         CoM = np.zeros(self.location.shape)
#         ranges = np.zeros([len(self.location), 2])
#
#         for i, data in enumerate(zip(self.location, self.length)):
#             start_point, length_single = data
#             CoM[i] = start_point + 0.5 * length_single
#             ranges[i] = [start_point, start_point+length_single]
#
#         self.CoM = CoM
#         self.ranges = ranges
#
#
#
#
#
# if __name__ == '__main__':
#     sample_loc = np.array([1, 0, 1])
#     sampe_length = np.array([1, 2, 1])
#     b1 = Brick(sample_loc, sampe_length)
#     print(b1.ranges)


