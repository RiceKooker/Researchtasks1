import BlockClass as bc
from const import brick_dims
from utils import BricksBoundary

brick_dim1 = brick_dims.copy()
brick_dim2 = brick_dims.copy()
brick_dim2[0] = brick_dim2[0]/2
block1 = bc.Block.dim_build(brick_dim1)
block2 = bc.Block.dim_build(brick_dim2)
row1 = bc.BlockGroup(block1)
row1.expand('right', times=4)
row2_temp1 = bc.BlockGroup(block2)
row2_temp2 = bc.BlockGroup(block1)
row2_temp2.expand('right', times=3)
row2_temp2.move([brick_dim2[0], 0, 0])
row2_temp3 = bc.BlockGroup(block2)
row2_temp3.move([brick_dim2[0]+4*brick_dim1[0], 0, 0])
row2 = row2_temp1.add(row2_temp2, row2_temp3)
row2.move([0, 0, brick_dim1[2]])

rows_repeat = row1.add(row2)
rows_repeat.expand(side='top', times=16)
wall = rows_repeat

brick_commands = wall.three_DEC_create()
boundary_bricks = BricksBoundary(vertices=wall.find_vertices(), thickness=0.8)

print(brick_commands + boundary_bricks.commands)
