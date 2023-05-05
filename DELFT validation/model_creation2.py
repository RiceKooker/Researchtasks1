import BlockClass as bc
from const import brick_dims
from utils import BricksBoundary, print_string_list

brick_dim1 = brick_dims.copy()
brick_dim2 = brick_dims.copy()
brick_dim2[0] = brick_dim2[0]/2
block1 = bc.Block.dim_build(brick_dim1)
block2 = bc.Block.dim_build(brick_dim2)
row1 = bc.BlockGroup(block1)
row1.expand('right', times=17)
row2_temp1 = bc.BlockGroup(block2)
row2_temp2 = bc.BlockGroup(block1)
row2_temp2.expand('right', times=16)
row2_temp2.move([brick_dim2[0], 0, 0])
row2_temp3 = bc.BlockGroup(block2)
row2_temp3.move([brick_dim2[0]+17*brick_dim1[0], 0, 0])
row2 = row2_temp1.add(row2_temp2, row2_temp3)
row2.move([0, 0, brick_dim1[2]])

rows_repeat = row1.add(row2)
rows_repeat.expand(side='top', times=16)
wall = rows_repeat
wall.draw()

brick_commands = wall.three_DEC_create_crack_surface()

boundary_bricks = BricksBoundary(vertices=wall.find_vertices(), thickness=0.8)

print(brick_commands + boundary_bricks.commands)
print(print_string_list(wall.get_id_list()))
