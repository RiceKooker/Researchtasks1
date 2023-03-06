import BlockClass as bc
import copy
from Const import brick_dims_UK

# Brick dims. Only the width changes.
brick_dims1 = [0.25, 0.125, 0.0675]
brick_dims2 = [0.125, 0.25, 0.0675]
brick_dims3 = [0.0625, 0.25, 0.0675]
wall_dims = [1, 0.25, 1.35]

origin = [0, 0, 0]
block1 = bc.Block.dim_build(brick_dims1)
row1 = bc.BlockGroup([block1, block1.duplicate('right')])
row1.expand('right')
block2 = bc.Block.dim_build(brick_dims2)
block3 = block2.duplicate('right', brick_dims3)
row2_list = [block2, block3]
for i in range(5):
    row2_list.append(row2_list[-1].duplicate('right', new_dims=brick_dims2))
row2_list.append(row2_list[-1].duplicate('right', new_dims=brick_dims3))
row2_list.append(row2_list[-1].duplicate('right', new_dims=brick_dims2))
row2 = bc.BlockGroup(row2_list)
row2.move([0, 0, brick_dims1[2]])
row3 = row1.duplicate('back')
row1.add(row2)
row1.add(row3)
row1.expand('top', times=4)
tall_wall = copy.deepcopy(row1)
row1.expand('top')
English_wall = row1

tall_wall.expand('top', times=2)
print(tall_wall.three_DEC_create())