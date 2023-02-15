import Classes
from Const import brick_dims_UK

# Brick dims. Only the width changes.
brick_dims1 = [0.25, 0.125, 0.0675]
brick_dims2 = [0.125, 0.125, 0.0675]
brick_dims3 = [0.0625, 0.125, 0.0675]
wall_dims = [1, 0.25, 1.35]

origin = [0, 0, 0]
row1 = Classes.BlockRow(start_point=origin, Block_Dims=brick_dims1, num_blocks=(wall_dims[0]/brick_dims1[0]))

row2 = Classes.BlockRow(start_point=origin, Block_Dims=brick_dims2, num_blocks=1)
row2.add_block('right', 1, brick_dims3)
row2.add_block('right', 5, brick_dims2)
row2.add_block('right', 1, brick_dims3)
row2.add_block('right', 1, brick_dims2)
row2.move([0, 0, brick_dims1[2]])