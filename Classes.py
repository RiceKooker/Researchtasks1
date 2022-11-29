import numpy as np
import Func
import matplotlib.pyplot as plt
from colorama import Fore, Style
from Const import origin, brick_dims_UK


class Block:
    def __init__(self, CoM, Dims):
        """
        Create a block with a given center of mass and dimensions
        :param CoM: Center of mass in 3 dimensional vectors
        :param Dims: Dimensions in 3 dimensional vectors
        """
        self.CoM = np.array(CoM)
        self.Dims = np.array(Dims)
        self.Vertices = Func.find_vertices(self.CoM, self.Dims)
        self.volume = self.Dims[0]*self.Dims[1]*self.Dims[2]

    def move(self, dis_vec):
        """
        Move the block with a certain displacement vector
        :param dis_vec: 3 dimensional displacement vector
        :return:
        """
        for i, vertex in enumerate(self.Vertices):
            self.Vertices[i] = vertex + dis_vec

        self.CoM = find_block_CoM(self.Vertices)

    def duplicate(self, side, new_Dims=None):
        """
        This function creates another block with potentially different dimensions to the side of the current block.
        :param side: one of {'top', 'bot', 'left', 'right'}
        :param new_Dims: The dimensions of the new block. It's taken to be the same as the current block by default.
        :return: A new block object
        """
        axis, direction = Func.find_axis_and_direction(side)
        temp_CoM = np.array(self.CoM)

        # Find the displacement between the old and new block
        if new_Dims is not None:
            CoM_Dis = 0.5*(self.Dims[axis] + new_Dims[axis])
        else:
            CoM_Dis = self.Dims[axis]
            new_Dims = self.Dims
        temp_CoM[axis] += direction*CoM_Dis

        return Block(temp_CoM, new_Dims)

    def three_DEC_create(self):
        x_lim = [self.Vertices[0][0], self.Vertices[3][0]]
        y_lim = [self.Vertices[0][1], self.Vertices[4][1]]
        z_lim = [self.Vertices[0][2], self.Vertices[1][2]]
        return f"block create brick {x_lim[0]:.10f} {x_lim[1]:.10f} {y_lim[0]:.10f} {y_lim[1]:.10f} {z_lim[0]:.10f} {z_lim[1]:.10f} \n"


def find_block_CoM(vertices):
    n = 0
    CoM = 0
    for vertex in vertices:
        CoM += vertex
        n += 1
    return CoM / n


class BlockRow1:

    def __init__(self, start_point, length, Block_Dims):
        block_list = []
        num_block = round(length/Block_Dims[0])
        CoM1 = np.array(start_point) + np.array(Block_Dims)*0.5
        block_prev = Block(CoM1, Block_Dims)
        block_list.append(block_prev)
        for i in range(num_block-1):
            new_block = block_prev.duplicate('right')
            block_list.append(new_block)
            block_prev = new_block
        self.block_list = block_list
        self.CoM = find_row_CoM(self.block_list)

    def add_block(self, side, num, block_dims=None):
        if side not in ['right', 'left']:
            print(Fore.RED + 'Error: cannot add blocks in the y and z directions')
            return None
        axis, direction = Func.find_axis_and_direction(side)
        if side == 'left':
            block_prev = self.block_list[0]
            for i in range(num):
                block_new = block_prev.duplicate(side, block_dims)
                self.block_list.insert(0, block_new)
                block_prev = block_new
        elif side == 'right':
            block_prev = self.block_list[-1]
            for i in range(num):
                block_new = block_prev.duplicate(side, block_dims)
                self.block_list.append(block_new)
                block_prev = block_new
        self.CoM = find_row_CoM(self.block_list)


def find_row_CoM(block_list):
    CoM = 0
    v = 0
    for block in block_list:
        CoM += block.CoM * block.volume
        v += block.volume
    return CoM / v


class BlockRow(BlockRow1):
    def __init__(self, start_point=None, length=None, Block_Dims=None, block_list=None, num_blocks=None):
        """
        The constructor creates a row of blocks and stores all blocks in a list by taking in the dimension specifications
        or a list of blocks
        :param start_point: the coordinates of the first vertex
        :param length: length of the row
        :param Block_Dims: dimensions of the blocks used
        :param block_list: used when copying a row
        """
        if block_list is None:
            if num_blocks is not None:
                BlockRow1.__init__(self, start_point, num_blocks*Block_Dims[0], Block_Dims)
            else:
                BlockRow1.__init__(self, start_point, length, Block_Dims)
        else:
            self.block_list = block_list
        self.CoM = find_row_CoM(self.block_list)

    def duplicate(self, side):
        """
        This function copies the current row and pastes it in any directions of the current block
        except for the primary axis of the current row
        :param side: can be one of 'top', 'bot', 'front'. 'back'
        :return: a new row
        """
        if side in ['left', 'right']:
            print(Fore.RED + 'Error: cannot copy rows in the x direction')
            return None
        new_blocks = []
        for block in self.block_list:
            new_blocks.append(block.duplicate(side))
        return BlockRow(block_list=new_blocks)

    def move(self, dis_vec):
        """
        This function moves the current row given a displacement vector
        :param dis_vec: a 3 dimensional vector indicating the movement
        :return: nothing
        """
        for i, block in enumerate(self.block_list):
            self.block_list[i].move(dis_vec)
        self.CoM = find_row_CoM(self.block_list)

    def three_DEC_create(self):
        total_commands = ''
        for block in self.block_list:
            total_commands += block.three_DEC_create()
        return total_commands


def create_type2_row(num_x, block_dims):
    Br = BlockRow(start_point=origin, num_blocks=num_x-1, Block_Dims=block_dims)

    half_dims = np.array(block_dims)
    half_dims[0] = half_dims[0]*0.5
    Br.add_block('left', 1, half_dims)
    Br.add_block('right', 1, half_dims)
    Br.move(np.array([half_dims[0], 0, 0]))
    return Br


# class BlockWall(BlockRow):
#     def __init__(self, start_point=None, Block_Dims=brick_dims_UK, num_blocks=None, row_list=None):
#         self.rows = []
#         if row_list is None:
#             Br = BlockRow.__init__(self, start_point=start_point, Block_Dims=Block_Dims, num_blocks=num_blocks)
#             self.rows.append(Br)
#         else:
#             for row in row_list:
#                 self.rows.append(row)


class BlockWall:
    def __init__(self, wall_dims, block_dims=brick_dims_UK):
        """
        The constructor creates a list of rows that makes up a wall.
        The index of each row in the list increases from bottom to top, then continues from the bottom of the next layer.
        The final dimensions of the wall will be close to the dimensions given but might differ due to the discrete blocks.
        :param wall_dims: dimensions of the wall
        :param block_dims: dimensions of the bricks
        """
        num_bricks = []  # Number of blocks in x,y and z directions
        for w_dim, b_dim in zip(wall_dims, block_dims):
            num_bricks.append(round(w_dim/b_dim))
        num_x, num_y, num_z = num_bricks[0], num_bricks[1], num_bricks[2]

        row_list = []
        for j in range(num_y):
            # Reset the first two rows in a layer
            first_row_type0 = BlockRow(start_point=origin, num_blocks=num_x, Block_Dims=block_dims)
            first_row_type1 = create_type2_row(num_x, block_dims)
            first_row_type1.move(np.array([0, 0, block_dims[2]]))
            dis_vec_y = j * np.array([0, block_dims[1], 0])  # Move the first two rows to the correct layer
            first_row_type0.move(dis_vec=dis_vec_y)
            first_row_type1.move(dis_vec=dis_vec_y)
            row_dict = {0: first_row_type0, 1: first_row_type1}
            row_list.append(row_dict[0])
            row_list.append(row_dict[1])
            for i in range(2, num_z):
                # Choose the right row type based on the z position
                if (i % 2) == 0:
                    row_type = 0
                else:
                    row_type = 1
                # Create a new row based on the previous row of the same type
                row_prev = row_dict[row_type]
                row_new = row_prev.duplicate('top')
                row_new.move(np.array([0, 0, block_dims[2]]))
                row_list.append(row_new)  # Add the new row to the row list
                row_dict[row_type] = row_new  # Update the reference row
        self.row_list = row_list

    def draw_wall(self):
        block_list = []
        for row in self.row_list:
            block_list += row.block_list
        print(f'Number of blocks : {len(block_list)}')
        Func.draw_blocks(block_list)

    def __len__(self):
        return len(self.row_list)

    def three_DEC_create(self):
        total_command = ''
        for row in self.row_list:
            total_command += row.three_DEC_create()
        return total_command


if __name__ == '__main__':
    # sample_CoM = [2.4, 4, 5.9]
    # sample_Dims = [2.9, 1.5, 0.9]
    # sample_Dims2 = [2.9*0.4, 3.3*2, 7.2*0.3]
    # Dims_half_x = [2.9*0.5, 1.5, 0.9]
    # Br1 = BlockRow([0,0,0], 15, sample_Dims)
    # Br1.add_block('left', 1, block_dims=Dims_half_x)
    # Br1.add_block('right', 1, block_dims=Dims_half_x)
    # Br2 = BlockRow([0,0,0], 18, sample_Dims)
    # Br2 = Br2.duplicate('top')
    # Br1.move([Dims_half_x[0], 0, 0])
    # Func.draw_blocks(Br1.block_list+Br2.block_list)

    num_blocks = np.array([8, 2, 5])
    sample_wall_dims = np.multiply(num_blocks, np.array(brick_dims_UK))
    W1 = BlockWall(sample_wall_dims)
    print(f'Number of rows: {len(W1)}')
    # W1.draw_wall()
    with open('sample.dec', 'w+') as f:
        f.write(W1.three_DEC_create())
