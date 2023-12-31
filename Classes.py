import numpy as np
import Func
from colorama import Fore, Style
from Const import origin, brick_dims_UK, grid_point_reader
from model_specs import brick_dim_num
from Const import output_geometry_file_separator as gfs
import pandas as pd
import Func as fc
from DisplacementMapping import RBlockClass as Rb


class BlockDraw:
    def __init__(self, vertices, transform=None):
        if transform is not None:
            vertices = transform(vertices)
        self.vertices = vertices


class Block1:
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

        self.CoM = fc.find_block_CoM(self.Vertices)


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


class Block(Block1):
    """
    This class is for creation commands. And it does not support rotations.
    """
    def __init__(self, CoM=None, Dims=None, vertices=None):
        if vertices is None:
            Block1.__init__(self, CoM, Dims)
        else:
            CoM, Dims = fc.verts_to_Dims_CoM(vertices)
            Block1.__init__(self, CoM, Dims)


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
        self.CoM = fc.find_row_CoM(self.block_list)

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
        self.CoM = fc.find_row_CoM(self.block_list)


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
        self.CoM = fc.find_row_CoM(self.block_list)

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
        self.CoM = fc.find_row_CoM(self.block_list)

    def three_DEC_create(self):
        total_commands = ''
        for block in self.block_list:
            total_commands += block.three_DEC_create()
        return total_commands

    def draw(self):
        Func.draw_blocks(self.block_list)


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
            num_bricks.append(int(round(w_dim/b_dim)))
        num_x, num_y, num_z = num_bricks[0], num_bricks[1], num_bricks[2]

        self.actual_dim = np.multiply(np.array(num_bricks), np.array(block_dims))
        self.num_blocks = num_bricks

        row_list = []
        for j in range(num_y):
            z_built = 0
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

    def draw(self):
        block_list = []
        for row in self.row_list:
            block_list += row.block_list
        print(f'Number of blocks : {len(block_list)}')
        Func.draw_blocks(block_list, wall_vert=self.find_vertices())

    def __len__(self):
        return len(self.row_list)

    def three_DEC_create(self):
        total_command = ''
        for row in self.row_list:
            total_command += row.three_DEC_create()
        return total_command

    def find_vertices(self):
        """
        This function returns the vertices of the wall following the same ordering condition as blocks.
        :return:
        """
        max_dims = np.zeros((3,))
        min_dims = 100000*np.ones((3,))
        for row in self.row_list:
            for block in row.block_list:
                for vertex in block.Vertices:
                    for axis, axis_value in enumerate(vertex):
                        if axis_value < min_dims[axis]:
                            min_dims[axis] = axis_value
                        if axis_value > max_dims[axis]:
                            max_dims[axis] = axis_value
        return Func.find_vertices_from_max(min_dims, max_dims)

    def all_vertices(self):
        for row in self.row_list:
            for i, block in enumerate(row.block_list):
                print(f'Block {i+1}')
                print('----------------------------------------------------------------------------')
                print('----------------------------------------------------------------------------')
                for vertex in block.Vertices:
                    print(vertex)


class ThreeDECScript:
    def __init__(self, model_creation, material_properties, boundary_conditions, loadings):
        self.model_creation = model_creation
        self.boundary_conditions = boundary_conditions
        self.loadings = loadings
        self.material_properties = material_properties

    def final_commands(self):
        return self.model_creation + '\n' + self.material_properties + '\n' + self.boundary_conditions + '\n' + self.loadings


class BlockGroup:
    def __init__(self, vert_list):
        self.block_list = []
        for block_verts in vert_list:
            self.block_list.append(BlockDraw(vertices=block_verts))

    def draw(self):
        Func.draw_blocks(self.block_list)


class GeometryReader:
    """
    This class reads the geometry text file generated from 3DEC and post-process it for reconstruction.
    """
    def __init__(self, file_name):
        self.txt = {gfs[0]: [], gfs[1]: [], gfs[2]: []}
        self.gridpoints = {}
        self.vertices = {}
        self.faces = {}
        self.blocks = {}
        with open(file_name, 'rb') as f:
            lines = f.readlines()
            separator_count = 0
            record = False
            decode_spec = "utf-8"
            for i, line in enumerate(lines):
                line = line.decode(decode_spec)
                line = line.replace('\n', '')
                if '*' in line and record:
                    record = False
                    separator_count += 1
                    if separator_count > 2:
                        break
                if record:
                    self.txt[gfs[separator_count]].append(line)
                if gfs[separator_count] in line:
                    record = True

    def get_block_vertices(self):
        """
        This function returns a list containing all the coordinates of all
        vertices of blocks described in the geometry txt file.
        :return: a 4-dimensional list (block, vertex, coordinate, axis)
        """
        block_vertices = []
        gp_read_count = 0
        # For each block
        for i, block_txt in enumerate(self.txt[gfs[2]]):
            vertices = []
            while len(vertices) < 8:
                gp_txt = self.txt[gfs[0]][gp_read_count].split()  # Get the split information of one single line of grid point description
                vertex = [float(j) for j in gp_txt[2:5]]  # Select the coordinates and convert the scientific notation
                vertices.append(vertex)
                gp_read_count += 1
            vertices = fc.transform_vertex_3DEC(vertices)
            block_vertices.append(vertices)

        return block_vertices


class GridpointReader:
    def __init__(self, file_dir):
        self.df = pd.read_csv(file_dir, sep=" ")
        num_block = int(self.df.max(axis=0)[grid_point_reader['ID_label']])  # Read the number of blocks
        self.block_list = []
        for i in range(1, num_block + 1):
            # Get the information of all grid points belong to a single block
            df_block = pd.DataFrame(self.df.loc[self.df[grid_point_reader['ID_label']] == i].reset_index(drop=True))
            # Pick the grid points at vertices - ordered
            df_vertices = pd.DataFrame(fc.get_vert_df(df_block)).reset_index(drop=True)
            # Read the displacements and positions of all vertices
            vert_disp = df_vertices[grid_point_reader['Disp_labels']].values
            vert_pos = df_vertices[grid_point_reader['Position_labels']].values
            gp_pos_new = []
            for pos, disp in zip(vert_pos, vert_disp):
                # Calculate the final position and append it to the list
                gp_pos_new.append(list(np.add(np.array(pos), np.array(disp))))
            self.block_list.append(Rb.RBlock.build(vertices=gp_pos_new))
            # self.block_list.append(BlockDraw(vertices=gp_pos_new))

    def draw(self): 
        Func.draw_blocks(self.block_list)


def create_block_gps(gps):
    """
    This function creates a block object based on the grid point coordinates
    :param gps: list of all grid point coordinates
    :return: a block object defined by the grid points
    """
    maxs, mins = fc.find_lims_from_gps(gps)
    Dims = []
    for lim_upper, lim_lower in zip(maxs, mins):
        Dims.append(lim_upper - lim_lower)
    Dims = np.array(Dims)
    CoM = np.array(mins) + Dims*0.5
    return Block(CoM, Dims)


def create_type2_row(num_x, block_dims):
    Br = BlockRow(start_point=origin, num_blocks=num_x-1, Block_Dims=block_dims)

    half_dims = np.array(block_dims)
    half_dims[0] = half_dims[0]*0.5
    Br.add_block('left', 1, half_dims)
    Br.add_block('right', 1, half_dims)
    Br.move(np.array([half_dims[0], 0, 0]))
    return Br


if __name__ == '__main__':
    # num_blocks = brick_dim_num
    # sample_wall_dims = np.multiply(num_blocks, np.array(brick_dims_UK))
    # W1 = BlockWall(sample_wall_dims)
    # print(f'Number of rows: {len(W1)}')
    # W1.draw()

    # filename = 'geo_info2.txt'
    # a = GeometryReader(filename)
    # sample_bg = BlockGroup(a.get_block_vertices())
    # sample_bg.draw()
    a = 'C:\\Users\\\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test3\\Gp_info.txt'
    sample_gp = GridpointReader(a)
    sample_gp.draw()

