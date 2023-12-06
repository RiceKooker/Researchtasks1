import Classes
import Func
from Func import find_axis_and_direction, find_lims_from_points, draw_blocks, rotate_3d_point
import numpy as np
from itertools import count

class Block:
    block_id = count(0)

    def __init__(self, x_lim, y_lim, z_lim):
        self.vertices = [[x_lim[0], y_lim[0], z_lim[0]], [x_lim[0], y_lim[0], z_lim[1]], [x_lim[1], y_lim[0], z_lim[1]],
                    [x_lim[1], y_lim[0], z_lim[0]], [x_lim[0], y_lim[1], z_lim[0]], [x_lim[0], y_lim[1], z_lim[1]],
                    [x_lim[1], y_lim[1], z_lim[1]], [x_lim[1], y_lim[1], z_lim[0]]]
        self.block_id = next(self.block_id)

    @classmethod
    def build(cls, vertices):
        """
        Create a block by the vertices
        :param vertices:
        :return:
        """
        temp = cls([0, 1], [0, 1], [0, 1])
        temp.vertices = vertices.copy()
        return temp

    @classmethod
    def dim_build(cls, dims, com=None):
        """
        Create a block by the dimensions and the start position
        :param dims:
        :param com:
        :return:
        """
        if com is None:
            com = [0.5*i for i in dims]
        vertices = Func.find_vertices(com, dims)
        return cls.build(vertices)

    def get_dims(self):
        """
        Get the dimensions of the block
        :return:
        """
        x_lim = [self.vertices[0][0], self.vertices[3][0]]
        y_lim = [self.vertices[0][1], self.vertices[4][1]]
        z_lim = [self.vertices[0][2], self.vertices[1][2]]
        return [x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]]

    def get_com(self):
        dims = self.get_dims()
        return [a+0.5*b for a, b in zip(self.vertices[0], dims)]

    def get_lims(self):
        x_lim = [self.vertices[0][0], self.vertices[3][0]]
        y_lim = [self.vertices[0][1], self.vertices[4][1]]
        z_lim = [self.vertices[0][2], self.vertices[1][2]]
        return x_lim, y_lim, z_lim

    def copy(self):
        """
        Make a copy of the block itself
        :return:
        """
        return Block.build(self.vertices)

    def move(self, vec):
        """
        Move the block with a displacement vector
        :param vec: displacement vector
        :return:
        """
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = [a+b for a, b in zip(vertex, vec)]

    def rot(self, angles, center='centroid'):
        if center == 'centroid':
            center = self.get_com()
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = rotate_3d_point(vertex, angles, origin=center)

    def duplicate(self, side=None, new_dims=None, vec=None):
        """
        Create a new block with potentially different dimensions. Place the block to one side of the self block.
        Or move the block with a displacement vector
        :param side: can be one of 'top', 'bot', 'front', 'back', 'left', 'right'
        :param new_dims:
        :param vec:
        :return:
        """
        if new_dims is None:
            new_dims = self.get_dims()
        else:
            new_dims = new_dims
        block_new = Block.dim_build(new_dims, com=self.get_com())
        if vec is not None:
            block_new.move(vec)
            return block_new
        axis, direction = find_axis_and_direction(side)
        disp_vec = [0, 0, 0]
        disp_vec[axis] = direction * 0.5 * (self.get_dims()[axis] + new_dims[axis])
        block_new.move(disp_vec)
        return block_new

    def draw(self):
        Func.draw_blocks2([self])

    def create_crack_surfaces(self, axis):
        """
        This generates 3DEC commands which can create two blocks representing the original block. An extra surface will be generated.
        :param axis: one of [0, 1, 2] representing [x, y, z], the extra surface generated will have a normal parallel to the axis specified.
        :return:
        """
        x_lim, y_lim, z_lim = self.get_lims()
        lims = [x_lim, y_lim, z_lim]
        lim_modified = lims[axis]
        lim1 = lims.copy()
        lim2 = lims.copy()
        mid_value = sum(lim_modified)/len(lim_modified)
        lim1[axis] = [lim_modified[0], mid_value]
        lim2[axis] = [mid_value, lim_modified[1]]
        return f"block create brick {lim1[0][0]:.10f} {lim1[0][1]:.10f} {lim1[1][0]:.10f} {lim1[1][1]:.10f} {lim1[2][0]:.10f} {lim1[2][1]:.10f} group 'GP{self.block_id}_1' \n" \
               f"block create brick {lim2[0][0]:.10f} {lim2[0][1]:.10f} {lim2[1][0]:.10f} {lim2[1][1]:.10f} {lim2[2][0]:.10f} {lim2[2][1]:.10f} group 'GP{self.block_id}_2' \n"

    def three_DEC_create(self):
        x_lim, y_lim, z_lim = self.get_lims()
        return f"block create brick {x_lim[0]:.10f} {x_lim[1]:.10f} {y_lim[0]:.10f} {y_lim[1]:.10f} {z_lim[0]:.10f} {z_lim[1]:.10f} group 'GP{self.block_id}' \n"



# class Block(Classes.Block):
#     def duplicate(self, side=None, new_Dims=None, disp_vec=None):
#         if new_Dims is None:
#             new_Dims = self.Dims
#         block_new = Block(CoM=self.CoM, Dims=new_Dims)
#         if disp_vec is not None and side is None:
#             block_new.move(disp_vec)
#             return block_new
#         disp_vec = [0, 0, 0]
#         axis, direction = find_axis_and_direction(side)
#         disp_vec[axis] = direction*0.5*(self.Dims[axis] + new_Dims[axis])
#         block_new.move(disp_vec)
#         return block_new


class BlockGroup:
    def __init__(self, blocks):
        if isinstance(blocks, Block):
            self.block_list = [blocks.copy()]
        else:
            self.block_list = blocks.copy()

    def copy(self):
        block_list = []
        for block in self.block_list:
            block_list.append(Block.build(block.vertices))
        return BlockGroup(block_list)

    def add(self, *other_blocks):
        temp = self.copy()
        for other_block in other_blocks:
            if isinstance(other_block, BlockGroup):
                temp.block_list += other_block.block_list.copy()
            else:
                temp.block_list.append(other_block.copy())
        return temp

    def move(self, dis_vec):
        for i, block in enumerate(self.block_list):
            self.block_list[i].move(dis_vec)

    def duplicate(self, side, times=1, disp_vec_in=None):
        if disp_vec_in is None:
            disp_vec_in = [0, 0, 0]
        new_list = []
        for i in range(times):
            mins, maxs = find_lims_from_points(self.block_list[0].vertices)
            for block in self.block_list:
                vert_test = list(block.vertices)
                vert_test.append(mins)
                vert_test.append(maxs)
                mins, maxs = find_lims_from_points(vert_test)
            dims_envelop = []
            for min_v, max_v in zip(mins, maxs):
                dims_envelop.append(max_v - min_v)
            disp_vec = [0, 0, 0]
            axis, direction = find_axis_and_direction(side)
            disp_vec[axis] = direction * dims_envelop[axis]
            disp_vec = [a+b for a, b in zip(disp_vec, disp_vec_in)]
            disp_vec = [a*(i+1) for a in disp_vec]
            for block in self.block_list:
                new_list.append(block.duplicate(side=side, vec=disp_vec))
        return BlockGroup(new_list)

    def expand(self, side, times=1, disp_vec=None):
        self.block_list += self.duplicate(side, times, disp_vec).block_list

    def draw(self):
        Func.draw_blocks2(block_list=self.block_list)

    def three_DEC_create(self):
        total_command = ''
        for block in self.block_list:
            total_command += block.three_DEC_create()
        return total_command

    def three_DEC_create_crack_surface(self, axis=0):
        total_command = ''
        for block in self.block_list:
            total_command += block.create_crack_surfaces(axis)
        return total_command

    def get_id_list(self):
        id_list = []
        for block in self.block_list:
            id_list.append(f'GP{block.block_id}')
        return id_list

    def find_vertices(self):
        """
        This function returns the vertices of the wall following the same ordering condition as blocks.
        :return:
        """
        max_dims = np.zeros((3,))
        min_dims = 100000*np.ones((3,))
        for block in self.block_list:
            for vertex in block.vertices:
                for axis, axis_value in enumerate(vertex):
                    if axis_value < min_dims[axis]:
                        min_dims[axis] = axis_value
                    if axis_value > max_dims[axis]:
                        max_dims[axis] = axis_value
        return Func.find_vertices_from_max(min_dims, max_dims)


if __name__ == '__main__':
    a = Block.dim_build([1, 1, 1])
    b = Block.dim_build([1, 1, 1])
    print(b.block_id)








