import BlockClass as bc
from Blender.const import face_vertex_index as f_indices
from Classes import GridpointReader
import os


def transform_vertex_order(vertices):
    """
    This function transforms the vertex numbering into the convention used in the obj cube example.
    :param vertices:
    :return:
    """
    temp = vertices.copy()
    temp[0] = vertices[5]
    temp[1] = vertices[1]
    temp[2] = vertices[2]
    temp[3] = vertices[6]
    temp[4] = vertices[4]
    temp[5] = vertices[0]
    temp[6] = vertices[3]
    temp[7] = vertices[7]
    return temp


def save_obj(block_list, filename):
    object_command = ''

    with open(filename, 'w') as f:
        for block_index, block in enumerate(block_list):
            vertices = transform_vertex_order(block.vertices)
            for vertex in vertices:
                f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            f.write('\n')
            object_command += f'g Block{block_index+1}\n'
            for i_face in f_indices:
                v_i_start = block_index*len(vertices)
                i_face = [a+v_i_start for a in i_face]
                object_command += f'f {i_face[0]} {i_face[1]} {i_face[2]} {i_face[3]}\n'
            object_command += '\n'
        f.write(object_command)


def save_obj_sep(block_list, filename):
    os.mkdir(filename)
    for block_index, block in enumerate(block_list):
        filename_each = f'{filename}\\Block_{block_index}.obj'
        with open(filename_each, 'w') as f:
            object_command = ''
            vertices = transform_vertex_order(block.vertices)
            for vertex in vertices:
                f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            f.write('\n')
            object_command += f'g Block{1}\n'
            for i_face in f_indices:
                v_i_start = 0
                i_face = [a+v_i_start for a in i_face]
                object_command += f'f {i_face[0]} {i_face[1]} {i_face[2]} {i_face[3]}\n'
            object_command += '\n'
            f.write(object_command)




if __name__ == '__main__':
    # Example: Generate obj files by manually generating blocks.
    # block1 = bc.Block.dim_build([5, 2, 10])
    # block1.move([3, 0, 0])
    # block2 = block1.duplicate(vec=[10, 0, 0])
    # block3 = block2.duplicate(vec=[0, 0, 5])
    # bg1 = bc.BlockGroup([block1, block2])
    # bg1.draw()
    # save_obj(sample_gp.block_list, 'sample3.obj')   # Some modifications needed here.
    # ---------------------------------------------------------------------------------------------------------------------------------

    # Generate obj files from geometry files generated from 3DEC.
    geo_file_dir = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\DELFT\\Long wall\\Cyclic\\2\\Undeformed\\Gp_info.txt'
    saved_file_dir= 'TUD_COMP-4-2-undeformed-test'
    grid_point_info = GridpointReader(geo_file_dir)
    save_obj_sep(grid_point_info.block_list, saved_file_dir)
    # ---------------------------------------------------------------------------------------------------------------------------------
