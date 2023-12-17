from DisplacementMapping import RBlockClass as rB
import Func

if __name__ == '__main__':
    sample_block1 = rB.RBlock.dim_build([1, 3, 2])
    sample_block1.rot([30, 45, -15])
    Func.draw_blocks4([sample_block1], highlight_points=sample_block1.generate_surface_points(4))

