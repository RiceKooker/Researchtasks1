import BlockClass as bc
from Blender import WavefrontOBJ as wf

if __name__ == '__main__':
    block1 = bc.Block.dim_build([1, 1, 1])
    print(block1.vertices)
    block1.rot(angles=[0, 45, 0])