from Classes import BlockWall
from utils import get_velocities, BricksBoundary
import model_specs as ms

wall = BlockWall(ms.wall_dim)
brick_commands = wall.three_DEC_create()
boundary_bricks = BricksBoundary(vertices=wall.find_vertices())
velocities = get_velocities(ms.displacements, ms.rotations, ms.velocity_max)

model_creation = f"""
model new
model config dynamic ; What does this mean? Is this necessary?
model large-strain {ms.large_strain}
block mech damp local {ms.damping}
""" + brick_commands + boundary_bricks.commands + f"""block export filename '{ms.geometry_file_name1}'"""

boundary_conditions = f"""
block face triangulate radial-8
block group 'BASE' range pos-z {boundary_bricks.z_lims[0][0]} {boundary_bricks.z_lims[0][1]}
block group 'TOP' range pos-z {boundary_bricks.z_lims[1][0]} {boundary_bricks.z_lims[1][1]}

;BOUNDARY CONDITION
block fix range group 'BASE'
block fix range group 'TOP'
"""

material_propeterties = f"""
;MATERAIL PROPERTIES
block property density {ms.brick_density}

;CONTACT PROPERTIES
block contact generate-subcontacts
block contact jmodel assign {ms.contact_model}
block contact property stiffness-normal {ms.normal_stiffness} stiffness-shear {ms.shear_stiffness} cohesion {ms.cohesion} tension {ms.tension} fric {ms.friction} fric-res {ms.fric_res}
block contact material-table default property stiffness-normal 1e9 stiffness-shear .4e9 cohesion 0 fric 35 ten 0 fric-res 35
"""


loadings = f""";MONITORING POINT
model gravity 0 0 -9.81
model solve

fish define psc_dis
    top_gp = block.gp.near({boundary_bricks.top_gp[0]},{boundary_bricks.top_gp[1]},{boundary_bricks.top_gp[2]})
    top_dis = math.abs(block.gp.disp.x(top_gp))
    command
        block apply velocity-x {velocities[0]} range group 'TOP'
        block apply velocity-y {velocities[1]} range group 'TOP'
        block apply velocity-z {velocities[2]} range group 'TOP'
        block initialize rvelocity-x {velocities[3]}  range group 'TOP'
        block initialize rvelocity-y {velocities[4]}  range group 'TOP'
        block initialize rvelocity-z {velocities[5]}  range group 'TOP'
    endcommand
    loop while top_dis < {ms.displacements[0]}
        command 
            model cycle {1000}
        endcommand
        top_dis = math.abs(block.gp.disp.x(top_gp))
    endloop
    ii = io.out('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    ii = io.out('Prescribed displacement is reached!')
    ii = io.out('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    command
        block export filename '{ms.geometry_file_name2}'
    endcommand
end
[psc_dis]

; Loading
model save '{ms.file_name}'
"""


script = model_creation + boundary_conditions + material_propeterties + loadings

if __name__ == '__main__':
    print(script)

