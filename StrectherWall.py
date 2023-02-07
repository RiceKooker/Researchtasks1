from Classes import BlockWall
from utils import get_velocities, BricksBoundary, stop_detection
import model_specs as ms

wall = BlockWall(ms.wall_dim)
brick_commands = wall.three_DEC_create()
boundary_bricks = BricksBoundary(vertices=wall.find_vertices())
velocities = get_velocities(ms.displacements, ms.velocity_max, ms.wall_dim)
stop_axis_index, stop_axis = stop_detection(ms.displacements)

model_creation = f"""
model new
model large-strain {ms.large_strain}
block mech damp local {ms.damping}
""" + brick_commands + boundary_bricks.commands + f"""block export filename '{ms.geometry_file_name1}'"""

boundary_conditions = f"""
block face triangulate radial-8
;block face triangulate edge-max 0.01  ;Use this for finer discretization
block group 'BOT' range pos-z {boundary_bricks.z_lims[0][0]} {boundary_bricks.z_lims[0][1]}
block group 'TOP' range pos-z {boundary_bricks.z_lims[1][0]} {boundary_bricks.z_lims[1][1]}

;BOUNDARY CONDITION
block fix range group 'BOT'
"""

material_propeterties = f"""
;MATERAIL PROPERTIES
block property density {ms.brick_density}

;CONTACT PROPERTIES
block contact generate-subcontacts
block contact jmodel assign {ms.contact_model}
block contact property stiffness-normal {ms.normal_stiffness} stiffness-shear {ms.shear_stiffness} cohesion {ms.cohesion} tension {ms.tension} fric {ms.friction} fric-res {ms.fric_res}
block contact property stiffness-normal {ms.boundary_spec['normal_stiffness']} stiffness-shear {ms.boundary_spec['shear_stiffness']} cohesion {ms.boundary_spec['cohesion']} tension {ms.boundary_spec['tension']} fric {ms.boundary_spec['friction']} fric-res {ms.boundary_spec['fric_res']} range group-intersection '{ms.group_name_top}' '{ms.group_name_body}'
block contact property stiffness-normal {ms.boundary_spec['normal_stiffness']} stiffness-shear {ms.boundary_spec['shear_stiffness']} cohesion {ms.boundary_spec['cohesion']} tension {ms.boundary_spec['tension']} fric {ms.boundary_spec['friction']} fric-res {ms.boundary_spec['fric_res']} range group-intersection '{ms.group_name_bot}' '{ms.group_name_body}'
block contact material-table default property stiffness-normal {ms.normal_stiffness} stiffness-shear {ms.shear_stiffness} cohesion 0 fric {ms.friction} ten 0 fric-res {ms.friction}

;Select face for vertical pressure
block face group 'TOPFACE' range pos-z {boundary_bricks.z_top[1]}
"""


loadings = f"""
;Apply gravity
model gravity 0 0 -9.81
model solve

;Apply vertical load to settle the model.
block face apply stress-zz {ms.vertical_pressure} range group 'TOPFACE'  ; Select the top face.
model solve
block fix range group 'TOP'

;Apply the prescribed displacement as a velocity
;Prescribed displacement: [{ms.load_descriptions[0]:.2f}%, {ms.load_descriptions[1]:.2f}%, 0] of wall height
;Prescribed rotations: [{ms.load_descriptions[3]:.3f}, {ms.load_descriptions[4]:.3f}, {ms.load_descriptions[5]:.3f}] in degrees
fish define psc_dis
    top_gp = block.gp.near({boundary_bricks.top_gp[0]},{boundary_bricks.top_gp[1]},{boundary_bricks.top_gp[2]})
    top_dis = math.abs(block.gp.disp.{stop_axis}(top_gp))
    command
        block apply velocity-x {velocities[0]} range group 'TOP'
        block apply velocity-y {velocities[1]} range group 'TOP'
        block apply velocity-z {velocities[2]} range group 'TOP'
        block initialize rvelocity-x {velocities[3]}  range group 'TOP'
        block initialize rvelocity-y {velocities[4]}  range group 'TOP'
        block initialize rvelocity-z {velocities[5]}  range group 'TOP'
    endcommand
    loop while top_dis < {abs(ms.displacements[stop_axis_index])}
        command 
            model cycle {ms.num_cycle}
        endcommand
        top_dis = math.abs(block.gp.disp.{stop_axis}(top_gp))
    endloop
    ii = io.out('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    ii = io.out('Prescribed displacement is reached!')
    ii = io.out('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    command
        [dump]
    endcommand
end
[psc_dis]

model save '{ms.file_name}'
"""

function_definition = """
; Definition of necessary functions used
fish def dump 
    Gp_information = list
    Column_names = list
    Column_names('end') = 'Block_ID Grid_point_ID Pos_x Pos_y Pos_z Disp_x Disp_y Disp_z'
    loop foreach bgpp block.gp.list
         blp = block.gp.hostblock(bgpp)
         block_i = block.id(blp)
         gp_i = block.gp.id(bgpp)
         pos = block.gp.pos(bgpp)
         disp = block.gp.disp(bgpp)
         Gp_information('end') = string(block_i) + ' ' +string(gp_i) + ' ' ...
          + string(pos->x)+' '+string(pos->y)+' '+string(pos->z) + ' ' ...
          + string(disp->x)+' '+string(disp->y)+' '+string(disp->z)
    end_loop                               
    file.open('Position2','write','text')
    file.write(Column_names)
    file.write(Gp_information)            
    file.close       
end 
"""


script = model_creation + boundary_conditions + function_definition + material_propeterties + loadings

if __name__ == '__main__':
    print(script)

