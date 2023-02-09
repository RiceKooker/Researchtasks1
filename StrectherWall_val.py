from Classes import BlockWall
from utils import get_velocities, BricksBoundary, stop_detection, print_list, print_location
import model_specs_val as ms

wall = BlockWall(ms.wall_dim)
brick_commands = wall.three_DEC_create()
boundary_bricks = BricksBoundary(vertices=wall.find_vertices())
velocities = get_velocities(ms.displacements, ms.velocity_max, ms.wall_dim)
stop_axis_index, stop_axis = stop_detection(ms.displacements)

model_creation = f"""
model new
model configure plugins 
model large-strain {ms.large_strain}
block mech damp local {ms.damping}
""" + brick_commands + boundary_bricks.commands + f"""block export filename '{ms.geometry_file_name1}'"""

boundary_conditions = f"""
block face triangulate radial-8
;block face triangulate edge-max 0.01  ;Use this for finer discretization
block group 'BOT' range pos-z {boundary_bricks.z_lims[0][0]:.8f} {boundary_bricks.z_lims[0][1]:.8f}
block group 'TOP' range pos-z {boundary_bricks.z_lims[1][0]:.8f} {boundary_bricks.z_lims[1][1]:.8f}

;BOUNDARY CONDITION
block fix range group 'BOT'
"""

material_propeterties = f"""
;MATERAIL PROPERTIES
block property density {ms.brick_density}

;CONTACT PROPERTIES
block contact generate-subcontacts
block contact jmodel assign {ms.contact_model}
block contact property stiffness-normal {ms.normal_stiffness} stiffness-shear {ms.shear_stiffness} ten {ms.tension} cohesion {ms.cohesion} fric {ms.friction} fric-residual {ms.fric_res} cohesion-residual {ms.cohesion_res} Gf_tension {ms.Gf_tension} Gf_shear {ms.Gf_shear} Gc_compression {ms.Gf_compression} fc {ms.fc}
block contact property stiffness-normal {ms.boundary_spec['normal_stiffness']} stiffness-shear {ms.boundary_spec['shear_stiffness']} ten {ms.boundary_spec['tension']} cohesion {ms.boundary_spec['cohesion']} fric {ms.boundary_spec['friction']} fric-residual {ms.boundary_spec['fric_res']} cohesion-residual {ms.boundary_spec['cohesion_res']} Gf_tension {ms.boundary_spec['Gf_tension']} Gf_shear {ms.boundary_spec['Gf_shear']} Gc_compression {ms.boundary_spec['Gf_compression']} fc {ms.boundary_spec['fc']} range group-intersection '{ms.group_name_top}' '{ms.group_name_body}'
block contact property stiffness-normal {ms.boundary_spec['normal_stiffness']} stiffness-shear {ms.boundary_spec['shear_stiffness']} ten {ms.boundary_spec['tension']} cohesion {ms.boundary_spec['cohesion']} fric {ms.boundary_spec['friction']} fric-residual {ms.boundary_spec['fric_res']} cohesion-residual {ms.boundary_spec['cohesion_res']} Gf_tension {ms.boundary_spec['Gf_tension']} Gf_shear {ms.boundary_spec['Gf_shear']} Gc_compression {ms.boundary_spec['Gf_compression']} fc {ms.boundary_spec['fc']} range group-intersection '{ms.group_name_bot}' '{ms.group_name_body}'

;Select face for vertical pressure
block face group 'TOPFACE' range pos-z {boundary_bricks.z_top[1]:.8f}
"""


loadings = f"""
;Apply gravity
model gravity 0 0 -9.81
model solve

;Apply vertical load to settle the model.
block face apply stress-zz {ms.vertical_pressure} range group 'TOPFACE'  ; Select the top face.
model solve
block fix range group 'TOP'

block history displacement-{ms.cyclic_specs['load_direction']} position {print_location(boundary_bricks.top_gp)} 

;Apply loading
[cyclic_test]

model save '{ms.file_name}'
"""

function_definition = f"""
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
    file.open('{ms.gp_file_name}','write','text')
    file.write(Column_names)
    file.write(Gp_information)            
    file.close       
end 


;Apply the prescribed displacement as a velocity
;Prescribed displacement: [{ms.load_descriptions[0]:.2f}%, {ms.load_descriptions[1]:.2f}%, 0] of wall height
;Prescribed rotations: [{ms.load_descriptions[3]:.3f}, {ms.load_descriptions[4]:.3f}, {ms.load_descriptions[5]:.3f}] in degrees
fish define psc_dis
    top_gp = block.gp.near({print_list(boundary_bricks.top_gp)})
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


;Cyclic test function
fish define cyclic_test
    top_gp = block.gp.near({print_list(boundary_bricks.top_gp)})
    top_dis = math.abs(block.gp.disp.y(top_gp))
    disp_amp = list.seq({print_list(ms.cyclic_specs['disp_amps'])})
    cycle_n = {ms.cyclic_specs['n_cycle']}
    velocity = {ms.cyclic_specs['velocity']}
    block_group = '{ms.cyclic_specs['load_group']}'
    v_sign = 1
    
    loop foreach amp disp_amp
        ;Initialize the velocity and immediately cycle it without checking.
        ;This assumes that the limit in displacement is not exceeded by the first cycle_n cycles
        command
            block apply velocity-{ms.cyclic_specs['load_direction']} [velocity * v_sign] range group [block_group]
            model cycle [cycle_n]
        endcommand
        top_dis = math.abs(block.gp.disp.{ms.cyclic_specs['load_direction']}(top_gp))
        ;Keep cycling until the amplitude is reached
        loop while top_dis < amp
            command 
                model cycle [cycle_n]
            endcommand
            top_dis = math.abs(block.gp.disp.{ms.cyclic_specs['load_direction']}(top_gp))
        endloop
        v_sign = -v_sign
    end_loop
    ; Return the block to the zero displacement location
    top_dis = block.gp.disp.{ms.cyclic_specs['load_direction']}(top_gp)
    command
        block apply velocity-{ms.cyclic_specs['load_direction']} [velocity * v_sign] range group [block_group]
        model cycle [cycle_n]
    endcommand
    loop while top_dis < 0
        command 
            model cycle [cycle_n]
        endcommand
        top_dis = block.gp.disp.{ms.cyclic_specs['load_direction']}(top_gp)
    endloop
    command
        [dump]
    endcommand
end
"""


script = model_creation + boundary_conditions + material_propeterties + function_definition+ loadings

if __name__ == '__main__':
    print(script)
    with open('C:\\Users\\dgian\\Documents\\Itasca\\3dec700\\My Projects\\DIANA_val.dat', 'w') as text_file:
        text_file.write(script)



