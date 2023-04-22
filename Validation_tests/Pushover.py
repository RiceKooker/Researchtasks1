from EnglishBondBlock import English_wall
from utils import get_velocities, BricksBoundary, stop_detection, print_list, print_location
import model_specs as ms

wall = English_wall
brick_commands = wall.three_DEC_create()
boundary_bricks = BricksBoundary(vertices=wall.find_vertices(), thickness=0.8)
# velocities = get_velocities(ms.displacements, ms.velocity_max, ms.wall_dim)
# stop_axis_index, stop_axis = stop_detection(ms.displacements)

model_creation = f"""
model new
model configure plugins 
model large-strain {ms.large_strain}
block mech damp local {ms.damping}
""" + brick_commands + boundary_bricks.commands

boundary_conditions = f"""
;block face triangulate radial-8
block face triangulate edge-max 0.05  ;Use this for finer discretization
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
block contact property stiffness-normal {ms.normal_stiffness} stiffness-shear {ms.shear_stiffness} ten {ms.tension} cohesion {ms.cohesion} fric {ms.friction:.8f} fric-residual {ms.fric_res:.8f} cohesion-residual {ms.cohesion_res} Gf_tension {ms.Gf_tension} Gf_shear {ms.Gf_shear} Gc_compression {ms.Gf_compression} fc {ms.fc}
block contact property stiffness-normal {ms.boundary_spec['normal_stiffness']} stiffness-shear {ms.boundary_spec['shear_stiffness']} ten {ms.boundary_spec['tension']} cohesion {ms.boundary_spec['cohesion']} fric {ms.boundary_spec['friction']:.8f} fric-residual {ms.boundary_spec['fric_res']:.8f} cohesion-residual {ms.boundary_spec['cohesion_res']} Gf_tension {ms.boundary_spec['Gf_tension']} Gf_shear {ms.boundary_spec['Gf_shear']} Gc_compression {ms.boundary_spec['Gf_compression']} fc {ms.boundary_spec['fc']} range group-intersection '{ms.group_name_top}' '{ms.group_name_body}'
block contact property stiffness-normal {ms.boundary_spec['normal_stiffness']} stiffness-shear {ms.boundary_spec['shear_stiffness']} ten {ms.boundary_spec['tension']} cohesion {ms.boundary_spec['cohesion']} fric {ms.boundary_spec['friction']:.8f} fric-residual {ms.boundary_spec['fric_res']:.8f} cohesion-residual {ms.boundary_spec['cohesion_res']} Gf_tension {ms.boundary_spec['Gf_tension']} Gf_shear {ms.boundary_spec['Gf_shear']} Gc_compression {ms.boundary_spec['Gf_compression']} fc {ms.boundary_spec['fc']} range group-intersection '{ms.group_name_bot}' '{ms.group_name_body}'

;Select face for vertical pressure
block face group 'TOPFACE' range pos-z {boundary_bricks.z_top[1]:.8f}
"""

loadings = f"""
;Record history
block history displacement-{ms.cyclic_specs['load_direction']} position {print_location(boundary_bricks.top_gp)} 
[Reaction_normal]
[Reaction_shear]
;Apply gravity
model gravity 0 0 -9.81
model solve

;Apply vertical load to settle the model.
block face apply stress-zz {ms.vertical_pressure} range group 'TOPFACE'  ; Select the top face.
model solve
block fix range group 'TOP'

;Apply loading
[test]

model save '{ms.file_name}'
"""

dump_function = f"""
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
"""

# prescribe_displacement_test = f"""
# ;Apply the prescribed displacement as a velocity
# ;Prescribed displacement: [{ms.load_descriptions[0]:.2f}%, {ms.load_descriptions[1]:.2f}%, 0] of wall height
# ;Prescribed rotations: [{ms.load_descriptions[3]:.3f}, {ms.load_descriptions[4]:.3f}, {ms.load_descriptions[5]:.3f}] in degrees
# fish define test
#     top_gp = block.gp.near({print_list(boundary_bricks.top_gp)})
#     top_dis = math.abs(block.gp.disp.{stop_axis}(top_gp))
#     command
#         block apply velocity-x {velocities[0]} range group 'TOP'
#         block apply velocity-y {velocities[1]} range group 'TOP'
#         block apply velocity-z {velocities[2]} range group 'TOP'
#         block initialize rvelocity-x {velocities[3]}  range group 'TOP'
#         block initialize rvelocity-y {velocities[4]}  range group 'TOP'
#         block initialize rvelocity-z {velocities[5]}  range group 'TOP'
#     endcommand
#     loop while top_dis < {abs(ms.displacements[stop_axis_index])}
#         command
#             model cycle {ms.num_cycle}
#         endcommand
#         top_dis = math.abs(block.gp.disp.{stop_axis}(top_gp))
#     endloop
#     ii = io.out('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     ii = io.out('Prescribed displacement is reached!')
#     ii = io.out('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     command
#         [dump]
#     endcommand
# end"""

cyclic_test = f"""
;Cyclic test function
fish define test
    top_gp = block.gp.near({print_list(boundary_bricks.top_gp)})
    top_dis = math.abs(block.gp.disp.{ms.cyclic_specs['load_direction']}(top_gp))
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


pushover_test = f"""
;Pushover test function
fish define test
    top_gp = block.gp.near({print_list(boundary_bricks.top_gp)})
    top_dis = math.abs(block.gp.disp.{ms.push_over_specs['load_direction']}(top_gp))
    cycle_n = {ms.push_over_specs['n_cycle']}
    velocity = {ms.push_over_specs['velocity']}
    block_group = '{ms.push_over_specs['load_group']}'
    amp = {ms.push_over_specs['amp']}
    command
        block apply velocity-{ms.cyclic_specs['load_direction']} [velocity] range group [block_group]
    endcommand
    ;Keep cycling until the amplitude is reached
    loop while top_dis < amp
        command 
            model cycle [cycle_n]
        endcommand
        top_dis = math.abs(block.gp.disp.{ms.cyclic_specs['load_direction']}(top_gp))
    endloop
    command
        [dump]
    endcommand
end
"""

fish_history = f"""
;History fish functions
block contact group-subcontact 'Top_boundary' range pos-z {boundary_bricks.z_top[0]:.8f}
fish def Reaction_shear
    top_shear = 0
    loop foreach bscp block.subcontact.list
        if block.subcontact.group(bscp) = 'Top_boundary'
            top_shear = top_shear + block.subcontact.force.shear.x(bscp)
        endif
    end_loop
    Reaction_shear = top_shear
end
fish history Reaction_shear


fish def Reaction_normal
    top_normal = 0
    loop foreach bscp block.subcontact.list
        if block.subcontact.group(bscp) = 'Top_boundary'
            top_normal = top_normal + block.subcontact.force.norm(bscp)
        endif
    end_loop
    Reaction_normal = top_normal
end
fish history Reaction_normal
"""

function_definition = dump_function + pushover_test + fish_history

script = model_creation + boundary_conditions + material_propeterties + function_definition + loadings

if __name__ == '__main__':
    print(script)
    # with open('C:\\Users\\dgian\\Documents\\Itasca\\3dec700\\My Projects\\Validation test\\Pushover.dat', 'w') as text_file:
    #     text_file.write(script)