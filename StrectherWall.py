from Classes import ThreeDECScript, BlockWall
from utils import generate_dimension, create_boundary_bricks
import model_specs as ms

wall = BlockWall(generate_dimension())
brick_commands = wall.three_DEC_create()
boundary_brick_commands, z_lims = create_boundary_bricks(wall.find_vertices())

model_creation = f"""
model new
model config dynamic ; What does this mean? Is this necessary?
model large-strain {ms.large_strain}
block mech damp local {ms.damping}
""" + brick_commands + boundary_brick_commands

boundary_conditions = f"""
block face triangulate radial-8
block group 'BASE' range pos-z {z_lims[0][0]} {z_lims[0][1]}
block group 'TOP' range pos-z {z_lims[1][0]} {z_lims[1][1]}
;BOUNDARY CONDITION
block fix range group 'BASE'
block fix rotation range group 'TOP'"""

material_propeterties = f"""
;MATERAIL PROPERTIES
block property density {ms.brick_density}
block property density {ms.brick_density} range group 'TOP'

;CONTACT PROPERTIES
block contact generate-subcontacts
block contact jmodel assign {ms.contact_model}
block contact property stiffness-normal {ms.normal_stiffness} stiffness-shear {ms.shear_stiffness} cohesion {ms.cohesion} tension {ms.tension} fric {ms.friction} fric-res {ms.fric_res}"""

loadings = """;MONITORING POINT
block history displacement-x gridpoint 8

model gravity 0 -9.81 0

fish define stopit
    maxdis = 0
    loop foreach gp block.gp.list
       maxdis = math.max(maxdis,math.mag(block.gp.dis(gp)))
    endloop
    if maxdis > 0.04 ;(Indicate limit in (m))
       system.error = 'Displacement exceeded'
    endif
end
[stopit]

;only call the function every 1000 steps
fish callback add stopit 1 interval 1000

model solve

block gridpoint apply  force-y -244.89796 range pos-y 2.05 3.00
model solve

; Loading
block apply velocity-x -0.001 range group 'TOP'

;block contact group-subcontact 'Reaction' range pos-y -0.01 +0.01
;fish def Base_N
;    Base_normal=0
;    loop foreach bscp block.subcontact.list
;        if block.subcontact.group(bscp)='Reaction'
;            Base_normal=Base_normal+block.subcontact.force.norm(bscp)
;        endif
;    end_loop
;    Base_N=Base_normal
;end
;fish history Base_N
;@Base_N
;
;fish def Base_S
;    Base_shear=0
;    loop foreach bscp block.subcontact.list
;        if block.subcontact.group(bscp)='Reaction'
;            Base_shear=Base_shear+block.subcontact.force.shear.x(bscp)
;        endif
;    end_loop
;    Base_S=Base_shear
;end
;fish history Base_S
;@Base_S

model cycle 300000

model save 'V2CONTACT030English05Fixed'
"""
script = model_creation + boundary_conditions + material_propeterties

