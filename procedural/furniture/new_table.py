import bpy
from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.tables.legs.single_stand import nodegroup_generate_single_stand
from infinigen.assets.objects.tables.legs.square import nodegroup_generate_leg_square
from infinigen.assets.objects.tables.legs.straight import nodegroup_generate_leg_straight
from infinigen.assets.objects.tables.strechers import nodegroup_strecher
from infinigen.assets.objects.tables.table_top import nodegroup_generate_table_top
from infinigen.assets.objects.tables.table_utils import nodegroup_create_anchors, nodegroup_create_legs_and_strechers
from infinigen.core import surface, tagging
from infinigen.core import tags as t
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import NoApply
from infinigen.core.util.random import weighted_sample

@node_utils.to_nodegroup('geometry_create_legs', singleton=False, type='GeometryNodeTree')
def geometry_create_legs(nw: NodeWrangler, **kwargs):
    createanchors = nw.new_node(nodegroup_create_anchors().name, input_kwargs={'Profile N-gon': kwargs['Leg Number'], 'Profile Width': kwargs['Leg Placement Top Relative Scale'] * kwargs['Top Profile Width'], 'Profile Aspect Ratio': kwargs['Top Profile Aspect Ratio']})
    if kwargs['Leg Style'] == 'single_stand':
        leg = nw.new_node(nodegroup_generate_single_stand(**kwargs).name, input_kwargs={'Leg Height': kwargs['Leg Height'], 'Leg Diameter': kwargs['Leg Diameter'], 'Resolution': 64})
        leg = nw.new_node(nodegroup_create_legs_and_strechers().name, input_kwargs={'Anchors': createanchors, 'Keep Legs': True, 'Leg Instance': leg, 'Table Height': kwargs['Top Height'], 'Leg Bottom Relative Scale': kwargs['Leg Placement Bottom Relative Scale'], 'Align Leg X rot': True})
    elif kwargs['Leg Style'] == 'straight':
        leg = nw.new_node(nodegroup_generate_leg_straight(**kwargs).name, input_kwargs={'Leg Height': kwargs['Leg Height'], 'Leg Diameter': kwargs['Leg Diameter'], 'Resolution': 32, 'N-gon': kwargs['Leg NGon'], 'Fillet Ratio': 0.1})
        strecher = nw.new_node(nodegroup_strecher().name, input_kwargs={'Profile Width': kwargs['Leg Diameter'] * 0.5})
        leg = nw.new_node(nodegroup_create_legs_and_strechers().name, input_kwargs={'Anchors': createanchors, 'Keep Legs': True, 'Leg Instance': leg, 'Table Height': kwargs['Top Height'], 'Strecher Instance': strecher, 'Strecher Index Increment': kwargs['Strecher Increament'], 'Strecher Relative Position': kwargs['Strecher Relative Pos'], 'Leg Bottom Relative Scale': kwargs['Leg Placement Bottom Relative Scale'], 'Align Leg X rot': True})
    elif kwargs['Leg Style'] == 'square':
        leg = nw.new_node(nodegroup_generate_leg_square(**kwargs).name, input_kwargs={'Height': kwargs['Leg Height'], 'Width': 0.707 * kwargs['Leg Placement Top Relative Scale'] * kwargs['Top Profile Width'] * kwargs['Top Profile Aspect Ratio'], 'Has Bottom Connector': kwargs['Strecher Increament'] > 0, 'Profile Width': kwargs['Leg Diameter']})
        leg = nw.new_node(nodegroup_create_legs_and_strechers().name, input_kwargs={'Anchors': createanchors, 'Keep Legs': True, 'Leg Instance': leg, 'Table Height': kwargs['Top Height'], 'Leg Bottom Relative Scale': kwargs['Leg Placement Bottom Relative Scale'], 'Align Leg X rot': True})
    else:
        raise NotImplementedError
    leg = nw.new_node(Nodes.SetMaterial, input_kwargs={'Geometry': leg, 'Material': kwargs['LegMaterial']})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': leg}, attrs={'is_active_output': True})

def geometry_assemble_table(nw: NodeWrangler, **kwargs):
    generatetabletop = nw.new_node(nodegroup_generate_table_top().name, input_kwargs={'Thickness': kwargs['Top Thickness'], 'N-gon': kwargs['Top Profile N-gon'], 'Profile Width': kwargs['Top Profile Width'], 'Aspect Ratio': kwargs['Top Profile Aspect Ratio'], 'Fillet Ratio': kwargs['Top Profile Fillet Ratio'], 'Fillet Radius Vertical': kwargs['Top Vertical Fillet Ratio']})
    tabletop_instance = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': generatetabletop, 'Translation': (0.0, 0.0, kwargs['Top Height'])})
    tabletop_instance = nw.new_node(Nodes.SetMaterial, input_kwargs={'Geometry': tabletop_instance, 'Material': kwargs['TopMaterial']})
    legs = nw.new_node(geometry_create_legs(**kwargs).name)
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [tabletop_instance, legs]})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry}, attrs={'is_active_output': True})

class TableDiningFactory(AssetFactory):

    def __init__(self, params):
        super(TableDiningFactory, self).__init__(0, coarse=False)
        self.clothes_scatter = NoApply()
        top_material = material_assignments.table_top[1][0]()()
        leg_material = material_assignments.tableware[0][0]()()
        x, y, z = params['dimensions']
        def_params = {'Top Profile N-gon': 4, 'Leg NGon': 4, 'Top Profile Width': 1.414 * x, 'Top Profile Aspect Ratio': y / x, 'Height': z, 'Top Height': z - params['Top Thickness'], 'Leg Height': 1.0, 'TopMaterial': top_material, 'LegMaterial': leg_material}
        params.update(def_params)
        self.params = params

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object
        surface.add_geomod(obj, geometry_assemble_table, apply=True, input_kwargs=self.params)
        tagging.tag_system.relabel_obj(obj)
        assert tagging.tagged_face_mask(obj, {t.Subpart.SupportSurface}).sum() != 0
        return obj
