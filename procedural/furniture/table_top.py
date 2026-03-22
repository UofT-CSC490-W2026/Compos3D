import bpy
from infinigen.assets.objects.tables.table_top import geometry_generate_table_top_wrapper
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory

class TableTopFactory(AssetFactory):

    def __init__(self, params):
        super().__init__(0, coarse=False)
        self.params = {'Profile N-gon': int(params.get('Profile N-gon', 4)), 'Profile Width': float(params.get('Profile Width', 1.0)), 'Profile Aspect Ratio': float(params.get('Profile Aspect Ratio', 1.0)), 'Profile Fillet Ratio': float(params.get('Profile Fillet Ratio', 0.2)), 'Thickness': float(params.get('Thickness', 0.1)), 'Vertical Fillet Ratio': float(params.get('Vertical Fillet Ratio', 0.2))}

    def create_asset(self, **kwargs):
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object
        surface.add_geomod(obj, geometry_generate_table_top_wrapper, apply=False, input_kwargs=self.params)
        return obj

    def finalize_assets(self, assets):
        pass
