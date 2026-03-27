import bpy
from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.table_decorations.vase import geometry_vases
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


class VaseFactory(AssetFactory):
    def __init__(self, params):
        super().__init__(0, coarse=False)
        self.params = dict(params)
        if "dimensions" in self.params:
            x, y, z = self.params["dimensions"]
            self.params.setdefault("Diameter", x)
            self.params.setdefault("Height", z)
        if "Material" not in self.params:
            self.params["Material"] = weighted_sample(
                material_assignments.marble + material_assignments.tableware
            )()()
        elif isinstance(self.params["Material"], str):
            mat_class = weighted_sample(
                material_assignments.marble + material_assignments.tableware
            )
            self.params["Material"] = mat_class()()

    def create_asset(self, **kwargs) -> bpy.types.Object:
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object
        surface.add_geomod(obj, geometry_vases, apply=True, input_kwargs=self.params)
        butil.modify_mesh(obj, "SOLIDIFY", apply=True, thickness=0.002)
        butil.modify_mesh(obj, "SUBSURF", apply=True, levels=2, render_levels=2)
        return obj

    def finalize_assets(self, assets):
        pass
