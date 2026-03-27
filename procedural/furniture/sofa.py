import bpy
from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.seating.sofa import nodegroup_sofa_geometry
from infinigen.core import surface, tagging
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil

ARM_TYPE_SQUARE = 0
ARM_TYPE_ROUND = 1
ARM_TYPE_ANGULAR = 2


def _default_sofa_params():
    return {
        "Dimensions": (1.0, 2.0, 0.85),
        "Arm Dimensions": (1.0, 0.1, 0.65),
        "Back Dimensions": (0.2, 0.0, 0.65),
        "Seat Dimensions": (1.0, 0.9, 0.22),
        "Foot Dimensions": (0.15, 0.06, 0.06),
        "Baseboard Height": 0.07,
        "Backrest Width": 0.15,
        "Seat Margin": 0.985,
        "Backrest Angle": -0.3,
        "arm_width": 0.75,
        "Arm Type": ARM_TYPE_SQUARE,
        "Arm_height": 0.85,
        "arms_angle": 0.5,
        "Footrest": False,
        "Count": 4,
        "Scaling footrest": 1.45,
        "Reflection": 0,
        "leg_type": False,
        "leg_dimensions": 0.6,
        "leg_z": 1.5,
        "leg_faces": 16,
        "Subdivide": True,
    }


class SofaFactory(AssetFactory):
    def __init__(self, params):
        super().__init__(0, coarse=False)
        self.params = dict(_default_sofa_params())
        for k, v in params.items():
            if k in self.params or k in _default_sofa_params():
                self.params[k] = v
        fabric_arg = params.get("sofa_fabric")
        if fabric_arg is None:
            self.sofa_fabric = material_assignments.fabrics[0][0]()()
        else:
            self.sofa_fabric = fabric_arg() if callable(fabric_arg) else fabric_arg

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_sofa_geometry(),
            ng_inputs=self.params,
            apply=True,
        )
        tagging.tag_system.relabel_obj(obj)
        surface.assign_material(obj, self.sofa_fabric)
        return obj

    def create_asset(self, i=0, placeholder=None, **kwargs) -> bpy.types.Object:
        if placeholder is None:
            placeholder = self.create_placeholder()
        hipoly = butil.copy(placeholder, keep_materials=True)
        butil.modify_mesh(hipoly, "SUBSURF", levels=1, apply=True)
        with butil.SelectObjects(hipoly):
            bpy.ops.object.shade_smooth()
        return hipoly

    def finalize_assets(self, assets):
        pass
