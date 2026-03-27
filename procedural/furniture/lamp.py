import bpy
from infinigen.assets.composition import material_assignments
from infinigen.assets.lighting.indoor_lights import PointLampFactory
from infinigen.assets.objects.lamp.lamp import nodegroup_lamp_geometry
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


def _ensure_vector(v):
    if hasattr(v, "__iter__") and len(v) >= 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    return (0.0, 0.0, 0.0)


class LampFactory(AssetFactory):
    def __init__(self, params):
        super().__init__(0, coarse=False)
        self.params = {
            "StandRadius": float(params.get("StandRadius", 0.01)),
            "BaseRadius": float(params.get("BaseRadius", 0.1)),
            "BaseHeight": float(params.get("BaseHeight", 0.02)),
            "ShadeHeight": float(params.get("ShadeHeight", 0.2)),
            "HeadTopRadius": float(params.get("HeadTopRadius", 0.1)),
            "HeadBotRadius": float(params.get("HeadBotRadius", 0.12)),
            "ReverseLamp": bool(params.get("ReverseLamp", True)),
            "RackThickness": float(params.get("RackThickness", 0.002)),
            "CurvePoint1": _ensure_vector(params.get("CurvePoint1", (0.0, 0.0, 0.0))),
            "CurvePoint2": _ensure_vector(params.get("CurvePoint2", (0.0, 0.0, 0.2))),
            "CurvePoint3": _ensure_vector(params.get("CurvePoint3", (0.0, 0.0, 0.3))),
        }
        lampshade_gen = params.get("lampshade_material")
        metal_gen = params.get("metal_material")
        if lampshade_gen is None:
            lampshade_gen = material_assignments.lampshade[0][0]()
        if metal_gen is None:
            metal_gen = material_assignments.furniture_leg[0][0]()
        self.params["BlackMaterial"] = metal_gen() if callable(metal_gen) else metal_gen
        self.params["MetalMaterial"] = metal_gen() if callable(metal_gen) else metal_gen
        self.params["LampshadeMaterial"] = (
            lampshade_gen() if callable(lampshade_gen) else lampshade_gen
        )
        self.add_bulb = bool(params.get("add_bulb", True))
        self.bulb_fac = PointLampFactory(0)
        self.bulb_fac.params["Temperature"] = max(
            self.bulb_fac.params["Temperature"] * 0.6, 2500
        )
        self.bulb_fac.params["Wattage"] *= 0.5

    def create_asset(self, i=0, **kwargs) -> bpy.types.Object:
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_lamp_geometry(),
            ng_inputs=self.params,
            apply=True,
        )
        if self.add_bulb:
            bulb = self.bulb_fac.spawn_asset(i)
            butil.parent_to(bulb, obj, no_inverse=True, no_transform=True)
            bulb.location.z = obj.bound_box[-2][2] - self.params["ShadeHeight"] * 0.5
        with butil.SelectObjects(obj):
            bpy.ops.object.shade_flat()
        return obj

    def finalize_assets(self, assets):
        pass
