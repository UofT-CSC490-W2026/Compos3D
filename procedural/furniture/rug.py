import bpy
import numpy as np
from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.object import new_base_circle, new_plane
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


class RugFactory(AssetFactory):
    def __init__(self, params):
        super().__init__(0, coarse=False)
        self.width = float(params["width"])
        self.length = float(params.get("length", self.width))
        self.rug_shape = params.get("rug_shape", "rectangle")
        if self.rug_shape == "circle":
            self.length = self.width
        self.rounded_buffer = float(params.get("rounded_buffer", self.width * 0.25))
        self.thickness = float(params.get("thickness", 0.015))
        self.surface = params.get("surface")
        if self.surface is None:
            self.surface = material_assignments.fabrics[0][0]()()

    def build_shape(self):
        if self.rug_shape == "rectangle":
            obj = new_plane()
            obj.scale = (self.length / 2, self.width / 2, 1)
            butil.apply_transform(obj, True)
        elif self.rug_shape == "rounded":
            obj = new_plane()
            obj.scale = (self.length / 2, self.width / 2, 1)
            butil.apply_transform(obj, True)
            butil.modify_mesh(obj, "BEVEL", width=self.rounded_buffer, segments=16)
        else:
            obj = new_base_circle(vertices=128)
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.edge_face_add()
            obj.scale = (self.length / 2, self.width / 2, 1)
            butil.apply_transform(obj, True)
        return obj

    def create_asset(self, **kwargs) -> bpy.types.Object:
        obj = self.build_shape()
        wrap_sides(obj, self.surface, "z", "x", "y")
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=1)
        return obj
