import bpy
import numpy as np
from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.decorate import (
    read_co,
    read_edge_center,
    read_edge_direction,
    remove_edges,
    remove_vertices,
    select_edges,
    solidify,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.draw import align_bezier, bezier_curve
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import NoApply
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj


def _tuple2(x, default=(0, 0)):
    if x is None:
        return default
    if hasattr(x, "__iter__") and len(x) >= 2:
        return (float(x[0]), float(x[1]))
    return default


def _back_profile(x, default=None):
    if default is None:
        default = [(0, 1)]
    if x is None or (isinstance(x, (list, tuple)) and len(x) == 0):
        return default
    out = []
    for item in x:
        if hasattr(item, "__iter__") and len(item) >= 2:
            out.append((float(item[0]), float(item[1])))
    return out if out else default


class ChairFactory(AssetFactory):
    def __init__(self, params):
        super().__init__(0, coarse=False)
        p = params
        self.width = float(p.get("width", 0.45))
        self.size = float(p.get("size", 0.42))
        self.thickness = float(p.get("thickness", 0.06))
        self.bevel_width = float(p.get("bevel_width", self.thickness * 0.3))
        self.seat_back = float(p.get("seat_back", 0.9))
        self.seat_mid = float(p.get("seat_mid", 0.75))
        self.seat_mid_x = float(p.get("seat_mid_x", 0.95))
        self.seat_mid_z = float(p.get("seat_mid_z", 0.25))
        self.seat_front = float(p.get("seat_front", 1.1))
        self.is_seat_round = bool(p.get("is_seat_round", True))
        self.is_seat_subsurf = bool(p.get("is_seat_subsurf", False))
        self.leg_thickness = float(p.get("leg_thickness", 0.05))
        self.limb_profile = float(p.get("limb_profile", 2.0))
        self.leg_height = float(p.get("leg_height", 0.47))
        self.back_height = float(p.get("back_height", 0.45))
        self.is_leg_round = bool(p.get("is_leg_round", True))
        self.leg_type = str(p.get("leg_type", "straight"))
        self.leg_x_offset = float(p.get("leg_x_offset", 0))
        self.leg_y_offset = _tuple2(p.get("leg_y_offset"), (0, 0))
        self.back_x_offset = float(p.get("back_x_offset", 0))
        self.back_y_offset = float(p.get("back_y_offset", 0.15))
        self.has_leg_x_bar = bool(p.get("has_leg_x_bar", True))
        self.has_leg_y_bar = bool(p.get("has_leg_y_bar", True))
        self.leg_bar_z_ratio = float(p.get("leg_bar_z_ratio", 0.5))
        self.has_arm = bool(p.get("has_arm", True))
        self.arm_thickness = float(p.get("arm_thickness", 0.05))
        self.arm_height = float(p.get("arm_height", self.arm_thickness * 0.8))
        self.arm_y = float(p.get("arm_y", 0.9 * self.size))
        self.arm_z = float(p.get("arm_z", 0.45 * self.back_height))
        arm_mid = p.get("arm_mid")
        if arm_mid is not None and hasattr(arm_mid, "__iter__") and (len(arm_mid) >= 3):
            self.arm_mid = np.array(
                [float(arm_mid[0]), float(arm_mid[1]), float(arm_mid[2])]
            )
        else:
            self.arm_mid = np.array([0.0, 0.03, -0.03])
        arm_profile = p.get("arm_profile")
        if (
            arm_profile is not None
            and hasattr(arm_profile, "__iter__")
            and (len(arm_profile) >= 2)
        ):
            self.arm_profile = [float(arm_profile[0]), float(arm_profile[1])]
        else:
            self.arm_profile = [0.5, 1.5]
        self.back_thickness = float(p.get("back_thickness", 0.045))
        self.back_type = str(p.get("back_type", "whole"))
        self.back_profile = _back_profile(p.get("back_profile"))
        self.back_vertical_cuts = int(p.get("back_vertical_cuts", 2))
        self.back_partial_scale = float(p.get("back_partial_scale", 1.2))
        self.back_bridge_smoothness = float(p.get("back_bridge_smoothness", 0.5))
        self.back_bridge_profile_shape_factor = float(
            p.get("back_bridge_profile_shape_factor", 0.2)
        )
        self.clothes_scatter = NoApply()
        limb_gen = p.get("limb_surface")
        if limb_gen is None:
            self.limb_surface_material_gen = material_assignments.furniture_leg[0][0]()
        else:
            self.limb_surface_material_gen = (
                limb_gen if callable(limb_gen) else lambda: limb_gen
            )
        surface_gen = p.get("surface")
        if surface_gen is None:
            self.surface_material_gen = material_assignments.furniture_hard_surface[0][
                0
            ]()
        else:
            self.surface_material_gen = (
                surface_gen if callable(surface_gen) else lambda: surface_gen
            )
        panel_gen = p.get("panel_surface")
        if panel_gen is None:
            self.panel_surface_material_gen = (
                material_assignments.furniture_hard_surface[0][0]()
            )
        else:
            self.panel_surface_material_gen = (
                panel_gen if callable(panel_gen) else lambda: panel_gen
            )

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        obj = new_bbox(
            -self.width / 2 - max(self.leg_x_offset, self.back_x_offset),
            self.width / 2 + max(self.leg_x_offset, self.back_x_offset),
            -self.size - self.leg_y_offset[1] - self.leg_thickness * 0.5,
            max(self.leg_y_offset[0], self.back_y_offset),
            -self.leg_height,
            self.back_height * 1.2,
        )
        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)
        return obj

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()
        self.panel_surface = self.panel_surface_material_gen()
        self.limb_surface = self.limb_surface_material_gen()
        obj = self.make_seat()
        legs = self.make_legs()
        backs = self.make_backs()
        parts = [obj] + legs + backs
        parts.extend(self.make_leg_decors(legs))
        if self.has_arm:
            parts.extend(self.make_arms(obj, backs))
        parts.extend(self.make_back_decors(backs))
        for o in legs:
            self.solidify(o, 2)
        for o in backs:
            self.solidify(o, 2, self.back_thickness)
        obj = join_objects(parts)
        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)
        surface.assign_material(obj, self.surface)
        surface.assign_material(obj, self.panel_surface, selection="panel")
        surface.assign_material(obj, self.limb_surface, selection="limb")
        return obj

    def finalize_assets(self, assets):
        pass

    def make_seat(self):
        x_anchors = (
            np.array([0, 0.1, 1, self.seat_mid_x, self.seat_back, 0]) * self.width / 2
        )
        y_anchors = (
            np.array([-self.seat_front, -self.seat_front, -1, -self.seat_mid, 0, 0])
            * self.size
        )
        z_anchors = np.array([0, 0, 0, self.seat_mid_z, 0, 0]) * self.thickness
        vector_locations = [4] if self.is_seat_round else [2, 4]
        obj = bezier_curve((x_anchors, y_anchors, z_anchors), vector_locations)
        butil.modify_mesh(obj, "MIRROR")
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_grid(use_interp_simple=True)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        subsurf(obj, 1, not self.is_seat_subsurf)
        butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        return obj

    def make_legs(self):
        leg_starts = np.array(
            [[-self.seat_back, 0, 0], [-1, -1, 0], [1, -1, 0], [self.seat_back, 0, 0]]
        ) * np.array([[self.width / 2, self.size, 0]])
        leg_ends = leg_starts.copy()
        leg_ends[[0, 1], 0] -= self.leg_x_offset
        leg_ends[[2, 3], 0] += self.leg_x_offset
        leg_ends[[0, 3], 1] += self.leg_y_offset[0]
        leg_ends[[1, 2], 1] -= self.leg_y_offset[1]
        leg_ends[:, -1] = -self.leg_height
        return self.make_limb(leg_ends, leg_starts)

    def make_limb(self, leg_ends, leg_starts):
        limbs = []
        for leg_start, leg_end in zip(leg_starts, leg_ends):
            if self.leg_type == "up-curved":
                axes = [(0, 0, 1), None]
                scale = [self.limb_profile, 1]
            elif self.leg_type == "down-curved":
                axes = [None, (0, 0, 1)]
                scale = [1, self.limb_profile]
            else:
                axes = None
                scale = None
            limb = align_bezier(np.stack([leg_start, leg_end], -1), axes, scale)
            limb.location = (
                np.array(
                    [
                        1 if leg_start[0] < 0 else -1,
                        1 if leg_start[1] < -self.size / 2 else -1,
                        0,
                    ]
                )
                * self.leg_thickness
                / 2
            )
            butil.apply_transform(limb, True)
            limbs.append(limb)
        return limbs

    def make_backs(self):
        back_starts = (
            np.array([[-self.seat_back, 0, 0], [self.seat_back, 0, 0]]) * self.width / 2
        )
        back_ends = back_starts.copy()
        back_ends[:, 0] += np.array([self.back_x_offset, -self.back_x_offset])
        back_ends[:, 1] = self.back_y_offset
        back_ends[:, 2] = self.back_height
        return self.make_limb(back_starts, back_ends)

    def make_leg_decors(self, legs):
        decors = []
        z_height = -self.leg_height * self.leg_bar_z_ratio
        if self.has_leg_x_bar:
            locs = []
            for leg in legs:
                co = read_co(leg)
                locs.append(co[np.argmin(np.abs(co[:, -1] - z_height))])
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[0], locs[3]], -1)), 0)
            )
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[1], locs[2]], -1)), 0)
            )
        if self.has_leg_y_bar:
            locs = []
            for leg in legs:
                co = read_co(leg)
                locs.append(co[np.argmin(np.abs(co[:, -1] - z_height))])
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[0], locs[1]], -1)), 1)
            )
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[2], locs[3]], -1)), 1)
            )
        for d in decors:
            write_attribute(d, 1, "limb", "FACE")
        return decors

    def make_back_decors(self, backs, finalize=True):
        obj = join_objects([deep_clone_obj(b) for b in backs])
        x, y, z = read_co(obj).T
        x += np.where(x > 0, self.back_thickness / 2, -self.back_thickness / 2)
        write_co(obj, np.stack([x, y, z], -1))
        smoothness = self.back_bridge_smoothness
        profile_shape_factor = self.back_bridge_profile_shape_factor
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            center = read_edge_center(obj)
            for z_min, z_max in self.back_profile:
                select_edges(
                    obj,
                    (z_min * self.back_height <= center[:, -1])
                    & (center[:, -1] <= z_max * self.back_height),
                )
                bpy.ops.mesh.bridge_edge_loops(
                    number_cuts=64,
                    interpolation="LINEAR",
                    smoothness=smoothness,
                    profile_shape_factor=profile_shape_factor,
                )
            bpy.ops.mesh.select_loose()
            bpy.ops.mesh.delete()
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=np.minimum(self.thickness, self.back_thickness),
            offset=0,
        )
        if finalize:
            butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        parts = [obj]
        if self.back_type == "vertical-bar":
            other = join_objects([deep_clone_obj(b) for b in backs])
            with butil.ViewportMode(other, "EDIT"):
                bpy.ops.mesh.select_mode(type="EDGE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.bridge_edge_loops(
                    number_cuts=self.back_vertical_cuts,
                    interpolation="LINEAR",
                    smoothness=smoothness,
                    profile_shape_factor=profile_shape_factor,
                )
                bpy.ops.mesh.select_all(action="INVERT")
                bpy.ops.mesh.delete()
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.delete(type="ONLY_FACE")
            remove_edges(other, np.abs(read_edge_direction(other)[:, -1]) < 0.5)
            remove_vertices(other, lambda x, y, z: z < -self.thickness / 2)
            remove_vertices(
                other,
                lambda x, y, z: (
                    z
                    > (self.back_profile[0][0] + self.back_profile[0][1])
                    * self.back_height
                    / 2
                ),
            )
            parts.append(self.solidify(other, 2, self.back_thickness))
        elif self.back_type == "partial":
            co = read_co(obj)
            co[:, 1] *= self.back_partial_scale
            write_co(obj, co)
        for p in parts:
            write_attribute(p, 1, "panel", "FACE")
        return parts

    def make_arms(self, base, backs):
        co = read_co(base)
        end = co[np.argmin(co[:, 0] - (np.abs(co[:, 1] + self.arm_y) < 0.02))]
        end[0] += self.arm_thickness / 4
        end_ = end.copy()
        end_[0] = -end[0]
        arms = []
        co = read_co(backs[0])
        start = co[np.argmin(co[:, 0] - (np.abs(co[:, -1] - self.arm_z) < 0.02))]
        start[0] -= self.arm_thickness / 4
        start_ = start.copy()
        start_[0] = -start[0]
        for start_pt, end_pt in zip([start, start_], [end, end_]):
            mid = np.array(
                [
                    end_pt[0] + self.arm_mid[0] * (-1 if end_pt[0] > 0 else 1),
                    end_pt[1] + self.arm_mid[1],
                    start_pt[2] + self.arm_mid[2],
                ]
            )
            arm = align_bezier(
                np.stack([start_pt, mid, end_pt], -1),
                np.array(
                    [
                        [end_pt[0] - start_pt[0], end_pt[1] - start_pt[1], 0],
                        [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                        [0, 0, 1],
                    ]
                ),
                [1, *self.arm_profile, 1],
            )
            if self.is_leg_round:
                surface.add_geomod(
                    arm,
                    geo_radius,
                    apply=True,
                    input_args=[self.arm_thickness / 2, 32],
                    input_kwargs={"to_align_tilt": False},
                )
            else:
                with butil.ViewportMode(arm, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.extrude_edges_move(
                        TRANSFORM_OT_translate={
                            "value": (
                                self.arm_thickness
                                if end_pt[0] < 0
                                else -self.arm_thickness,
                                0,
                                0,
                            )
                        }
                    )
                butil.modify_mesh(arm, "SOLIDIFY", thickness=self.arm_height, offset=0)
            write_attribute(arm, 1, "limb", "FACE")
            arms.append(arm)
        return arms

    def solidify(self, obj, axis, thickness=None):
        if thickness is None:
            thickness = self.leg_thickness
        if self.is_leg_round:
            solidify(obj, axis, thickness)
            butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        else:
            surface.add_geomod(
                obj, geo_radius, apply=True, input_args=[thickness / 2, 32]
            )
        write_attribute(obj, 1, "limb", "FACE")
        return obj
