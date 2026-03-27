from furniture.chair import ChairFactory

CLS = ChairFactory
DOCUMENTATION = '\nParameters for ChairFactory\n\nBuilds a procedural chair (seat, legs, back, optional arms and leg bars).\nAll linear dimensions in meters. Many params are normalized or booleans.\n\nSeat\n----\nwidth, size : float\n    Seat footprint (x, y). Defaults 0.45, 0.42.\nthickness : float\n    Seat thickness. Default 0.06.\nbevel_width : float\n    Edge bevel. Default thickness * 0.3.\nseat_back, seat_mid, seat_mid_x, seat_mid_z, seat_front : float\n    Seat profile control (0--1.2 range). Defaults define a typical shape.\nis_seat_round : bool\n    Round vs angular seat front. Default True.\nis_seat_subsurf : bool\n    Subsurf on seat. Default False.\n\nLegs\n----\nleg_thickness : float\n    Leg cross-section. Default 0.05.\nlimb_profile : float\n    Curve profile for up/down-curved legs. Default 2.\nleg_height : float\n    Leg length. Default 0.47.\nleg_type : str\n    "vertical", "straight", "up-curved", "down-curved". Default "straight".\nis_leg_round : bool\n    Round (tube) vs square legs. Default True.\nleg_x_offset, leg_y_offset : float, (float, float)\n    Leg splay. leg_y_offset is (front, back). Defaults 0, (0,0).\nhas_leg_x_bar, has_leg_y_bar : bool\n    Horizontal bars between legs. Default True.\nleg_bar_z_ratio : float\n    Height of bars as fraction of leg_height (0--1). Default 0.5.\n\nBack\n----\nback_height : float\n    Back height. Default 0.45.\nback_thickness : float\n    Back panel thickness. Default 0.045.\nback_type : str\n    "whole", "partial", "horizontal-bar", "vertical-bar". Default "whole".\nback_profile : list[(float, float)]\n    Per-segment (z_min, z_max) normalized 0--1 for bridge. Default [(0,1)].\nback_vertical_cuts : int\n    For vertical-bar. Default 2.\nback_partial_scale : float\n    For partial. Default 1.2.\nback_x_offset, back_y_offset : float\n    Back splay. Default 0, 0.15.\nback_bridge_smoothness : float\n    Bridge smoothness 0--1. Default 0.5.\nback_bridge_profile_shape_factor : float\n    Bridge profile 0--0.4. Default 0.2.\n\nArms\n----\nhas_arm : bool\n    Default True.\narm_thickness, arm_height : float\n    Default 0.05, 0.04.\narm_y, arm_z : float\n    Position (y along seat, z height). Defaults 0.9*size, 0.45*back_height.\narm_mid : (float, float, float)\n    Arm curve control. Default (0, 0.03, -0.03).\narm_profile : (float, float)\n    Curve profile. Default (0.5, 1.5).\n\nMaterials (optional)\n--------------------\nlimb_surface, surface, panel_surface\n    Material generators (callable). Omit for defaults (furniture_leg, furniture_hard_surface).\n'
chair_whole_params = {
    "width": 0.45,
    "size": 0.42,
    "thickness": 0.06,
    "bevel_width": 0.018,
    "seat_back": 0.9,
    "seat_mid": 0.75,
    "seat_mid_x": 0.95,
    "seat_mid_z": 0.25,
    "seat_front": 1.1,
    "is_seat_round": True,
    "is_seat_subsurf": False,
    "leg_thickness": 0.05,
    "limb_profile": 2.0,
    "leg_height": 0.47,
    "back_height": 0.45,
    "is_leg_round": True,
    "leg_type": "straight",
    "leg_x_offset": 0,
    "leg_y_offset": (0, 0),
    "back_x_offset": 0,
    "back_y_offset": 0.15,
    "has_leg_x_bar": True,
    "has_leg_y_bar": True,
    "leg_bar_z_ratio": 0.5,
    "has_arm": True,
    "arm_thickness": 0.05,
    "arm_height": 0.04,
    "back_thickness": 0.045,
    "back_type": "whole",
    "back_profile": [(0, 1)],
}
chair_simple_params = {**chair_whole_params, "has_arm": False, "leg_type": "vertical"}
chair_partial_params = {
    **chair_whole_params,
    "back_type": "partial",
    "back_profile": [(0.5, 1)],
    "leg_type": "up-curved",
    "limb_profile": 2.0,
}
PARAM_OPTS = {
    "a": chair_whole_params,
    "b": chair_simple_params,
    "c": chair_partial_params,
}
