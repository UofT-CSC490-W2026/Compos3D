from furniture.table_top import TableTopFactory
CLS = TableTopFactory
DOCUMENTATION = '\nParameters for TableTopFactory\n\nBuilds a procedural table top / shelf surface (rounded rectangle profile). Meters.\n\nParams\n------\nProfile N-gon : int\n    Default 4.\nProfile Width : float\n    Default 1.0.\nProfile Aspect Ratio : float\n    Default 1.0.\nProfile Fillet Ratio : float\n    Default 0.2.\nThickness : float\n    Default 0.1.\nVertical Fillet Ratio : float\n    Default 0.2.\n'
PARAM_OPTS = {'a': {'Profile Width': 1.0, 'Thickness': 0.1}, 'b': {'Profile Width': 0.8, 'Thickness': 0.06}, 'c': {'Profile Width': 1.2, 'Thickness': 0.12, 'Profile Fillet Ratio': 0.1}}
