from furniture.rug import RugFactory

CLS = RugFactory
DOCUMENTATION = '\nParameters for RugFactory\n\nBuilds a procedural rug. Width/length in meters (scene scale).\n\nRequired\n--------\nwidth : float\n    Width of the rug (meters).\n\nOptional\n--------\nlength : float\n    Length (default = width). For circle, length is forced to width.\nrug_shape : str\n    "rectangle", "circle", "rounded", or "ellipse". Default "rectangle".\nrounded_buffer : float\n    Bevel width for "rounded" shape. Default width * 0.25.\nthickness : float\n    Rug thickness. Default 0.015.\nsurface\n    Material instance. Omit for default fabric.\n'
PARAM_OPTS = {
    "a": {"width": 2.0, "length": 3.0, "rug_shape": "rectangle"},
    "b": {"width": 2.5, "length": 2.5, "rug_shape": "circle"},
    "c": {"width": 2.0, "length": 3.5, "rug_shape": "rounded", "rounded_buffer": 0.5},
}
