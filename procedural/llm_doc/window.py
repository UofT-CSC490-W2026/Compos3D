from infinigen.assets.kenny.window import WindowFactory

CLS = WindowFactory
DOCUMENTATION = "\nParameters for WindowFactory\n\nBuilds a procedural window (frame, panels, optional curtain/shutter). Meters.\n\nGeometry\n--------\nWidth, Height : float\n    Overall size. Defaults 2, 2.\nFrameWidth, FrameThickness : float\nSubFrameWidth, SubFrameThickness : float\nPanelHAmount, PanelVAmount : int\nSubPanelHAmount, SubPanelVAmount : int\nGlassThickness : float\nOpenHAngle, OpenVAngle, OpenOffset, OEOffset : float\nCurtain : bool\nCurtainFrameDepth, CurtainDepth, CurtainIntervalNumber : float\nCurtainFrameRadius, CurtainMidL, CurtainMidR : float\nShutter : bool\nShutterPanelRadius, ShutterWidth, ShutterThickness : float\nShutterRotation, ShutterInterval : float\n\nMaterials (optional)\n--------------------\nFrameMaterial, CurtainFrameMaterial, CurtainMaterial, Material\n"
PARAM_OPTS = {
    "a": {"Width": 2.0, "Height": 2.0},
    "b": {"Width": 1.5, "Height": 1.5, "Shutter": True},
    "c": {"Width": 2.5, "Height": 2.0, "Curtain": True},
}
