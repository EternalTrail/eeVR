""" Initialize eeVR blender plugin """

# pylint: disable=import-error

if "bpy" in locals():
    import importlib
    importlib.reload(VRRenderer)
else:
    from . import VRRenderer

from .VRRenderer import RenderImage, RenderAnimation, RenderToolsPanel, VRRendererCancel

import bpy

# eeVR's properties
class Properties(bpy.types.PropertyGroup):

    renderModeEnum : bpy.props.EnumProperty(
        items=[
            ("EQUI", "Equirectangular", "Renders in equirectangular projection"),
            ("DOME", "Full Dome", "Renders in full dome projection"),
        ],
        default="EQUI",
        name="Mode",
    )

    domeModeEnum : bpy.props.EnumProperty(
        items=[
            ("0", "Equidistant (VTA)", "Renders in equidistant dome projection"),
            ("1", "Hemispherical (VTH)", "Renders in hemispherical dome projection"),
            ("2", "Equisolid", "Renders in equisolid dome projection"),
            ("3", "Stereographic", "Renders in Stereographic dome projection"),
        ],
        default="0",
        name="Method",
    )

    renderFOV : bpy.props.FloatProperty(
        180.0,
        default=180.0,
        name="FOV",
        min=180,
        max=360,
        description="Field of view in degrees",
    )

    cancelVRRenderer : bpy.props.BoolProperty(
        name="Cancel", default=True
    )

    @classmethod
    def register(cls):
        """ Register eeVR's properties to Blender """
        bpy.types.Scene.eeVR = bpy.props.PointerProperty(type=cls)

    @classmethod
    def unregister(cls):
        """ Unregister eeVR's properties from Blender """
        del bpy.types.Scene.eeVR


bl_info = {
    "name": "eeVR",
    "description": "Render in different projections using Eevee engine",
    "author": "EternalTrail",
    "version": (0, 1),
    "blender": (2, 82, 7),
    "location": "View3D > UI",
    "warning": "This addon is still in early alpha, may break your blend file!",
    "wiki_url": "https://github.com/EternalTrail/eeVR",
    "tracker_url": "https://github.com/EternalTrail/eeVR/issues",
    "support": "TESTING",
    "category": "Render",
}

# REGISTER
register, unregister = bpy.utils.register_classes_factory((
    Properties,
    RenderToolsPanel,
    RenderImage,
    RenderAnimation,
    VRRendererCancel,
))
