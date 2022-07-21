""" Initialize eeVR blender plugin """

# pylint: disable=import-error

if "bpy" in locals():
    import importlib
    importlib.reload(VRRenderer)
else:
    from . import VRRenderer

from .VRRenderer import Properties, RenderImage, RenderAnimation, RenderToolsPanel, VRRendererCancel

import bpy

bl_info = {
    "name": "eeVR",
    "description": "Render in different projections using Eevee engine",
    "author": "EternalTrail, SAMtak",
    "version": (0, 2, 0),
    "blender": (3, 0, 0),
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
