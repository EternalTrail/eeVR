""" Initialize eeVR blender plugin """

# pylint: disable=import-error

from datetime import datetime
from math import sin, cos, pi
import os
import bpy
import gpu
import bgl
import mathutils
import numpy as np
from bpy.types import Operator, Panel
from gpu_extras.batch import batch_for_shader
from .VRRenderer import RenderImage, RenderAnimation, RenderToolsPanel, VRRendererCancel

# Register all classes
def register():
    """ Register eeVR to Blender """
    bpy.types.Scene.renderModeEnum = bpy.props.EnumProperty(
        items=[
            ("EQUI", "Equirectangular", "Renders in equirectangular projection"),
            ("DOME", "Full Dome", "Renders in full dome projection"),
        ],
        default="EQUI",
        name="Mode",
    )
    bpy.types.Scene.domeModeEnum = bpy.props.EnumProperty(
        items=[
            ("0", "Equidistant (VTA)", "Renders in equidistant dome projection"),
            ("1", "Hemispherical (VTH)", "Renders in hemispherical dome projection"),
            ("2", "Equisolid", "Renders in equisolid dome projection"),
            ("3", "Stereographic", "Renders in Stereographic dome projection"),
        ],
        default="0",
        name="Method",
    )
    bpy.types.Scene.renderFOV = bpy.props.FloatProperty(
        180.0,
        default=180.0,
        name="FOV",
        min=180,
        max=360,
        description="Field of view in degrees",
    )
    bpy.types.Scene.cancelVRRenderer = bpy.props.BoolProperty(
        name="Cancel", default=True
    )
    bpy.utils.register_class(RenderImage)
    bpy.utils.register_class(RenderAnimation)
    bpy.utils.register_class(RenderToolsPanel)
    bpy.utils.register_class(VRRendererCancel)


# Unregister all classes
def unregister():
    """ Unregister eeVR from Blender """
    del bpy.types.Scene.domeModeEnum
    del bpy.types.Scene.renderModeEnum
    del bpy.types.Scene.renderFOV
    bpy.utils.unregister_class(RenderImage)
    bpy.utils.unregister_class(RenderAnimation)
    bpy.utils.unregister_class(RenderToolsPanel)
    bpy.utils.unregister_class(VRRendererCancel)


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

# If the script is not an addon when it is run, register the classes
if __name__ == "__main__":
    register()
