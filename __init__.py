""" Initialize eeVR blender plugin """

# pylint: disable=import-error

import os
import time
from datetime import datetime
from math import radians

if "bpy" in locals():
    import importlib
    importlib.reload(renderer)
else:
    from . import renderer

from .renderer import Renderer

import bpy
from bpy.types import Context, Operator, Panel

bl_info = {
    "name": "eeVR",
    "description": "Render in different projections using Eevee engine",
    "author": "EternalTrail",
    "version": (0, 5, 2),
    "blender": (2, 82, 7),
    "location": "View3D > Tool Tab (Available when EEVEE or Workbench)",
    "warning": "This addon is still in early alpha, may break your blend file!",
    "wiki_url": "https://github.com/EternalTrail/eeVR",
    "tracker_url": "https://github.com/EternalTrail/eeVR/issues",
    "support": "COMMUNITY",
    "category": "Render",
}


def has_invalid_condition(self : 'Operator', context : 'Context'):
    if context.scene.camera == None:
        self.report({'ERROR'}, "eeVR ERROR : Scene camera is not set.")
        return True
    return False


class RenderImage(Operator):
    """Render out the animation"""

    bl_idname = 'eevr.render_image'
    bl_label = "Render a single frame"

    def execute(self, context):
        print("eeVR: execute")

        if has_invalid_condition(self, context):
            return {'FINISHED'}

        renderer = Renderer(context, False)
        now = time.time()
        try:
            renderer.render_and_save()
        finally:
            renderer.clean_up(context)

        print(f"eeVR: {round(time.time() - now, 2)} seconds")

        return {'FINISHED'}


class RenderAnimation(Operator):
    """Render out the animation"""

    bl_idname = 'eevr.render_animation'
    bl_label = "Render the animation"

    def __del__(self):
        print("eeVR: end")

    def modal(self, context, event):
        if event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            wm = context.window_manager
            wm.event_timer_remove(self.timer)

            if context.scene.eeVR.cancel:
                self.cancel(context)
                return {'CANCELLED'}

            if context.scene.frame_current <= self.frame_end:
                print(f"eeVR: Rendering frame {context.scene.frame_current}")
                now = time.time()
                try:
                    self.renderer.render_and_save()
                except Exception as e:
                    self.clean(context)
                    raise e
                print(f"eeVR: {round(time.time() - now, 2)} seconds")
                self.timer = wm.event_timer_add(0.1, window=context.window)
            else:
                self.clean(context)
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        print("eeVR: execute")

        if has_invalid_condition(self, context):
            return {'FINISHED'}

        context.scene.eeVR.cancel = False

        # knowing it's animation, creates folder outside vrrender class, pass folder name to it
        start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        folder_name = f"Render Result {start_time}/"
        path = bpy.path.abspath("//")
        os.makedirs(path+folder_name, exist_ok=True)
        self.renderer = Renderer(context, True, folder_name)
        
        self.frame_end = context.scene.frame_end
        frame_start = context.scene.frame_start
        context.scene.frame_set(frame_start)
        wm = context.window_manager
        self.timer = wm.event_timer_add(5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        print("eeVR: cancel")
        self.clean(context)

    def clean(self, context):
        self.renderer.clean_up(context)
        context.scene.eeVR.cancel = True


class Cancel(Operator):
    """Render out the animation"""

    bl_idname = 'eevr.render_cancel'
    bl_label = "Cancel the render"

    def execute(self, context):
        context.scene.eeVR.cancel = True
        return {'FINISHED'}


class ToolPanel(Panel):
    """Tool panel for VR rendering"""

    bl_idname = "EEVR_PT_tool"
    bl_label = "eeVR"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"

    COMPAT_ENGINES = {'BLENDER_RENDER', 'BLENDER_EEVEE', 'BLENDER_WORKBENCH'}

    @classmethod
    def poll(cls, context):
        return (context.engine in cls.COMPAT_ENGINES)

    def draw(self, context):

        props : Properties = context.scene.eeVR
        # Draw the buttons for each of the rendering operators
        layout = self.layout
        col = layout.column()
        col.prop(props, 'renderModeEnum')
        if props.renderModeEnum == 'DOME':
            col.prop(props, 'domeMethodEnum')
        col.prop(props, 'fovModeEnum')
        if props.fovModeEnum == '180':
            col.prop(props, 'HFOV180')
        else:
            col.prop(props, 'HFOV360')
        col.prop(props, 'VFOV')
        col.prop(props, 'stitchMargin')
        col.prop(props, 'frontViewOverscan')
        col = layout.column()
        col.prop(props, 'noSidePlane')
        col.enabled = props.IsEnableNoSidePlane()
        if context.scene.render.use_multiview and props.GetHFOV() > radians(180):
            col = layout.column()
            col.label(icon='ERROR', text="eeVR cannot support stereo over 180° fov correctly.")
        layout.separator()
        col = layout.column()
        col.operator(RenderImage.bl_idname, text="Render Image")
        col.operator(RenderAnimation.bl_idname, text="Render Animation")
        if not props.cancel:
            col.operator(Cancel.bl_idname, text="Cancel")
            col.label(text=f"Rendering frame {context.scene.frame_current}")


# eeVR's properties
class Properties(bpy.types.PropertyGroup):

    renderModeEnum: bpy.props.EnumProperty(
        items=[
            ("EQUI", "Equirectangular", "Renders in equirectangular projection"),
            ("DOME", "Full Dome", "Renders in full dome projection"),
        ],
        default="EQUI",
        name="Mode",
    )

    domeMethodEnum: bpy.props.EnumProperty(
        items=[
            ("0", "Equidistant (VTA)", "Renders in equidistant dome projection"),
            ("1", "Hemispherical (VTH)", "Renders in hemispherical dome projection"),
            ("2", "Equisolid", "Renders in equisolid dome projection"),
            ("3", "Stereographic", "Renders in Stereographic dome projection"),
        ],
        default="0",
        name="Method",
    )

    fovModeEnum: bpy.props.EnumProperty(
        items=[
            ("180", "180°", "VR 180"),
            ("360", "360°", "VR 360 (over 180°, not support stereo)"),
            ("ANY", "Custom FOV", "over 180°, not support stereo"),
        ],
        default="180",
        name="VR Format",
    )

    HFOV360: bpy.props.FloatProperty(
        name="Horizontal FOV",
        subtype='ANGLE',
        precision=0,
        step=100,
        default=radians(360),
        min=radians(1),
        max=radians(360),
        description="Horizontal Field of view in degrees",
    )

    HFOV180: bpy.props.FloatProperty(
        name="Horizontal FOV",
        subtype='ANGLE',
        unit='ROTATION',
        precision=0,
        step=100,
        default=radians(180),
        min=radians(1),
        max=radians(180),
        description="Horizontal Field of view in degrees",
    )

    VFOV: bpy.props.FloatProperty(
        name="Vertical FOV",
        subtype='ANGLE',
        unit='ROTATION',
        precision=0,
        step=100,
        default=radians(180),
        min=radians(1),
        max=radians(180),
        description="Vertical Field of view in degrees",
    )

    stitchMargin: bpy.props.FloatProperty(
        name="Stitch Margin",
        subtype='ANGLE',
        unit='ROTATION',
        precision=0,
        step=100,
        default=radians(5),
        min=radians(0),
        max=radians(15),
        description="Margin for Seam Blending in degrees",
    )

    frontViewOverscan: bpy.props.FloatProperty(
        name="Front View Overscan",
        subtype='PERCENTAGE',
        precision=0,
        step=100,
        default=25,
        min=0,
        max=100,
        description="Overscan Rate for Front View Rendering",
    )

    noSidePlane: bpy.props.BoolProperty(
        name="No Side Plane",
        default=False,
        description="Not render side views. This is enable when Horizontal FOV under 160°",
    )

    cancel: bpy.props.BoolProperty(
        name="Cancel",
        default=True
    )

    @staticmethod
    def snap_angle(src: float) -> float:
        if abs(src - radians(90)) < 0.000001:
            return radians(90)
        elif abs(src - radians(180)) < 0.000001:
            return radians(180)
        elif abs(src - radians(270)) < 0.000001:
            return radians(270)
        elif abs(src - radians(360)) < 0.000001:
            return radians(360)
        return src

    def GetHFOV(self) -> float:
        return self.snap_angle(self.HFOV180 if self.fovModeEnum == '180' else self.HFOV360)

    def GetVFOV(self) -> float:
        return self.snap_angle(self.VFOV)

    def IsEnableNoSidePlane(self) -> bool:
        return self.GetHFOV() < radians(165)

    def GetNoSidePlane(self) -> bool:
        return self.IsEnableNoSidePlane() and self.noSidePlane

    @classmethod
    def register(cls):
        """ Register eeVR's properties to Blender """
        bpy.types.Scene.eeVR = bpy.props.PointerProperty(type=cls)

    @classmethod
    def unregister(cls):
        """ Unregister eeVR's properties from Blender """
        del bpy.types.Scene.eeVR


class Preferences(bpy.types.AddonPreferences):
    # this must match the addon name, use '__package__'
    # when defining this in a submodule of a python package.
    bl_idname = __name__

    remain_temporaries: bpy.props.BoolProperty(
        name='Remain temporal work files',
        description='Temporal rendering images will not be removed.',
        default=False
    )

    def draw(self, context):
        self.layout.row().prop(self, 'remain_temporaries')


# REGISTER
register, unregister = bpy.utils.register_classes_factory((
    Properties,
    ToolPanel,
    RenderImage,
    RenderAnimation,
    Cancel,
    Preferences,
))
