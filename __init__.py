""" Initialize eeVR blender plugin """

# pylint: disable=import-error

import os
import time
from datetime import datetime

if "bpy" in locals():
    import importlib
    importlib.reload(VRRenderer)
else:
    from . import VRRenderer

from .VRRenderer import Renderer

import bpy
from bpy.types import Operator, Panel

bl_info = {
    "name": "eeVR",
    "description": "Render in different projections using Eevee engine",
    "author": "EternalTrail, SAMtak",
    "version": (0, 2, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Tool Tab (Available when EEVEE or Workbench)",
    "warning": "This addon is still in early alpha, may break your blend file!",
    "wiki_url": "https://github.com/EternalTrail/eeVR",
    "tracker_url": "https://github.com/EternalTrail/eeVR/issues",
    "support": "TESTING",
    "category": "Render",
}

class RenderImage(Operator):
    """Render out the animation"""

    bl_idname = 'eevr.render_image'
    bl_label = "Render a single frame"

    def execute(self, context):
        print("eeVR: execute")

        mode = context.scene.eeVR.renderModeEnum
        HFOV = context.scene.eeVR.renderHFOV
        VFOV = context.scene.eeVR.renderVFOV
        renderer = Renderer(context, False, mode, HFOV, VFOV)
        now = time.time()
        renderer.render_and_save()
        print(f"eeVR: {round(time.time() - now, 2)} seconds")
        renderer.clean_up(context)

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
                self.renderer.render_and_save()
                print(f"eeVR: {round(time.time() - now, 2)} seconds")
                self.timer = wm.event_timer_add(0.1, window=context.window)
            else:
                self.clean(context)
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        print("eeVR: execute")

        context.scene.eeVR.cancel = False

        mode = context.scene.eeVR.renderModeEnum
        HFOV = context.scene.eeVR.renderHFOV
        VFOV = context.scene.eeVR.renderVFOV
        # knowing it's animation, creates folder outside vrrender class, pass folder name to it
        start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        folder_name = f"Render Result {start_time}/"
        path = bpy.path.abspath("//")
        os.makedirs(path+folder_name, exist_ok=True)
        self.renderer = Renderer(context, True, mode, HFOV, VFOV, folder_name)

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

        props = context.scene.eeVR
        # Draw the buttons for each of the rendering operators
        layout = self.layout
        col = layout.column()
        col.prop(props, 'renderModeEnum')
        if props.renderModeEnum == 'DOME':
            col.prop(props, 'domeModeEnum')
        col.prop(props, 'renderHFOV')
        col.prop(props, 'renderHFill')
        col.label(text=f"Actual Degree : {round(props.renderHFOV * props.renderHFill)}°")
        col.prop(props, 'renderVFOV')
        col.prop(props, 'renderVFill')
        col.label(text=f"Actual Degree : {round(props.renderVFOV * props.renderVFill)}°")
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

    domeModeEnum: bpy.props.EnumProperty(
        items=[
            ("0", "Equidistant (VTA)", "Renders in equidistant dome projection"),
            ("1", "Hemispherical (VTH)", "Renders in hemispherical dome projection"),
            ("2", "Equisolid", "Renders in equisolid dome projection"),
            ("3", "Stereographic", "Renders in Stereographic dome projection"),
        ],
        default="0",
        name="Method",
    )

    renderHFOV: bpy.props.FloatProperty(
        180.0,
        default=180.0,
        name="Horizontal FOV",
        min=1,
        max=360,
        description="Horizontal Field of view in degrees",
    )

    renderVFOV: bpy.props.FloatProperty(
        180.0,
        default=180.0,
        name="Vertical FOV",
        min=1,
        max=180,
        description="Vertical Field of view in degrees",
    )

    renderHFill: bpy.props.FloatProperty(
        1.0,
        default=1.0,
        name="Horizontal Fill Rate",
        min=0.1,
        max=1.0,
        description="Horizontal Render Regeon in rate of Horizontal FOV",
    )

    renderVFill: bpy.props.FloatProperty(
        1.0,
        default=1.0,
        name="Vertical Fill Rate",
        min=0.1,
        max=1.0,
        description="Vertical Render Regeon in rate of Vertical FOV",
    )

    renderVFOV: bpy.props.FloatProperty(
        180.0,
        default=180.0,
        name="VFOV",
        min=1,
        max=180,
        description="Vertical Field of view in degrees",
    )

    cancel: bpy.props.BoolProperty(
        name="Cancel",
        default=True
    )

    @classmethod
    def register(cls):
        """ Register eeVR's properties to Blender """
        bpy.types.Scene.eeVR = bpy.props.PointerProperty(type=cls)

    @classmethod
    def unregister(cls):
        """ Unregister eeVR's properties from Blender """
        del bpy.types.Scene.eeVR

# REGISTER
register, unregister = bpy.utils.register_classes_factory((
    Properties,
    ToolPanel,
    RenderImage,
    RenderAnimation,
    Cancel,
))
