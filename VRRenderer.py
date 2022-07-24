import bpy
import os
import time
import gpu
import bgl
import numpy as np
from math import sin, cos, tan, pi, ceil, radians
from datetime import datetime
from gpu_extras.batch import batch_for_shader

commdef = '''
#define PI       3.1415926535897932384626
#define FOVFRAC  %f
#define SIDEFRAC %f
#define TBFRAC   %f
#define HCLIP    %f
#define VCLIP    %f
#define MARGIN   %f

const float INVSIDEFRAC = 1 / SIDEFRAC;
const float INVTBFRAC = 1 / TBFRAC;
const float MARGINSCALE = 1 - 2 * MARGIN;

vec2 tr(vec2 src, vec2 offset, vec2 scale)
{
    return (src + offset) * scale;
}

vec2 tr(vec2 src, vec2 offset, float scale)
{
    return (src + offset) * scale;
}

vec2 tr(vec2 src, float offset, float scale)
{
    return tr(src, vec2(offset, offset), scale);
}

vec2 to_uv(float x, float y)
{
    return tr(vec2(x, y), 1.0, 0.5);
}

vec2 apply_margin(vec2 src)
{
    return src * MARGINSCALE + vec2(MARGIN, MARGIN);
}

vec2 to_uv_right(vec3 pt)
{
    return to_uv(-pt.z/pt.x, pt.y/pt.x) * vec2(INVSIDEFRAC, 1);
}

vec2 to_uv_left(vec3 pt)
{
    return tr(to_uv(-pt.z/pt.x, -pt.y/pt.x), vec2(SIDEFRAC - 1, 0), vec2(INVSIDEFRAC, 1));
}

vec2 to_uv_top(vec3 pt)
{
    return to_uv(pt.x/pt.y, -pt.z/pt.y) * vec2(1, INVTBFRAC);
}

vec2 to_uv_bottom(vec3 pt)
{
    return tr(to_uv(-pt.x/pt.y, -pt.z/pt.y), vec2(0, TBFRAC - 1), vec2(1, INVTBFRAC));
}

vec2 to_uv_front(vec3 pt)
{
    return apply_margin(to_uv(pt.x/pt.z, pt.y/pt.z));
}

vec2 to_uv_back(vec3 pt)
{
    return apply_margin(to_uv(pt.x/pt.z, -pt.y/pt.z));
}

// Input cubemap textures
uniform sampler2D cubeLeftImage;
uniform sampler2D cubeRightImage;
uniform sampler2D cubeBottomImage;
uniform sampler2D cubeTopImage;
uniform sampler2D cubeBackImage;
uniform sampler2D cubeFrontImage;

in vec2 vTexCoord;

out vec4 fragColor;

void main() {

'''

dome = '''
    vec2 d = vTexCoord.xy;
    float r = length(d);
    if(r > 1.0) discard;
    
    // Calculate the position on unit sphere
    vec2 dunit = normalize(d);
    float phi = FOVFRAC * PI * r;
    vec3 pt;
    %s
    pt.z = cos(phi);

'''

domemodes = [
    'pt.xy = dunit * phi;',
    'pt.xy = dunit * sin(phi);',
    'pt.xy = 2.0 * dunit * sin(phi / 2.0);',
    'pt.xy = 2.0 * dunit * tan(phi / 2.0);'
]

equi = '''
    // Calculate the pointing angle
    float azimuth = FOVFRAC * PI * vTexCoord.x;
    if(abs(azimuth) > PI * FOVFRAC * HCLIP) discard;
    float elevation = 0.5 * PI * vTexCoord.y;
    
    // Calculate the position on unit sphere
    vec3 pt;
    pt.x = cos(elevation) * sin(azimuth);
    pt.y = sin(elevation);
    pt.z = cos(elevation) * cos(azimuth);
'''

fetch_setup = '''
    // Select the correct pixel
    // left or right
    float lor = step(abs(pt.y), abs(pt.x)) * step(abs(pt.z), abs(pt.x));
    // top or bottom
    float tob = (1.0 - lor) * step(abs(pt.z), abs(pt.y));
    // front or back
    float fob = (1.0 - lor) * (1.0 - tob);
    float right = step(0.0, pt.x);
    float up = step(0.0, pt.y);
    float front = step(0.0, pt.z);
    float angle = 0.0;
    angle = fob * abs(pt.y/pt.z) * 0.5;
    angle += lor * abs(pt.y/pt.x) * 0.5;
    angle += tob * (2 - abs(pt.z/pt.y)) * 0.5;
    if(angle > VCLIP) discard;
'''

fetch_sides = '''
    fragColor += lor * right * texture(cubeRightImage, to_uv_right(pt));
    fragColor += lor * (1.0 - right) * texture(cubeLeftImage, to_uv_left(pt));
'''

fetch_top_bottom = '''
    fragColor += tob * up * texture(cubeTopImage, to_uv_top(pt));
    fragColor += tob * (1.0 - up) * texture(cubeBottomImage, to_uv_bottom(pt));
'''

fetch_front_back = '''
    fragColor += fob * front * texture(cubeFrontImage, to_uv_front(pt));
    fragColor += fob * (1.0 - front) * texture(cubeBackImage, to_uv_back(pt));
}
'''

fetch_front_only = '''
    {
        vec2 uv = to_uv_front(pt);
        fragColor += fob * front * texture(cubeFrontImage, uv);
        float alpha = lor * right * smoothstep(1.0, 0.0, clamp((uv.x - MARGINSCALE - MARGIN) / MARGIN, 0.0, 1.0));
        fragColor = (1.0 - alpha) * fragColor + alpha * texture(cubeFrontImage, uv);
        
        alpha = lor * (1.0 - right) * smoothstep(0.0, 1.0, clamp(uv.x / MARGIN, 0.0, 1.0));
        fragColor = (1.0 - alpha) * fragColor + alpha * texture(cubeFrontImage, uv);
    }
}
'''

# Define the vertex shader
vertex_shader = '''
in vec3 aVertexPosition;
in vec2 aVertexTextureCoord;

out vec2 vTexCoord;

void main() {
    vTexCoord = aVertexTextureCoord;
    gl_Position = vec4(aVertexPosition, 1);
}
'''

class Renderer:
    
    def __init__(self, context, is_animation = False, folder = ''):
        
        # Check if the file is saved or not, can cause errors when not saved
        if not bpy.data.is_saved:
            raise PermissionError("Save file before rendering")
        
        eeVR = context.scene.eeVR

        # Set internal variables for the class
        self.scene = context.scene
        # save original active object
        self.viewlayer_active_object_origin = context.view_layer.objects.active
        # save original active camera handle
        self.camera_origin = context.scene.camera
        # create a new camera for rendering
        bpy.ops.object.camera_add()
        self.camera = context.object
        self.camera.name = 'eeVR_camera'
        # set new cam active
        context.scene.camera = self.camera
        # set coordinates same as origin by using world matrix already transformed but not location or rotation
        # and always using it to update correct coordinates before rendering
        # no constraints, parent, drivers and keyframes for new cam, now we can handle cameras with those stuff
        self.camera.matrix_world = self.camera_origin.matrix_world
        # transfer key attributes that may affect rendering, conv dis not needed 'cause it is parallel
        self.camera.data.stereo.interocular_distance = self.camera_origin.data.stereo.interocular_distance
        self.path = bpy.path.abspath("//")
        self.is_stereo = context.scene.render.use_multiview
        self.is_animation = is_animation
        self.is_dome = (eeVR.renderModeEnum == 'DOME')
        self.domeMethod = bpy.context.scene.eeVR.domeMethodEnum
        if self.is_dome:
            self.FOV = min(max(eeVR.domeFOV, 180), 360)
        else:
            self.FOV = 180.0 if eeVR.equiModeEnum == "180" else eeVR.equi360HFOV
        equiHFOV = eeVR.equi180HFOV if eeVR.equiModeEnum == "180" else eeVR.equi360HFOV
        self.HFOV = (self.FOV if eeVR.renderModeEnum == 'DOME' else equiHFOV)
        self.VFOV = (self.FOV if eeVR.renderModeEnum =='DOME' else eeVR.equiVFOV)
        self.no_back_image = (self.HFOV <= 270)
        self.no_side_images = (self.HFOV <= 90)
        self.no_top_bottom_images = (self.VFOV <= 90)
        self.createdFiles = set()
        
        # Generate fragment shader code
        fovfrac = self.FOV / 360.0
        if self.HFOV > 270:
            sidefrac = 1.0
        elif self.HFOV > 180:
            sidefrac = sin(radians(self.HFOV - 180)) * 0.5 + 0.5
        else:
            sidefrac = sin(radians(self.HFOV - 90)) * 0.5
        tbfrac = max(sidefrac, sin(radians(self.VFOV - 90)) * 0.5)
        hclip = self.HFOV / self.FOV
        vclip = self.VFOV / 180.0
        margin = 0.5 - 0.5 / tan(radians(45 + eeVR.stitchMargin))
        # margin = (tan(radians(frontfov/2)) - tan(radians(frontfov/2)) / tan(radians(frontfov/2 + eeVR.stitchMargin))) / 2
        print(fovfrac, sidefrac, tbfrac, hclip, vclip, margin, 1 - 2 * margin)
        self.frag_shader = \
           (commdef % (fovfrac, sidefrac, tbfrac, hclip, vclip, margin))\
         + (dome % domemodes[int(self.domeMethod)] if self.is_dome else equi)\
         + fetch_setup\
         + ('' if self.no_side_images else fetch_sides)\
         + ('' if self.no_top_bottom_images else fetch_top_bottom)\
         + (fetch_front_only if self.no_back_image else fetch_front_back)
        
        # Set the image name to the current time
        self.start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # get folder name from outside
        self.folder_name = folder
        
        # Get initial camera and output information
        # now origin camera data not need store, and no more need to use empty as proxy
        self.camera_rotation = list(self.camera.rotation_euler)
        self.IPD = self.camera.data.stereo.interocular_distance
        
        # Set camera variables for proper result
        self.camera.data.type = 'PANO'
        self.camera.data.stereo.convergence_mode = 'PARALLEL'
        self.camera.data.stereo.pivot = 'CENTER'
        
        self.resolution_x_origin = self.scene.render.resolution_x
        self.resolution_y_origin = self.scene.render.resolution_y
        self.resolution_percentage_origin = self.scene.render.resolution_percentage
        scale = (self.resolution_percentage_origin / 100.0)
        self.image_size = int(ceil(self.scene.render.resolution_x * scale)), int(ceil(self.scene.render.resolution_y * scale))
        self.base_resolution = self.image_size[0] * (90.0 / self.FOV)
        self.camera_shift = {
            'top': [0.0, 0.5*(tbfrac-1), self.base_resolution, tbfrac*self.base_resolution],
            'bottom': [0.0, 0.5*(1-tbfrac), self.base_resolution, tbfrac*self.base_resolution],
            'left': [0.5*(1-sidefrac), 0.0, sidefrac*self.base_resolution, self.base_resolution],
            'right': [0.5*(sidefrac-1), 0.0, sidefrac*self.base_resolution, self.base_resolution],
            'front': [0.0, 0.0, self.base_resolution, self.base_resolution],
            'back': [0.0, 0.0, self.base_resolution, self.base_resolution]
        }
        print(self.camera_shift)
        if self.is_stereo:
            self.view_format = self.scene.render.image_settings.views_format
            self.scene.render.image_settings.views_format = 'STEREO_3D'
            self.stereo_mode = self.scene.render.image_settings.stereo_3d_format.display_mode
            self.scene.render.image_settings.stereo_3d_format.display_mode = 'TOPBOTTOM'

        self.direction_offsets = self.find_direction_offsets()
    
    
    def cubemap_to_panorama(self, imageList, outputName):
        
        # Generate the OpenGL shader
        pos = [(-1.0, -1.0, -1.0),  # left,  bottom, back
               (-1.0,  1.0, -1.0),  # left,  top,    back
               (1.0, -1.0, -1.0),   # right, bottom, back
               (1.0,  1.0, -1.0)]   # right, top,    back
        coords = [(-1.0, -1.0),  # left,  bottom
                  (-1.0,  1.0),  # left,  top
                  (1.0, -1.0),   # right, bottom
                  (1.0,  1.0)]   # right, top
        vertexIndices = [(0, 3, 1),(3, 0, 2)]
        shader = gpu.types.GPUShader(vertex_shader, self.frag_shader)
        
        batch = batch_for_shader(shader, 'TRIS', {
            "aVertexPosition": pos,
            "aVertexTextureCoord": coords
        }, indices=vertexIndices)
        
        # Change the color space of all of the images to Linear
        # and load them into OpenGL textures
        for image in imageList:
            image.colorspace_settings.name='Linear'
            image.gl_load()
        
        # set the size of the final image
        width = self.image_size[0]
        height = self.image_size[1]

        # Create an offscreen render buffer and texture
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            bgl.glClearColor(0.0, 0.0, 0.0, 1.0)
            bgl.glClear(bgl.GL_COLOR_BUFFER_BIT)

            shader.bind()
            
            def bind_and_filter(tex, bindcode, image=None, imageNum=None):
                bgl.glActiveTexture(tex)
                bgl.glBindTexture(bgl.GL_TEXTURE_2D, bindcode)
                bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR)
                bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR)
                bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
                bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
                if image!=None and imageNum!=None:
                    shader.uniform_int(image, imageNum)
            
            # Bind all of the cubemap textures and enable correct filtering and wrapping
            # to prevent seams
            bind_and_filter(bgl.GL_TEXTURE0, imageList[0].bindcode, "cubeFrontImage", 0)
            if self.no_side_images:
                if not self.no_top_bottom_images:
                    bind_and_filter(bgl.GL_TEXTURE1, imageList[1].bindcode, "cubeBottomImage", 1)
                    bind_and_filter(bgl.GL_TEXTURE2, imageList[2].bindcode, "cubeTopImage", 2)
                    if not self.no_back_image:
                        bind_and_filter(bgl.GL_TEXTURE3, imageList[3].bindcode, "cubeBackImage", 3)
            else:
                bind_and_filter(bgl.GL_TEXTURE1, imageList[1].bindcode, "cubeLeftImage", 1)
                bind_and_filter(bgl.GL_TEXTURE2, imageList[2].bindcode, "cubeRightImage", 2)
                if not self.no_top_bottom_images:
                    bind_and_filter(bgl.GL_TEXTURE3, imageList[3].bindcode, "cubeBottomImage", 3)
                    bind_and_filter(bgl.GL_TEXTURE4, imageList[4].bindcode, "cubeTopImage", 4)
                    if not self.no_back_image:
                        bind_and_filter(bgl.GL_TEXTURE5, imageList[5].bindcode, "cubeBackImage", 5)
                else:
                    if not self.no_back_image:
                        bind_and_filter(bgl.GL_TEXTURE3, imageList[3].bindcode, "cubeBackImage", 3)
            
            # Bind the resulting texture
            bind_and_filter(bgl.GL_TEXTURE6, offscreen.color_texture)
            
            # Render the image
            batch.draw(shader)
            
            # Unload the textures
            for image in imageList:
                image.gl_free()
            
            # Read the resulting pixels into a buffer
            buffer = bgl.Buffer(bgl.GL_FLOAT, width * height * 4)
            bgl.glGetTexImage(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA, bgl.GL_FLOAT, buffer)

        # Unload the offscreen texture
        offscreen.free()
        
        # Remove the cubemap textures:
        for image in imageList:
            bpy.data.images.remove(image)
        
        # Copy the pixels from the buffer to an image object
        if not outputName in bpy.data.images.keys():
            bpy.data.images.new(outputName, width, height)
        imageRes = bpy.data.images[outputName]
        imageRes.scale(width, height)
        imageRes.pixels.foreach_set(buffer)
        return imageRes

    
    def find_direction_offsets(self):
        
        # update location and rotation of our camera from origin one
        self.camera.matrix_world = self.camera_origin.matrix_world
        # Calculate the pointing directions of the camera for each face of the cube
        # Using euler.rotate_axis() to handle, notice that rotation should be done on copies
        eul = self.camera.rotation_euler.copy()
        direction_offsets = {}
        #front
        direction_offsets['front'] = list(eul)
        #back
        eul.rotate_axis('Y', pi)
        direction_offsets['back'] = list(eul)
        #top
        eul = self.camera.rotation_euler.copy()
        eul.rotate_axis('X', pi/2)
        direction_offsets['top'] = list(eul)
        #bottom
        eul.rotate_axis('X', pi)
        direction_offsets['bottom'] = list(eul)
        #left
        eul = self.camera.rotation_euler.copy()
        eul.rotate_axis('Y', pi/2)
        direction_offsets['left'] = list(eul)
        #right
        eul.rotate_axis('Y', pi)
        direction_offsets['right'] = list(eul)
        return direction_offsets
    
    
    def set_camera_direction(self, direction):
        
        # Set the camera to the required postion    
        self.camera.rotation_euler = self.direction_offsets[direction]
        self.camera.data.angle_x = radians(90) if direction != 'front' else radians(90 + 2 * self.scene.eeVR.stitchMargin)
        self.camera.data.shift_x = self.camera_shift[direction][0]
        self.camera.data.shift_y = self.camera_shift[direction][1]
        self.scene.render.resolution_x = int(ceil(self.camera_shift[direction][2]))
        self.scene.render.resolution_y = int(ceil(self.camera_shift[direction][3]))
        self.scene.render.resolution_percentage = 100
        # print(f"res {self.scene.render.resolution_x}, {self.scene.render.resolution_y} {self.camera_shift[direction]}")


    def clean_up(self, context):

        # Reset all the variables that were changed
        context.view_layer.objects.active = self.viewlayer_active_object_origin
        context.scene.camera = self.camera_origin
        bpy.data.objects.remove(self.camera)
        self.scene.render.resolution_x = self.resolution_x_origin
        self.scene.render.resolution_y = self.resolution_y_origin
        self.scene.render.resolution_percentage = self.resolution_percentage_origin
        if self.is_stereo:
            self.scene.render.image_settings.views_format = self.view_format
            self.scene.render.image_settings.stereo_3d_format.display_mode = self.stereo_mode
        for filename in self.createdFiles:
            os.remove(filename)
        self.createdFiles.clear()
    
    
    def render_image(self, direction):
        
        # Render the image and load it into the script
        name = f'temp_img_store_{direction}'
        if self.is_stereo:
            nameL = name + '_L'
            nameR = name + '_R'
        tmp = self.scene.render.filepath
        
        # If rendering for VR, render the side images separately to avoid seams
        if self.is_stereo and direction in {'right', 'left'}:
            if nameL in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[nameL])
            if nameR in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[nameR])
            
            self.scene.render.use_multiview = False
            tmp_loc = list(self.camera.location)
            camera_angle = self.direction_offsets['front'][2]
            self.camera.location = [tmp_loc[0]+(0.5*self.IPD*cos(camera_angle)),\
                                    tmp_loc[1]+(0.5*self.IPD*sin(camera_angle)),\
                                    tmp_loc[2]]

            self.scene.render.filepath = self.path + nameL + '.png'
            bpy.ops.render.render(write_still=True)
            self.createdFiles.add(self.scene.render.filepath)
            renderedImageL = bpy.data.images.load(self.scene.render.filepath)
            renderedImageL.name = nameL
            
            self.camera.location = [tmp_loc[0]-(0.5*self.IPD*cos(camera_angle)),\
                                    tmp_loc[1]-(0.5*self.IPD*sin(camera_angle)),\
                                    tmp_loc[2]]

            self.scene.render.filepath = self.path + nameR + '.png'
            bpy.ops.render.render(write_still=True)
            self.createdFiles.add(self.scene.render.filepath)
            renderedImageR = bpy.data.images.load(self.scene.render.filepath)
            renderedImageR.name = nameR

            self.scene.render.use_multiview = True
            self.camera.location = tmp_loc
        
        elif self.is_stereo:
            if name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[name])
            if nameL in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[nameL])
            if nameR in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[nameR])

            self.scene.render.filepath = self.path + name + '.png'
            bpy.ops.render.render(write_still=True)
            self.createdFiles.add(self.scene.render.filepath)
            renderedImage =  bpy.data.images.load(self.scene.render.filepath)
            renderedImage.name = name
            renderedImage.colorspace_settings.name='Linear'
            imageLen = len(renderedImage.pixels)
            renderedImageL = bpy.data.images.new(nameL, self.scene.render.resolution_x, self.scene.render.resolution_y)
            renderedImageR = bpy.data.images.new(nameR, self.scene.render.resolution_x, self.scene.render.resolution_y)
            
            # Split the render into two images
            buff = np.empty((imageLen,), dtype=np.float32)
            renderedImage.pixels.foreach_get(buff)
            if direction == 'back':
                renderedImageL.pixels.foreach_set(buff[imageLen//2:])
                renderedImageR.pixels.foreach_set(buff[:imageLen//2])
            else:
                renderedImageR.pixels.foreach_set(buff[imageLen//2:])
                renderedImageL.pixels.foreach_set(buff[:imageLen//2])
            renderedImageL.pack()
            renderedImageR.pack()
            bpy.data.images.remove(renderedImage)
        else:
            if name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[name])

            self.scene.render.filepath = self.path + name + '.png'
            bpy.ops.render.render(write_still=True)
            self.createdFiles.add(self.scene.render.filepath)
            renderedImageL = bpy.data.images.load(self.scene.render.filepath)
            renderedImageL.name = name
            renderedImageR = None
        
        self.scene.render.filepath = tmp
        return renderedImageL, renderedImageR
    
    
    def render_images(self):
        
        # Render the images for every direction
        image_list_1 = []
        image_list_2 = []
        
        directions = ['front']
        if not self.no_side_images:
            directions += ['left', 'right']
        if not self.no_top_bottom_images:
            directions += ['bottom', 'top']
        if not self.no_back_image:
            directions += ['back']

        self.direction_offsets = self.find_direction_offsets()
        for direction in reversed(directions): # I want the results of the front camera to remain in the render window... just that.
            self.set_camera_direction(direction)
            img1, img2 = self.render_image(direction)
            image_list_1.insert(0, img1)
            image_list_2.insert(0, img2)
        
        return image_list_1, image_list_2


    def render_and_save(self):

        frame_step = self.scene.frame_step
        
        # Render the images and return their names
        imageList, imageList2 = self.render_images()
        if self.is_animation:
            image_name = f"frame{self.scene.frame_current:06d}.png"
        else:
            image_name = f"Render Result {self.start_time}.png"
        
        start_time = time.time()
        # Convert the rendered images to equirectangular projection image and save it to the disk
        if self.is_stereo:
            imageResult1 = self.cubemap_to_panorama(imageList, "Render Left")
            imageResult2 = self.cubemap_to_panorama(imageList2, "Render Right")

            # If it doesn't already exist, create an image object to store the resulting render
            if not image_name in bpy.data.images.keys():
                imageResult = bpy.data.images.new(image_name, imageResult1.size[0], 2 * imageResult1.size[1])
            
            imageResult = bpy.data.images[image_name]
            if self.stereo_mode == 'SIDEBYSIDE':
                imageResult.scale(2*imageResult1.size[0], imageResult1.size[1])
                img2arr = np.empty((imageResult2.size[1], 4 * imageResult2.size[0]), dtype=np.float32)
                imageResult2.pixels.foreach_get(img2arr.ravel())
                img1arr = np.empty((imageResult1.size[1], 4 * imageResult1.size[0]), dtype=np.float32)
                imageResult1.pixels.foreach_get(img1arr.ravel())
                imageResult.pixels.foreach_set(np.concatenate((img2arr, img1arr), axis=1).ravel())
            else:
                imageResult.scale(imageResult1.size[0], 2*imageResult1.size[1])
                buff = np.empty((imageResult1.size[0] * 2 * imageResult1.size[1] * 4,), dtype=np.float32)
                imageResult2.pixels.foreach_get(buff[:buff.shape[0]//2].ravel())
                imageResult1.pixels.foreach_get(buff[buff.shape[0]//2:].ravel())
                imageResult.pixels.foreach_set(buff.ravel())
            bpy.data.images.remove(imageResult1)
            bpy.data.images.remove(imageResult2)

        else:
            imageResult = self.cubemap_to_panorama(imageList, "RenderResult")
        
        save_start_time = time.time()
        if self.is_animation:
            # Color Management Settings issue solved by nagadomi
            imageResult.file_format = 'PNG'
            imageResult.filepath_raw = self.path+self.folder_name+image_name
            imageResult.save()
            self.scene.frame_set(self.scene.frame_current+frame_step)
        else:
            imageResult.file_format = 'PNG'
            imageResult.filepath_raw = self.path+image_name
            imageResult.save()

        print(f'''Saved '{imageResult.filepath_raw}'
 Time : {round(time.time() - start_time, 2)} seconds (Saving : {round(time.time() - save_start_time, 2)} seconds)
 ''')

        bpy.data.images.remove(imageResult)
