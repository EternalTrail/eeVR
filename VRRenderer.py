import bpy
import os
import gpu
import bgl
import numpy as np
from math import sin, cos, pi
from datetime import datetime
from gpu_extras.batch import batch_for_shader

commdef = '''
#define PI 3.1415926535897932384626

// Input cubemap textures
uniform sampler2D cubeLeftImage;
uniform sampler2D cubeRightImage;
uniform sampler2D cubeBottomImage;
uniform sampler2D cubeTopImage;
uniform sampler2D cubeBackImage;
uniform sampler2D cubeFrontImage;

in vec2 vTexCoord;

out vec4 fragColor;

'''

dome = '''
void main() {{

    vec2 d = vTexCoord.xy;
    float r = length(d);
    if( r > 1.0 ) discard;

    float hfovfrac = {0};
    float sidefrac = {1};
    
    vec2 dunit = normalize(d);
    float phi = hfovfrac*r*PI;
    vec3 pt;
    {2}
    pt.z = cos(phi);

    // Select the correct pixel
    float side = float((abs(pt.x) >= abs(pt.y)) && (abs(pt.x) >= abs(pt.z)));
    float notfront = float(abs(pt.y) >= abs(pt.z));
'''

domemodes = [
    'pt.xy = dunit * phi;',
    'pt.xy = dunit * sin(phi);',
    'pt.xy = 2.0 * dunit * sin(phi / 2.0);',
    'pt.xy = 2.0 * dunit * tan(phi / 2.0);'
]

equi = '''
void main() {{

    // Calculate the pointing angle
    float hfovfrac = {0};
    float sidefrac = {1};
    float azimuth = vTexCoord.x * PI * hfovfrac;
    float elevation = vTexCoord.y * PI / 2.0;
    
    // Calculate the pointing vector
    vec3 pt;
    pt.x = cos(elevation) * sin(azimuth);
    pt.y = sin(elevation);
    pt.z = cos(elevation) * cos(azimuth);
    
    // Select the correct pixel
    float side = float((abs(pt.x) >= abs(pt.y)) && (abs(pt.x) >= abs(pt.z)));
    float notfront = float(abs(pt.y) >= abs(pt.z));
'''

fetch_sides = '''
    float left = float(pt.x <= 0.0);
    fragColor += side * left * texture(cubeLeftImage, vec2((((-pt.z/pt.x))+(2.0*sidefrac-1.0))/(2.0*sidefrac),((-pt.y/pt.x)+1.0)/2.0));
    fragColor += side * (1.0 - left) * texture(cubeRightImage, vec2(((-pt.z/pt.x)+1.0)/(2.0*sidefrac),((pt.y/pt.x)+1.0)/2.0));
'''

fetch_top_bottom = '''
    float down = float(pt.y <= 0.0);
    fragColor += (1.0 - side) * notfront * down * texture(cubeBottomImage, vec2(((-pt.x/pt.y)+1.0)/2.0,((-pt.z/pt.y)+(2.0*sidefrac-1.0))/(2.0*sidefrac)));
    fragColor += (1.0 - side) * notfront * (1.0 - down) * texture(cubeTopImage, vec2(((pt.x/pt.y)+1.0)/2.0,((-pt.z/pt.y)+1.0)/(2.0*sidefrac)));
'''

fetch_front_back = '''
    float back = float(pt.z <= 0.0);
    fragColor += (1.0 - side) * (1.0 - notfront) * back * texture(cubeBackImage, vec2(((pt.x/pt.z)+1.0)/2.0,((-pt.y/pt.z)+1.0)/2.0));
    fragColor += (1.0 - side) * (1.0 - notfront) * (1.0 - back) * texture(cubeFrontImage, vec2(((pt.x/pt.z)+1.0)/2.0,((pt.y/pt.z)+1.0)/2.0));
}}
'''

fetch_front_only = '''
    fragColor += (1.0 - side) * (1.0 - notfront) * texture(cubeFrontImage, vec2(((pt.x/pt.z)+1.0)/2.0,((pt.y/pt.z)+1.0)/2.0));
}}
'''

class Renderer:
    
    def __init__(self, is_stereo = False, is_animation = False, mode = 'EQUI', HFOV = 180, VFOV = 180, folder = ''):
        
        # Check if the file is saved or not, can cause errors when not saved
        if not bpy.data.is_saved:
            raise PermissionError("Save file before rendering")
        
        # Set internal variables for the class
        self.scene = bpy.context.scene
        ##### newly edited codes begin
        # save original active camera handle
        self.camera_origin = bpy.context.scene.camera
        # create a new camera for rendering
        bpy.ops.object.camera_add()
        self.camera = bpy.context.object
        self.camera.name = 'eeVR_camera'
        # set new cam active
        bpy.context.scene.camera = self.camera
        # set coordinates same as origin by using world matrix already transformed but not location or rotation
        # and always using it to update correct coordinates before rendering
        # no constraints, parent, drivers and keyframes for new cam, now we can handle cameras with those stuff
        self.camera.matrix_world = self.camera_origin.matrix_world
        # transfer key attributes that may affect rendering, conv dis not needed 'cause it is parallel
        self.camera.data.stereo.interocular_distance = self.camera_origin.data.stereo.interocular_distance
        ##### newly edited codes end
        self.path = bpy.path.abspath("//")
        self.is_stereo = is_stereo
        self.is_animation = is_animation
        self.HFOV = min(max(HFOV, 1), 360)
        self.VFOV = min(max(VFOV, 1), 180)
        self.no_back_image = (self.HFOV <= 270)
        self.no_side_images = (self.HFOV <= 90)
        self.no_top_bottom_images = (self.VFOV <= 90)
        self.is_dome = (mode == 'DOME')
        self.domeMode = bpy.context.scene.eeVR.domeModeEnum
        self.createdFiles = set()
        
        # Select the correct shader
        if self.is_dome:
            self.frag_shader = commdef + dome
        else:
            self.frag_shader = commdef + equi
        if not self.no_side_images:
            self.frag_shader += fetch_sides
        if not self.no_top_bottom_images:
            self.frag_shader += fetch_top_bottom
        if self.no_back_image:
            self.frag_shader += fetch_front_only
        else:
            self.frag_shader += fetch_front_back
        if self.is_dome:
            # Insert the FOV and Mode  into the shader
            self.frag_shader = self.frag_shader.format(
                self.HFOV / 360.0, min(1.0, (self.HFOV - 90.0) / 180.0), domemodes[int(self.domeMode)]
            )
        else:
            # Insert the FOV into the shader
            self.frag_shader = self.frag_shader.format(self.HFOV / 360.0, min(1.0, (self.HFOV - 90.0) / 180.0))
        
        # Set the image name to the current time
        self.start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # get folder name from outside
        self.folder_name = folder
        
        # Get initial camera and output information
        ##### newly edited codes begin
        # now origin camera data not need store, and no more need to use empty as proxy
        self.camera_rotation = list(self.camera.rotation_euler)
        self.IPD = self.camera.data.stereo.interocular_distance
        #self.camera_type = self.camera.data.type
        #self.camera_angle = self.camera.data.angle
        #self.stereo_type = self.camera.data.stereo.convergence_mode
        #self.stereo_pivot = self.camera.data.stereo.pivot
        
        # Create an empty to be used to control the camera
        #try:
        #    self.camera_empty = bpy.data.objects['eeVR_CAMERA_EMPTY']
        #except KeyError:
        #    self.camera_empty = bpy.data.objects.new('eeVR_CAMERA_EMPTY', None)
        
        # Create a copy transforms constraint for the camera
        #self.trans_constraint = self.camera.constraints.new('COPY_TRANSFORMS')
        ##### newly edited codes end
        
        #self.camera.rotation_euler
        
        # Set camera variables for proper result
        self.camera.data.type = 'PANO'
        self.camera.data.stereo.convergence_mode = 'PARALLEL'
        self.camera.data.stereo.pivot = 'CENTER'
        self.camera.data.angle = pi/2
        
        self.image_size = (self.scene.render.resolution_x, self.scene.render.resolution_y)
        self.org_resolution_percentage = self.scene.render.resolution_percentage
        
        self.side_resolution = int(max(self.image_size)+4-max(self.image_size)%4)/2 if max(self.image_size)%4 > 0\
                               else int(max(self.image_size)/2)
        if self.is_stereo:
            self.view_format = self.scene.render.image_settings.views_format
            self.scene.render.image_settings.views_format = 'STEREO_3D'
            self.stereo_mode = self.scene.render.image_settings.stereo_3d_format.display_mode
            self.scene.render.image_settings.stereo_3d_format.display_mode = 'TOPBOTTOM'

        self.direction_offsets = self.find_direction_offsets()
        if self.no_back_image:
            fract = (self.HFOV-90)/180
            self.camera_shift = {'top':[0.0, 0.5*(fract-1), self.side_resolution, fract*self.side_resolution],\
                                 'bottom':[0.0, 0.5*(1-fract), self.side_resolution, fract*self.side_resolution],\
                                 'left':[0.5*(1-fract), 0.0, fract*self.side_resolution, self.side_resolution],\
                                 'right':[0.5*(fract-1), 0.0, fract*self.side_resolution, self.side_resolution],\
                                 'front':[0.0, 0.0, self.side_resolution, self.side_resolution]}
    
    
    def cubemap_to_panorama(self, imageList, outputName):
        
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
        
        batch = batch_for_shader(shader, 'TRIS', {"aVertexPosition": pos,\
                                                  "aVertexTextureCoord": coords},\
                                                  indices=vertexIndices)
        
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
        
        ##### newly added codes begin
        # update location and rotation of our camera from origin one
        self.camera.matrix_world = self.camera_origin.matrix_world
        ##### newly added codes end
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
        #self.camera_empty.rotation_euler = self.direction_offsets[direction]
        self.camera.rotation_euler = self.direction_offsets[direction]
        
        if self.no_back_image:
            self.camera.data.shift_x = self.camera_shift[direction][0]
            self.camera.data.shift_y = self.camera_shift[direction][1]
            self.scene.render.resolution_x = int(self.camera_shift[direction][2])
            self.scene.render.resolution_y = int(self.camera_shift[direction][3])
            self.scene.render.resolution_percentage = 100
        

    def clean_up(self):
        
        # Reset all the variables that were changed
        ##### newly edited codes begin
        #self.camera.constraints.remove(self.trans_constraint)
        #self.camera.data.type = self.camera_type
        #self.camera.data.stereo.convergence_mode = self.stereo_type
        #self.camera.data.stereo.pivot = self.stereo_pivot
        #self.camera.data.angle = self.camera_angle
        #self.camera.rotation_euler = self.camera_rotation
        #self.camera.data.shift_x = 0
        #self.camera.data.shift_y = 0
        bpy.context.scene.camera = self.camera_origin
        bpy.data.objects.remove(self.camera)
        ##### newly edited codes end
        self.scene.render.resolution_x = self.image_size[0]
        self.scene.render.resolution_y = self.image_size[1]
        self.scene.render.resolution_percentage = self.org_resolution_percentage
        if self.is_stereo:
            self.scene.render.image_settings.views_format = self.view_format
            self.scene.render.image_settings.stereo_3d_format.display_mode = self.stereo_mode
        for filename in self.createdFiles:
            os.remove(filename)
    
    
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
            #tmp_loc = list(self.camera_empty.location)
            tmp_loc = list(self.camera.location)
            camera_angle = self.direction_offsets['front'][2]
            #self.camera_empty.location = [tmp_loc[0]+(0.5*self.IPD*cos(camera_angle)),\
            #                        tmp_loc[1]+(0.5*self.IPD*sin(camera_angle)),\
            #                        tmp_loc[2]]
            self.camera.location = [tmp_loc[0]+(0.5*self.IPD*cos(camera_angle)),\
                                    tmp_loc[1]+(0.5*self.IPD*sin(camera_angle)),\
                                    tmp_loc[2]]

            self.scene.render.filepath = self.path + nameL + '.png'
            bpy.ops.render.render(write_still=True)
            self.createdFiles.add(self.scene.render.filepath)
            renderedImageL = bpy.data.images.load(self.scene.render.filepath)
            renderedImageL.name = nameL
            
            #self.camera_empty.location = [tmp_loc[0]-(0.5*self.IPD*cos(camera_angle)),\
            #                        tmp_loc[1]-(0.5*self.IPD*sin(camera_angle)),\
            #                        tmp_loc[2]]
            self.camera.location = [tmp_loc[0]-(0.5*self.IPD*cos(camera_angle)),\
                                    tmp_loc[1]-(0.5*self.IPD*sin(camera_angle)),\
                                    tmp_loc[2]]

            self.scene.render.filepath = self.path + nameR + '.png'
            bpy.ops.render.render(write_still=True)
            self.createdFiles.add(self.scene.render.filepath)
            renderedImageR = bpy.data.images.load(self.scene.render.filepath)
            renderedImageR.name = nameR

            self.scene.render.use_multiview = True
            #self.camera_empty.location = tmp_loc
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
            if self.no_back_image and direction in {'top', 'bottom'}:
                renderedImageL = bpy.data.images.new(nameL, self.side_resolution, int(self.side_resolution/2))
                renderedImageR = bpy.data.images.new(nameR, self.side_resolution, int(self.side_resolution/2))
            else:
                renderedImageL = bpy.data.images.new(nameL, self.side_resolution, self.side_resolution)
                renderedImageR = bpy.data.images.new(nameR, self.side_resolution, self.side_resolution)
            
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
        #self.camera_empty.location = self.camera.location
        #self.camera_empty.rotation_euler = self.camera.rotation_euler
        #self.trans_constraint.target = self.camera_empty
        for direction in reversed(directions): # I want the results of the front camera to remain in the render window... just that.
            self.set_camera_direction(direction)
            img1, img2 = self.render_image(direction)
            image_list_1.insert(0, img1)
            image_list_2.insert(0, img2)
        
        self.set_camera_direction('front')
        #self.trans_constraint.target = None
        
        return image_list_1, image_list_2


    def render_and_save(self):

        # Set the render resolution dimensions to the maximum of the two input dimensions
        self.scene.render.resolution_x = self.side_resolution
        self.scene.render.resolution_y = self.side_resolution
        self.scene.render.resolution_percentage = 100
        self.camera.data.shift_x = 0
        self.camera.data.shift_y = 0
        
        frame_step = self.scene.frame_step
        
        # Render the images and return their names
        imageList, imageList2 = self.render_images()
        if self.is_animation:
            image_name = f"frame{self.scene.frame_current:06d}.png"
        else:
            image_name = f"Render Result {self.start_time}.png"
        
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
        
        bpy.data.images.remove(imageResult)
 