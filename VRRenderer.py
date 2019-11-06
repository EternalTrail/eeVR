import bpy
import os
import gpu
import bgl
from bpy.types import Operator, Panel
from math import sin, cos, pi
from datetime import datetime
from gpu_extras.batch import batch_for_shader

class VRRenderer:
    
    def __init__(self, is_stereo = False, is_animation = False, mode = 'EQUI', FOV = 180):
        
        # Check if the file is saved or not, can cause errors when not saved
        if not bpy.data.is_saved:
            raise PermissionError("Save file before rendering")
        
        # Set internal variables for the class
        self.camera = bpy.context.scene.camera
        self.path = bpy.path.abspath("//")
        self.is_stereo = is_stereo
        self.is_animation = is_animation
        self.FOV = FOV
        self.no_back_image = (self.FOV <= 270)
        self.no_side_images = (self.FOV <= 90) # TODO - Not implemented yet
        self.is_dome = (mode == 'DOME')
        self.createdFiles = set()
        
        # Get initial camera and output information
        self.camera_rotation = list(self.camera.rotation_euler)
        self.IPD = self.camera.data.stereo.interocular_distance
        self.camera_type = self.camera.data.type
        self.camera_angle = self.camera.data.angle
        self.stereo_type = self.camera.data.stereo.convergence_mode
        self.stereo_pivot = self.camera.data.stereo.pivot
        
        # Set camera variables for proper result
        self.camera.data.type = 'PANO'
        self.camera.data.stereo.convergence_mode = 'PARALLEL'
        self.camera.data.stereo.pivot = 'CENTER'
        self.camera.data.angle = pi/2
        
        self.image_size = [bpy.context.scene.render.resolution_x,\
                           bpy.context.scene.render.resolution_y]
        
        self.side_resolution = int(max(self.image_size)+4-max(self.image_size)%4)/2 if max(self.image_size)%4 > 0\
                               else int(max(self.image_size)/2)
        if self.is_stereo:
            self.view_format = bpy.context.scene.render.image_settings.views_format
            bpy.context.scene.render.image_settings.views_format = 'STEREO_3D'
            self.stereo_mode = bpy.context.scene.render.image_settings.stereo_3d_format.display_mode
            bpy.context.scene.render.image_settings.stereo_3d_format.display_mode = 'TOPBOTTOM'

        self.direction_offsets = self.find_direction_offsets()
        if self.no_back_image:
            fract = (self.FOV-90)/180
            self.camera_shift = {'top':[0.0, 0.5*(fract-1), self.side_resolution, fract*self.side_resolution],\
                                 'bottom':[0.0, 0.5*(1-fract), self.side_resolution, fract*self.side_resolution],\
                                 'left':[0.5*(1-fract), 0.0, fract*self.side_resolution, self.side_resolution],\
                                 'right':[0.5*(fract-1), 0.0, fract*self.side_resolution, self.side_resolution],\
                                 'front':[0.0, 0.0, self.side_resolution, self.side_resolution]}
    
    
    def cubemap_to_equirectangular(self, imageList, outputName):
        
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
        
        # Define the fragment shader for the 360 degree conversion
        fragment_shader_360 = '''
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

            void main() {
            
                // Calculate the pointing angle
                float azimuth = vTexCoord.x * PI;
                float elevation = vTexCoord.y * PI / 2.0;
                
                // Calculate the pointing vector
                vec3 pt;
                pt.x = cos(elevation) * sin(azimuth);
                pt.y = sin(elevation);
                pt.z = cos(elevation) * cos(azimuth);
                
                // Select the correct pixel
                if ((abs(pt.x) >= abs(pt.y)) && (abs(pt.x) >= abs(pt.z))) {
                    if (pt.x <= 0.0) {
                        fragColor = texture(cubeLeftImage, vec2(((-pt.z/pt.x)+1.0)/2.0,((-pt.y/pt.x)+1.0)/2.0));
                    } else {
                        fragColor = texture(cubeRightImage, vec2(((-pt.z/pt.x)+1.0)/2.0,((pt.y/pt.x)+1.0)/2.0));
                    }
                } else if (abs(pt.y) >= abs(pt.z)) {
                    if (pt.y <= 0.0) {
                        fragColor = texture(cubeBottomImage, vec2(((-pt.x/pt.y)+1.0)/2.0,((-pt.z/pt.y)+1.0)/2.0));
                    } else {
                        fragColor = texture(cubeTopImage, vec2(((pt.x/pt.y)+1.0)/2.0,((-pt.z/pt.y)+1.0)/2.0));
                    }
                } else {
                    if (pt.z <= 0.0) {
                        fragColor = texture(cubeBackImage, vec2(((pt.x/pt.z)+1.0)/2.0,((-pt.y/pt.z)+1.0)/2.0));
                    } else {
                        fragColor = texture(cubeFrontImage, vec2(((pt.x/pt.z)+1.0)/2.0,((pt.y/pt.z)+1.0)/2.0));
                    }
                }
            }
        '''
        
        # Define the fragment shader for the 180 degree conversion
        fragment_shader_180 = '''
            #define PI 3.1415926535897932384626
            
            // Input cubemap textures
            uniform sampler2D cubeLeftImage;
            uniform sampler2D cubeRightImage;
            uniform sampler2D cubeBottomImage;
            uniform sampler2D cubeTopImage;
            uniform sampler2D cubeFrontImage;

            in vec2 vTexCoord;

            out vec4 fragColor;

            void main() {{
            
                // Calculate the pointing angle
                float fovd = {0};
                float fovfrac = fovd/360.0;
                float sidefrac = (fovd-90.0)/180;
                float azimuth = vTexCoord.x * PI * fovfrac;
                float elevation = vTexCoord.y * PI / 2.0;
                
                // Calculate the pointing vector
                vec3 pt;
                pt.x = cos(elevation) * sin(azimuth);
                pt.y = sin(elevation);
                pt.z = cos(elevation) * cos(azimuth);
                
                // Select the correct pixel
                if ((abs(pt.x) >= abs(pt.y)) && (abs(pt.x) >= abs(pt.z))) {{
                    if (pt.x <= 0.0) {{
                        fragColor = texture(cubeLeftImage, vec2((((-pt.z/pt.x))+(2.0*sidefrac-1.0))/(2.0*sidefrac),((-pt.y/pt.x)+1.0)/2.0));
                    }} else {{
                        fragColor = texture(cubeRightImage, vec2(((-pt.z/pt.x)+1.0)/(2.0*sidefrac),((pt.y/pt.x)+1.0)/2.0));
                    }}
                }} else if (abs(pt.y) >= abs(pt.z)) {{
                    if (pt.y <= 0.0) {{
                        fragColor = texture(cubeBottomImage, vec2(((-pt.x/pt.y)+1.0)/2.0,((-pt.z/pt.y)+(2.0*sidefrac-1.0))/(2.0*sidefrac)));
                    }} else {{
                        fragColor = texture(cubeTopImage, vec2(((pt.x/pt.y)+1.0)/2.0,((-pt.z/pt.y)+1.0)/(2.0*sidefrac)));
                    }}
                }} else {{
                    fragColor = texture(cubeFrontImage, vec2(((pt.x/pt.z)+1.0)/2.0,((pt.y/pt.z)+1.0)/2.0));
                }}
            }}
        '''.format(self.FOV)

        # Define the fragment shader for the dome conversion
        fragment_shader_dome = '''
            #define PI 3.1415926535897932384626
            
            // Input cubemap textures
            uniform sampler2D cubeLeftImage;
            uniform sampler2D cubeRightImage;
            uniform sampler2D cubeBottomImage;
            uniform sampler2D cubeTopImage;
            uniform sampler2D cubeFrontImage;
            uniform sampler2D cubeBackImage;

            in vec2 vTexCoord;

            out vec4 fragColor;

            void main() {{

                float fovd = {0};
                float fovfrac = fovd/360.0;
                float sidefrac = (fovd-90.0)/180;
                float hfov = fovfrac*PI;
                vec2 d = vTexCoord.xy;

                float r = length( d );
                if( r > 1.0 ) {{
                    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                    return;
                }}
                
                vec2 dunit = normalize( d );
                float phi = r * hfov;
                vec3 pt = vec3( 1.0, 1.0, 1.0 );
                pt.xy = dunit * sin( phi );
                pt.z = cos( phi );  // Select the correct pixel
                
                // Select the correct pixel
                if ((abs(pt.x) >= abs(pt.y)) && (abs(pt.x) >= abs(pt.z))) {{
                    if (pt.x <= 0.0) {{
                        fragColor = texture(cubeLeftImage, vec2((((-pt.z/pt.x))+(2.0*sidefrac-1.0))/(2.0*sidefrac),((-pt.y/pt.x)+1.0)/2.0));
                    }} else {{
                        fragColor = texture(cubeRightImage, vec2(((-pt.z/pt.x)+1.0)/(2.0*sidefrac),((pt.y/pt.x)+1.0)/2.0));
                    }}
                }} else if (abs(pt.y) >= abs(pt.z)) {{
                    if (pt.y <= 0.0) {{
                        fragColor = texture(cubeBottomImage, vec2(((-pt.x/pt.y)+1.0)/2.0,((-pt.z/pt.y)+(2.0*sidefrac-1.0))/(2.0*sidefrac)));
                    }} else {{
                        fragColor = texture(cubeTopImage, vec2(((pt.x/pt.y)+1.0)/2.0,((-pt.z/pt.y)+1.0)/(2.0*sidefrac)));
                    }}
                }} else {{
                    if (pt.z <= 0.0) {{
                        fragColor = texture(cubeBackImage, vec2(((pt.x/pt.z)+1.0)/2.0,((-pt.y/pt.z)+1.0)/2.0));
                    }} else {{
                        fragColor = texture(cubeFrontImage, vec2(((pt.x/pt.z)+1.0)/2.0,((pt.y/pt.z)+1.0)/2.0));
                    }}
                }}
            }}
        '''.format(self.FOV)
        
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
        if self.is_dome:
            shader = gpu.types.GPUShader(vertex_shader, fragment_shader_dome)
        else:
            if self.no_back_image:
                shader = gpu.types.GPUShader(vertex_shader, fragment_shader_180)
            else:
                shader = gpu.types.GPUShader(vertex_shader, fragment_shader_360)
        batch = batch_for_shader(shader, 'TRIS', {"aVertexPosition": pos,\
                                                  "aVertexTextureCoord": coords},\
                                                  indices=vertexIndices)
        
        # Change the color space of all of the images to Linear
        # and load them into OpenGL textures
        imageLeft = imageList[0]
        imageLeft.colorspace_settings.name='Linear'
        imageLeft.gl_load()

        imageRight = imageList[1]
        imageRight.colorspace_settings.name='Linear'
        imageRight.gl_load()

        imageBottom = imageList[2]
        imageBottom.colorspace_settings.name='Linear'
        imageBottom.gl_load()

        imageTop = imageList[3]
        imageTop.colorspace_settings.name='Linear'
        imageTop.gl_load()

        imageFront = imageList[4]
        imageFront.colorspace_settings.name='Linear'
        imageFront.gl_load()
        
        if not self.no_back_image:
            imageBack = imageList[5]
            imageBack.colorspace_settings.name='Linear'
            imageBack.gl_load()
        
        # set the size of the final image
        width = self.image_size[0]
        height = self.image_size[1]

        # Create an offscreen render buffer and texture
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            bgl.glClear(bgl.GL_COLOR_BUFFER_BIT)

            shader.bind()
            
            # Bind all of the cubemap textures and enable correct filtering and wrapping
            # to prevent seams
            bgl.glActiveTexture(bgl.GL_TEXTURE0)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, imageLeft.bindcode)
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
            shader.uniform_int("cubeLeftImage", 0)
            
            bgl.glActiveTexture(bgl.GL_TEXTURE1)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, imageRight.bindcode)
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
            shader.uniform_int("cubeRightImage", 1)
            
            bgl.glActiveTexture(bgl.GL_TEXTURE2)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, imageBottom.bindcode)
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
            shader.uniform_int("cubeBottomImage", 2)
            
            bgl.glActiveTexture(bgl.GL_TEXTURE3)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, imageTop.bindcode)
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
            shader.uniform_int("cubeTopImage", 3)
            
            bgl.glActiveTexture(bgl.GL_TEXTURE4)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, imageFront.bindcode)
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
            shader.uniform_int("cubeFrontImage", 4)
            
            if not self.no_back_image:
                bgl.glActiveTexture(bgl.GL_TEXTURE5)
                bgl.glBindTexture(bgl.GL_TEXTURE_2D, imageBack.bindcode)
                bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR );
                bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR );
                bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
                bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
                shader.uniform_int("cubeBackImage", 5)
            
            
            # Bind the resulting texture
            bgl.glActiveTexture(bgl.GL_TEXTURE6)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, offscreen.color_texture)
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR);
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
            bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
            
            # Render the image
            batch.draw(shader)
            
            # Unload the textures
            imageLeft.gl_free()
            imageRight.gl_free()
            imageBottom.gl_free()
            imageTop.gl_free()
            imageFront.gl_free()
            if not self.no_back_image:
                imageBack.gl_free()
            
            # Read the resulting pixels into a buffer
            buffer = bgl.Buffer(bgl.GL_FLOAT, width * height * 4)
            bgl.glGetTexImage(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA, bgl.GL_FLOAT, buffer);

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
        imageRes.pixels = buffer
        return imageRes

    
    def find_direction_offsets(self):
        
        # Calculate the pointing directions of the camera for each face of the cube
        self.camera_rotation = list(self.camera.rotation_euler)
        rotation = self.camera_rotation
        direction_offsets = {}
        direction_offsets['front'] = [rotation[0], 0,\
                                      rotation[2]]
        direction_offsets['back'] = [pi-rotation[0], 0,\
                                     pi+rotation[2]]
        direction_offsets['top'] = [rotation[0]+pi/2, 0,\
                                   rotation[2]]
        direction_offsets['bottom'] = [rotation[0]-pi/2, 0,\
                                     rotation[2]]
        direction_offsets['left'] = [pi/2,\
                                     pi/2-rotation[0],\
                                     rotation[2]+pi/2]
        direction_offsets['right'] = [pi/2,\
                                      rotation[0]-pi/2,\
                                      rotation[2]-pi/2]
        
        return direction_offsets
    
    
    def set_camera_direction(self, direction):
        
        # Set the camera to the required postion    
        self.camera.rotation_euler = self.direction_offsets[direction]
        
        if self.no_back_image:
            self.camera.data.shift_x = self.camera_shift[direction][0]
            self.camera.data.shift_y = self.camera_shift[direction][1]
            bpy.context.scene.render.resolution_x = self.camera_shift[direction][2]
            bpy.context.scene.render.resolution_y = self.camera_shift[direction][3]
        

    def clean_up(self):
        
        # Reset all the variables that were changed
        self.camera.data.type = self.camera_type
        self.camera.data.stereo.convergence_mode = self.stereo_type
        self.camera.data.stereo.pivot = self.stereo_pivot
        self.camera.data.angle = self.camera_angle
        self.camera.rotation_euler = self.camera_rotation
        self.camera.data.shift_x = 0
        self.camera.data.shift_y = 0
        bpy.context.scene.render.resolution_x = self.image_size[0]
        bpy.context.scene.render.resolution_y = self.image_size[1]
        if self.is_stereo:
            bpy.context.scene.render.image_settings.views_format = self.view_format
            bpy.context.scene.render.image_settings.stereo_3d_format.display_mode = self.stereo_mode
        for filename in self.createdFiles:
            os.remove(filename)
    
    
    def render_image(self, direction):
        
        # Render the image and load it into the script
        tmp = bpy.data.scenes['Scene'].render.filepath
        bpy.data.scenes['Scene'].render.filepath = self.path + 'temp_img_store_'+direction+'.png'
        
        # If rendering for VR, render the side images separately to avoid seams
        if self.is_stereo and direction in {'right', 'left'}:
            imageL = 'temp_img_store_'+direction+'_L.png'
            imageR = 'temp_img_store_'+direction+'_R.png'
            if imageL in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[imageL])
            if imageR in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[imageR])
            
            bpy.context.scene.render.use_multiview = False
            tmp_loc = list(self.camera.location)
            camera_angle = self.direction_offsets['front'][2]
            self.camera.location = [tmp_loc[0]+(0.5*self.IPD*cos(camera_angle)),\
                                    tmp_loc[1]+(0.5*self.IPD*sin(camera_angle)),\
                                    tmp_loc[2]]
            bpy.data.scenes['Scene'].render.filepath = self.path + imageL
            bpy.ops.render.render(write_still=True)
            renderedImageL = bpy.data.images.load(self.path + imageL)
            
            self.camera.location = [tmp_loc[0]-(0.5*self.IPD*cos(camera_angle)),\
                                    tmp_loc[1]-(0.5*self.IPD*sin(camera_angle)),\
                                    tmp_loc[2]]
            bpy.data.scenes['Scene'].render.filepath = self.path + imageR
            bpy.ops.render.render(write_still=True)
            renderedImageR = bpy.data.images.load(self.path + imageR)
            bpy.context.scene.render.use_multiview = True
            self.createdFiles.update({self.path+imageR, self.path+imageL})
            self.camera.location = tmp_loc
        
        elif self.is_stereo:
            bpy.ops.render.render(write_still=True)
            image_name = 'temp_img_store_'+direction+'.png'
            imageL = 'temp_img_store_'+direction+'_L.png'
            imageR = 'temp_img_store_'+direction+'_R.png'
            if image_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[image_name])
            if imageL in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[imageL])
            if imageR in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[imageR])
            renderedImage =  bpy.data.images.load(self.path + image_name)
            imageLen = len(renderedImage.pixels)
            if self.no_back_image and direction in {'top', 'bottom'}:
                renderedImageL = bpy.data.images.new(imageL, self.side_resolution,\
                                                     int(self.side_resolution/2))
                renderedImageR = bpy.data.images.new(imageR, self.side_resolution,\
                                                     int(self.side_resolution/2))
            else:
                renderedImageL = bpy.data.images.new(imageL, self.side_resolution, self.side_resolution)
                renderedImageR = bpy.data.images.new(imageR, self.side_resolution, self.side_resolution)
            
            # Split the render into two images
            if direction == 'back':
                renderedImageL.pixels = renderedImage.pixels[int(imageLen/2):]
                renderedImageR.pixels = renderedImage.pixels[0:int(imageLen/2)]
            else:
                renderedImageR.pixels = renderedImage.pixels[int(imageLen/2):]
                renderedImageL.pixels = renderedImage.pixels[0:int(imageLen/2)]
            renderedImageL.pack()
            renderedImageR.pack()
            bpy.data.images.remove(renderedImage)
            self.createdFiles.add(self.path + 'temp_img_store_'+direction+'.png')
        else:
            bpy.ops.render.render(write_still=True)
            image_name = 'temp_img_store_'+direction+'.png'
            if image_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[image_name])
            renderedImageL = bpy.data.images.load(self.path + image_name)
            renderedImageR = None
            self.createdFiles.add(self.path + 'temp_img_store_'+direction+'.png')
        
        bpy.data.scenes['Scene'].render.filepath = tmp
        return renderedImageL, renderedImageR
    
    
    def render_images(self):
        
        # Render the images for every direction
        directions = ['left', 'right', 'bottom', 'top', 'front', 'back']
        image_list_1 = []
        image_list_2 = []
        
        self.direction_offsets = self.find_direction_offsets()
        for direction in directions:
            if direction == 'back' and self.no_back_image:
                continue
            else:
                self.set_camera_direction(direction)
                img1, img2 = self.render_image(direction)
                image_list_1.append(img1)
                image_list_2.append(img2)
        
        self.set_camera_direction('front')
        
        return image_list_1, image_list_2
    
    
    def render_and_save(self):
        
        # Set the render resolution dimensions to the maximum of the two input dimensions
        bpy.context.scene.render.resolution_x = self.side_resolution
        bpy.context.scene.render.resolution_y = self.side_resolution
        self.camera.data.shift_x = 0
        self.camera.data.shift_y = 0 
        
        # Set the image/folder name to the current time
        time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        
        # Render the image or video
        if self.is_animation:
            
            # Find which frames need to be rendered
            frame_start = bpy.context.scene.frame_start
            frame_end = bpy.context.scene.frame_end
            frame_step = bpy.context.scene.frame_step
            bpy.context.scene.frame_set(frame_start)
            
            # Render and store every frame
            i = 1
            while bpy.context.scene.frame_current <= frame_end:
                
                # Render the images and return their names
                imageList, imageList2 = self.render_images()
                folder_name = "Render Result {}/".format(time_now)
                image_name = "frame{:06d}.png".format(i)
                
                # Convert the rendered images to equirectangular projection image and save it to the disk
                if self.is_stereo:
                    imageResult1 = self.cubemap_to_equirectangular(imageList, "Render Left")
                    imageResult2 = self.cubemap_to_equirectangular(imageList2, "Render Right")
                    
                    # If it doesn't already exist, create an image object to store the resulting render
                    if not image_name in bpy.data.images.keys():
                        imageResult = bpy.data.images.new(image_name, imageResult1.size[0],\
                                                          2*imageResult1.size[1])
                    imageResult = bpy.data.images[image_name]
                    imageResult.scale(imageResult1.size[0], 2*imageResult1.size[1])
                    imageResult.pixels = list(imageResult2.pixels) + list(imageResult1.pixels)
                    bpy.data.images.remove(imageResult1)
                    bpy.data.images.remove(imageResult2)
                    
                else:
                    imageResult = self.cubemap_to_equirectangular(imageList, "RenderResult")
                imageResult.save_render(self.path+folder_name+image_name)
                bpy.data.images.remove(imageResult)
                bpy.context.scene.frame_set(frame_start+i*frame_step)
                i += 1
                
        else:
            
            # Render the images and return their names
            imageList, imageList2 = self.render_images()
            image_name = "RenderResult"
            
            # Convert the rendered images to equirectangular projection image and save it to the disk
            if self.is_stereo:
                imageResult1 = self.cubemap_to_equirectangular(imageList, "Render Left")
                imageResult2 = self.cubemap_to_equirectangular(imageList2, "Render Right")
                
                # If it doesn't already exist, create an image object to store the resulting render
                if not image_name in bpy.data.images.keys():
                    imageResult = bpy.data.images.new(image_name, imageResult1.size[0],\
                                                      2*imageResult1.size[1])
                imageResult = bpy.data.images[image_name]
                imageResult.scale(imageResult1.size[0], 2*imageResult1.size[1])
                imageResult.pixels = list(imageResult2.pixels) + list(imageResult1.pixels)
                bpy.data.images.remove(imageResult1)
                bpy.data.images.remove(imageResult2)
                
            else:
                imageResult = self.cubemap_to_equirectangular(imageList, "RenderResult")
            imageResult.save_render(self.path+"Render Result {}.png".format(time_now))
        self.clean_up()
        
        return {'FINISHED'}


class RenderImage(Operator):
    """Render the image"""
    
    bl_idname = 'wl.render_image'
    bl_label = "Render image"
    
    def execute(self, context):
        
        mode = bpy.context.scene.renderModeEnum
        FOV = bpy.context.scene.renderFOV
        return VRRenderer(bpy.context.scene.render.use_multiview, False, mode, FOV).render_and_save()


class RenderAnimation(Operator):
    """Render out the animation"""
    
    bl_idname = 'wl.render_animation'
    bl_label = "Render the animation"
    
    def execute(self, context):
        
        mode = bpy.context.scene.renderModeEnum
        FOV = bpy.context.scene.renderFOV
        return VRRenderer(bpy.context.scene.render.use_multiview, True, mode, FOV).render_and_save()



class RenderToolsPanel(Panel):
    """Tools panel for VR rendering"""
    
    bl_idname = "RENDER_TOOLS_PT_render_tools_panel"
    bl_label = "Render Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'

    def draw(self, context):
        
        # Draw the buttons for each of the rendering operators
        layout = self.layout
        col = layout.column()
        col.prop(context.scene, 'renderModeEnum')
        col.prop(context.scene, 'renderFOV')
        col.operator("wl.render_image", text="Render Image")
        col.operator("wl.render_animation", text="Render Animation")


# Register all classes
def register():
    bpy.types.Scene.renderModeEnum = bpy.props.EnumProperty(
        items = [('EQUI', 'Equirectangular', 'Renders in equirectangular projection'),
                 ('DOME', 'Full Dome', 'Renders in full dome projection')],
        default='EQUI',
        name="Mode")
    bpy.types.Scene.renderFOV = bpy.props.FloatProperty(180.0,default=180.0, name="FOV", min=180, max=360,
                                description="Field of view in degrees")
    bpy.utils.register_class(RenderImage)
    bpy.utils.register_class(RenderAnimation)
    bpy.utils.register_class(RenderToolsPanel)
    


# Unregister all classes
def unregister():
    del bpy.types.Scene.renderModeEnum
    del bpy.types.Scene.renderFOV
    bpy.utils.unregister_class(RenderImage)
    bpy.utils.unregister_class(RenderAnimation)
    bpy.utils.unregister_class(RenderToolsPanel)


# If the script is not an addon when it is run, register the classes
if __name__=="__main__":
    register()
