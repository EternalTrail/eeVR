# eeVR
Blender addon to render 360° and 180° images and videos in eevee engine with support for stereoscopic rendering.

## Getting Started

You will need to get [**Blender 2.8X**](https://www.blender.org), install it, download the script **VRRenderer.py**, load the addon into Blender by installing the VRRenderer.py file in Blender Preferences > Add-ons > Install, search for "eeVR" under the Testing tab and click the checkbox to enable it. A tool panel will appear in the 3D Viewport with mode selection, FOV value adjustment, and buttons for rendering stills and animations. 

**Beware this is still in development so I suggest backing up whatever project you're using this on before you begin rendering.**

## What doesn't work at the moment

The script is very picky with the way the things are set up before rendering at the moment. These are the most common problems people run into:
- Output format must be PNG, or else the render will fail, and will potentially mess up some setting/positions of the camera.
- The camera cannot be a child of any object as the script internally sets an empty as its parent, your renders will turn out corrupted if your camera has another parent.
- Bloom and Ambient Occlusion will create visible seams in positions where the pictures join together, so it's best if those settings are turned off for the moment.
- Some of the rendering settings will be force set since they create issues, such as stereoscopy mode in camera settings is forced to parallel, since that was done in order to prevent seams.

These are only *some* of the things that don't work properly yet, some can be easily fixed with a few hours of coding, some will probably take much longer. As I'm only working on this project intermittently, I cannot provide a timeline of these fixes. If you have any questionm let me know!


### Keep in mind!

If you have suggestions for future updates or come across any bugs don't hesitate to open up a "new issue" in the issue tab or write me an email at [andriux19960823@gmail.com](mailto:andriux19960823@gmail.com).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* The image conversion OpenGL shader was originally created by [Xyene](https://github.com/Xyene) and modified for use in Blender so thank you.
