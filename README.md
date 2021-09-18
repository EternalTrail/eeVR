# eeVR
**This project is not currently in active development, you're still free to post issues and make pull requests and I will try my best to guide you/merge, even if you have to wait a while. I do plan on coming back to this, although that won't happen for probably a year or two.**

Blender addon to render 360° and 180° images and videos in eevee engine with support for stereoscopic rendering.

## Getting Started

You will need to get [**Blender 2.8X** or **Blender 2.9X**](https://www.blender.org), install it, download the zip file from this GitHub, load the addon into Blender by installing the zip file in Blender Preferences > Add-ons > Install, search for "eeVR" under the Testing tab and click the checkbox to enable it. A tool panel will appear in the 3D Viewport with mode selection, FOV value adjustment, and buttons for rendering stills and animations. **The rendered images/image sequences will be stored in the same directory as the .blend file**.

**VRRenderer_ALT.py** is an alternative version, it includes simplified VR renderer, useful to render previews, when view angle not bigger than 180deg (better for 120deg), it will work. And it has the **custom save folder** name option, may be useful for some situations.

**Beware this is still in development so I suggest backing up whatever project you're using this on before you begin rendering.**

## What doesn't work at the moment

The script is very picky with the way the things are set up before rendering at the moment. These are the most common problems people run into:
- Output format must be PNG, or else the render will fail, and will potentially mess up some settings of the camera.
- Screen space effects will create visible seams in positions where the pictures join together, so it's best if those settings are turned off for the moment. **A possible way to mitigate some of the seams is to use overscan which is located under _Render Properties > Film > Overscan_**.
- Some of the rendering settings will be force set since they create issues, such as stereoscopy mode in camera settings is forced to parallel, since that was done in order to prevent seams.

These are only *some* of the things that don't work properly yet, some can be easily fixed with a few hours of coding, some will probably take much longer. As I'm only working on this project intermittently, I cannot provide a timeline of these fixes. If you have any questionm let me know!


### Keep in mind!

If you have suggestions for future updates or come across any bugs don't hesitate to open up a "new issue" in the issue tab or write me an email at [andriux19960823@gmail.com](mailto:andriux19960823@gmail.com).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* The image conversion OpenGL shader was originally created by [Xyene](https://github.com/Xyene) and modified for use in Blender so thank you.
