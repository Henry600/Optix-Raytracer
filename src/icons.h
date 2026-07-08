#ifndef OPTIX_RAYTRACER_ICONS_H
#define OPTIX_RAYTRACER_ICONS_H

// FontAwesome 6 Free Solid glyph codes, UTF-8 encoded.
// Requires fa-solid-900.ttf merged into the ImGui default font
// (see Application::initImGui).  Only the codepoints used in this
// project are listed here; add more from fontawesome.com/icons as needed.

#define ICON_FA_CUBE         "\xef\x86\xb2"  // U+F1B2  mesh node
#define ICON_FA_CAMERA       "\xef\x80\xb0"  // U+F030  camera node
#define ICON_FA_LAYER_GROUP  "\xef\x97\xbd"  // U+F5FD  group node
#define ICON_FA_CIRCLE       "\xef\x84\x91"  // U+F111  implicit sphere
#define ICON_FA_SQUARE       "\xef\x83\x88"  // U+F0C8  implicit box
#define ICON_FA_DATABASE     "\xef\x87\x80"  // U+F1C0  implicit cylinder
#define ICON_FA_EYE          "\xef\x81\xae"  // U+F06E  visibility on
#define ICON_FA_EYE_SLASH    "\xef\x81\xb0"  // U+F070  visibility off
#define ICON_FA_BRAILLE      "\xef\x8a\xa1"  // U+F2A1  gaussian splat node (dot cloud)

#endif // OPTIX_RAYTRACER_ICONS_H
