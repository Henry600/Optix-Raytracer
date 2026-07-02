#ifndef OPTIX_RAYTRACER_CAMERA_H
#define OPTIX_RAYTRACER_CAMERA_H

#include "matrix4x4.h"
#include <cmath>
#include <string>

struct Camera
{
    Matrix4x4   transform;                // camera-to-world (position + orientation in world space)
    Matrix4x4   view;                     // world-to-camera (inverse of transform; OpenGL "view matrix")

    // Physical lens / sensor parameters — these drive the FOV and depth of field.
    // Conventions: sensorSize is the horizontal sensor width in mm; 1 world unit = 1 metre.
    float focalLength   = 50.0f;   // mm  — drives FOV together with sensorSize
    float sensorSize    = 35.0f;   // mm  — sensor width (horizontal)
    float fStop         = 8.0f;    // f-number; lens radius = focalLength / (2 × fStop × 1000) m
    float focusDistance = 10.0f;   // world units to focal plane along optical axis
    float bokehEdgeBias = 0.0f;    // [0, 1]: 0 = uniform disk, 1 = pure rim ring
    bool  dofEnabled    = true;    // false → pinhole camera (lensRadius forced to 0)

    // Derived / legacy fields — kept for backward compatibility and glTF import.
    float       yFov        = 0.872665f;  // radians — NOT the primary FOV source; derived from focalLength + sensorSize
    float       aspectRatio = 16.0f / 9.0f;
    float       zNear       = 0.01f;
    float       zFar        = 10000.0f;
    std::string name;

    // Returns a default camera at (0, 0, 3) looking toward the origin along -Z.
    static Camera makeDefault();
};

inline Camera Camera::makeDefault()
{
    Camera cam;
    cam.name = "Default";

    // Camera-to-world: identity rotation + translation (0, 0, 3).
    cam.transform.m[0][0] = 1.0f;
    cam.transform.m[1][1] = 1.0f;
    cam.transform.m[2][2] = 1.0f;
    cam.transform.m[2][3] = 3.0f;
    cam.transform.m[3][3] = 1.0f;

    // View (world-to-camera) = T(0, 0, -3)
    cam.view.m[0][0] = 1.0f;
    cam.view.m[1][1] = 1.0f;
    cam.view.m[2][2] = 1.0f;
    cam.view.m[2][3] = -3.0f;
    cam.view.m[3][3] = 1.0f;

    // Focus distance = distance from camera to origin, so the default orbit
    // target (origin) is exactly in focus.
    const float px = cam.transform.m[0][3];
    const float py = cam.transform.m[1][3];
    const float pz = cam.transform.m[2][3];
    cam.focusDistance = sqrtf(px*px + py*py + pz*pz);

    // Cache yFov from physical parameters (16:9 default aspect ratio)
    const float sensorHeight = cam.sensorSize / (16.0f / 9.0f);
    cam.yFov = 2.0f * atanf(sensorHeight / (2.0f * cam.focalLength));

    return cam;
}

#endif // OPTIX_RAYTRACER_CAMERA_H
