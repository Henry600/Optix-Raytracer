// application_camera.cpp — free-fly camera controller and camera-node sync.

#include "application.h"

#include <algorithm>
#include <cmath>
#include <string>

void Application::syncFlyCameraFromNode(int nodeIdx)
{
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(m_scene->nodes().size()))
    {
        return;
    }
    if (std::string(m_scene->nodeAt(nodeIdx).typeName()) != "Camera")
    {
        return;
    }

    // Extract position, yaw, and pitch from the node's world-space transform.
    // This mirrors the one-time extraction in loadScene() so that editing a
    // CameraNode via the gizmo or TRS sliders immediately moves the camera.
    const Matrix4x4 world = m_scene->computeWorldTransform(nodeIdx);
    m_camPos = { world.m[0][3], world.m[1][3], world.m[2][3] };

    // Camera looks down its local -Z axis; column 2 is that -Z in world space.
    const float fx = -world.m[0][2];
    const float fy = -world.m[1][2];
    const float fz = -world.m[2][2];
    const float fLen = std::max(1e-6f, sqrtf(fx*fx + fy*fy + fz*fz));
    m_camPitch = asinf(std::max(-1.0f, std::min(1.0f, fy / fLen)));
    m_camYaw   = atan2f(fx / fLen, -(fz / fLen));
}

// ─── Camera controller ───────────────────────────────────────────────────────

void Application::updateCamera()
{
    ImGuiIO& io = ImGui::GetIO();

    // ── Mouse delta ───────────────────────────────────────────────────────────
    double mouseX, mouseY;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);
    const float dx = static_cast<float>(mouseX - m_prevMouseX);
    const float dy = static_cast<float>(mouseY - m_prevMouseY);
    m_prevMouseX = mouseX;
    m_prevMouseY = mouseY;

    const bool rmb          = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool rmbFirstFrame = rmb && !m_prevRmb;  // true only on the press event
    m_prevRmb = rmb;

    // ── Rotation / Orbit — right-drag while the Viewport panel is under the cursor
    // Ctrl+RMB: orbit around m_orbitPivot, which is kept current by every action
    //           that should change it (node selection, middle-click focus pick).
    // Shift+RMB: rotate the environment map azimuthally.
    // Plain RMB: free-look (rotate orientation in place, position fixed).
    // rmbFirstFrame is skipped to avoid a position-jump on the first drag frame.
    if (rmb && !rmbFirstFrame && m_viewportHovered)
    {
        const bool shiftHeld = glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS
                            || glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS;
        const bool ctrlHeld  = glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS
                            || glfwGetKey(m_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        if (shiftHeld)
        {
            // Rotate environment map azimuthally — same sensitivity as camera rotation.
            // Wrapped to (-pi, pi] so it stays in range for the Environment panel's slider.
            m_envMapRotation += dx * m_rotSpeed;
            m_envMapRotation = fmodf(m_envMapRotation + 3.14159265f, 6.28318531f) - 3.14159265f;
            m_accumDirty = true;
        }
        else if (ctrlHeld)
        {
            const float3 pivot = m_orbitPivot;

            // Express camera position in spherical coords relative to pivot.
            const float ox = m_camPos.x - pivot.x;
            const float oy = m_camPos.y - pivot.y;
            const float oz = m_camPos.z - pivot.z;
            const float r  = sqrtf(ox*ox + oy*oy + oz*oz);

            if (r > 1e-4f)
            {
                float azimuth   = atan2f(ox, oz);
                float elevation = asinf(std::max(-1.0f, std::min(1.0f, oy / r)));

                azimuth   -= dx * m_rotSpeed;
                elevation += dy * m_rotSpeed;

                const float kPoleLimit = 1.5533430f;  // 89°
                elevation = std::max(-kPoleLimit, std::min(kPoleLimit, elevation));

                m_camPos.x = pivot.x + r * cosf(elevation) * sinf(azimuth);
                m_camPos.y = pivot.y + r * sinf(elevation);
                m_camPos.z = pivot.z + r * cosf(elevation) * cosf(azimuth);

                // Apply the same delta to yaw/pitch so the view direction
                // rotates with the position — no re-centering on the pivot.
                m_camYaw   += dx * m_rotSpeed;
                m_camPitch -= dy * m_rotSpeed;
                m_camPitch  = std::max(-kPoleLimit, std::min(kPoleLimit, m_camPitch));
            }
        }
        else
        {
            // Free-look: update view direction, position stays fixed.
            m_camYaw   += dx * m_rotSpeed;
            m_camPitch -= dy * m_rotSpeed;

            // Clamp pitch to just under ±90° to avoid gimbal singularity
            const float kPitchLimit = 1.5533430f;  // 89 degrees in radians
            m_camPitch = std::max(-kPitchLimit, std::min(kPitchLimit, m_camPitch));
        }
    }

    // ── Translation — WASD in camera space ───────────────────────────────────
    if (!io.WantCaptureKeyboard)
    {
        // dt clamped so an initial stall frame doesn't teleport the camera
        const float dt   = std::max(0.001f, std::min(m_frameTimeMs * 0.001f, 0.1f));
        const float dist = m_moveSpeed * dt;

        const float sy = sinf(m_camYaw),   cy = cosf(m_camYaw);
        const float sp = sinf(m_camPitch),  cp = cosf(m_camPitch);

        // forward = direction the camera looks, right = camera's local +X, up = camera's local +Y
        const float3 forward = {  sy*cp,  sp, -cy*cp };
        const float3 right   = {  cy,    0.0f,  sy    };
        const float3 up      = { -sy*sp,  cp,  cy*sp };

        const bool wKey = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
        const bool sKey = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
        const bool aKey = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
        const bool dKey = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
        const bool eKey = glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS;
        const bool qKey = glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS;

        if (wKey || sKey)
        {
            const float fwd = wKey ? dist : -dist;
            m_camPos.x += forward.x * fwd;
            m_camPos.y += forward.y * fwd;
            m_camPos.z += forward.z * fwd;
        }
        if (aKey || dKey)
        {
            const float strafe = dKey ? dist : -dist;
            m_camPos.x += right.x * strafe;
            m_camPos.z += right.z * strafe;
        }
        if (eKey || qKey)
        {
            const float lift = eKey ? dist : -dist;
            m_camPos.x += up.x * lift;
            m_camPos.y += up.y * lift;
            m_camPos.z += up.z * lift;
        }
    }

    // ── Rebuild camera-to-world matrix ────────────────────────────────────────
    // Row-major Matrix4x4, columns are world-space camera axes:
    //   col 0 = right   = {cy,      0,     sy    }
    //   col 1 = up      = {-sy*sp,  cp,    cy*sp }
    //   col 2 = +Z cam  = {-sy*cp, -sp,    cy*cp } (camera looks down -Z)
    //   col 3 = pos
    const float sy = sinf(m_camYaw),   cy = cosf(m_camYaw);
    const float sp = sinf(m_camPitch),  cp = cosf(m_camPitch);

    Camera cam = m_scene->camera();
    cam.transform.m[0][0] =  cy;  cam.transform.m[0][1] = -sy*sp; cam.transform.m[0][2] = -sy*cp; cam.transform.m[0][3] = m_camPos.x;
    cam.transform.m[1][0] = 0.0f; cam.transform.m[1][1] =  cp;    cam.transform.m[1][2] = -sp;    cam.transform.m[1][3] = m_camPos.y;
    cam.transform.m[2][0] =  sy;  cam.transform.m[2][1] =  cy*sp; cam.transform.m[2][2] =  cy*cp; cam.transform.m[2][3] = m_camPos.z;
    cam.transform.m[3][0] = 0.0f; cam.transform.m[3][1] =  0.0f;  cam.transform.m[3][2] =  0.0f;  cam.transform.m[3][3] = 1.0f;
    m_scene->setCamera(std::move(cam));
}
