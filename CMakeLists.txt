cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# ── Policy guards ────────────────────────────────────────────────────────────
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # FetchContent timestamp behaviour
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW) # find_package honours <Pkg>_ROOT variables
endif()
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD) # Allow FetchContent_Populate (deprecated in CMake 4.x)
endif()

# Third-party dependencies declare cmake_minimum_required versions < 3.10, which
# triggers a CMake 4.x deprecation warning that we cannot fix in their source.
# Suppress it globally — our own CMake code does not use deprecated features.
set(CMAKE_WARN_DEPRECATED OFF CACHE INTERNAL "")

# ── CUDA architectures (must be set before project()) ────────────────────────
# sm_75 = Turing (RTX 20xx)  sm_86 = Ampere (RTX 30xx)  sm_89 = Ada (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# ── Project ──────────────────────────────────────────────────────────────────
project(OptixRaytracer
    VERSION 0.1.0
    LANGUAGES CXX CUDA C  # C declared upfront so detection runs before GLFW's nested project()
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Place all runtime outputs under build/bin/<Config>/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" CFG)
    set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG}" "${CMAKE_BINARY_DIR}/bin/${cfg}")
endforeach()

# ── Custom CMake modules ──────────────────────────────────────────────────────
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ── System / SDK dependencies ─────────────────────────────────────────────────
find_package(OptiX     REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Vulkan    REQUIRED)

# ── Third-party via FetchContent ──────────────────────────────────────────────
include(FetchContent)

# On Windows/MSVC, pthreads do not exist.  Pre-populate the three cache
# variables that CMake's FindThreads module creates when its compile/link
# probes fail.  The check_*() macros skip any test whose result variable is
# already defined in the cache, so this avoids three slow probe round-trips
# (compile + link) on every clean configure.  FindThreads still finds
# Threads::Threads via the unconditional Windows fallback at the end of the
# module.  Must be set before any FetchContent_MakeAvailable call that
# internally calls find_package(Threads) — e.g. GLFW.
if(WIN32)
    set(CMAKE_HAVE_LIBC_PTHREAD    0 CACHE INTERNAL "pthread in libc (pre-populated: N/A on MSVC)")
    set(CMAKE_HAVE_PTHREADS_CREATE 0 CACHE INTERNAL "pthreads lib    (pre-populated: N/A on MSVC)")
    set(CMAKE_HAVE_PTHREAD_CREATE  0 CACHE INTERNAL "pthread lib     (pre-populated: N/A on MSVC)")
endif()

# GLFW 3.4 — windowing
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ImGui docking branch — UI with docking + multi-viewport support
# GIT_SHALLOW FALSE is required when targeting a branch rather than a tag
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        docking
    GIT_SHALLOW    FALSE
)
FetchContent_MakeAvailable(imgui)

# ImGui has no CMakeLists.txt — create the STATIC target manually
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)
target_compile_options(imgui PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W0>)

# ImGuizmo — 3D gizmo (translate / rotate / scale) overlay for Dear ImGui.
# No CMakeLists.txt; compiled as part of the imgui STATIC target so it
# shares ImGui's context and link dependencies automatically.
FetchContent_Declare(imguizmo
    GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(imguizmo)
if(NOT imguizmo_POPULATED)
    FetchContent_Populate(imguizmo)
endif()
target_sources(imgui PRIVATE ${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp)
target_include_directories(imgui PUBLIC ${imguizmo_SOURCE_DIR}/src)

# tinygltf v2.8.21 — glTF 2.0 loader (header-only; implementation defined in scene_loader.cpp)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.8.21
    GIT_SHALLOW    TRUE
)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL              OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY          ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

# Imath 3.1 — math types (half, Box2i, …) required by OpenEXR 3.x.
# Must be fetched before OpenEXR so the Imath::Imath target is already in scope
# when OpenEXR's CMakeLists checks 'if(NOT TARGET Imath::Imath)'.
FetchContent_Declare(imath
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/Imath.git
    GIT_TAG        v3.1.12
    GIT_SHALLOW    TRUE
)
set(IMATH_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(imath)

# OpenEXR 3.3 — full EXR codec support: NONE, RLE, ZIPS, ZIP, PIZ,
#               PXR24, B44, B44A, DWAA, DWAB (replaces tinyexr).
# Imath::Imath is already in CMake scope from above, so OpenEXR's
# find_package(Imath) guard is satisfied without a separate install step.
FetchContent_Declare(openexr
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
    GIT_TAG        v3.3.3
    GIT_SHALLOW    TRUE
)
set(OPENEXR_INSTALL          OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_TOOLS      OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL_TOOLS    OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(openexr)

# nativefiledialog-extended v1.2.1 — native OS file dialog (Open / Save)
FetchContent_Declare(nfd
    GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
    GIT_TAG        v1.2.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nfd)

# libwebp v1.6.0 — lossless WebP decoding for SOG gaussian splat images.
# Only the codec library is needed; all command-line tools are disabled.
FetchContent_Declare(libwebp
    GIT_REPOSITORY https://github.com/webmproject/libwebp.git
    GIT_TAG        v1.6.0
    GIT_SHALLOW    TRUE
)
set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_CWEBP      OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_DWEBP      OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_GIF2WEBP   OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_IMG2WEBP   OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_VWEBP      OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPINFO   OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPMUX    OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_EXTRAS     OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libwebp)

# miniz 3.0.2 — ZIP archive reading for bundled .sog files.  The release
# amalgamation (miniz.c + miniz.h) has no CMakeLists.txt, so the STATIC
# target is created manually — same pattern as ImGui above.
FetchContent_Declare(miniz
    URL https://github.com/richgel999/miniz/releases/download/3.0.2/miniz-3.0.2.zip
)
FetchContent_GetProperties(miniz)
if(NOT miniz_POPULATED)
    FetchContent_Populate(miniz)
endif()
add_library(miniz STATIC ${miniz_SOURCE_DIR}/miniz.c)
target_include_directories(miniz PUBLIC ${miniz_SOURCE_DIR})
target_compile_options(miniz PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W0>)

# FontAwesome 6 Free Solid — icon glyphs merged into the ImGui font atlas.
# Downloaded once at configure time into fonts/; copied to the executable
# directory at build time (see src/CMakeLists.txt).
set(FA_FONT_DIR  "${CMAKE_SOURCE_DIR}/fonts")
set(FA_FONT_PATH "${FA_FONT_DIR}/fa-solid-900.ttf")
if(NOT EXISTS "${FA_FONT_PATH}")
    message(STATUS "Downloading fa-solid-900.ttf (FontAwesome 6.7.2)")
    file(MAKE_DIRECTORY "${FA_FONT_DIR}")
    file(DOWNLOAD
        "https://github.com/FortAwesome/Font-Awesome/raw/6.7.2/webfonts/fa-solid-900.ttf"
        "${FA_FONT_PATH}"
        STATUS _fa_status
        TLS_VERIFY ON
    )
    list(GET _fa_status 0 _fa_result)
    if(NOT _fa_result EQUAL 0)
        message(WARNING "FontAwesome download failed — scene-graph icons will not appear")
        file(REMOVE "${FA_FONT_PATH}")
    endif()
endif()

# ── Group third-party targets in Solution Explorer ────────────────────────────
# GLFW's own CMakeLists turns on USE_FOLDERS (and puts its own helper targets
# under "GLFW3"); without an explicit FOLDER every other fetched target
# (tinygltf, OpenEXR/Imath, libwebp, miniz, nfd, CTest's dashboard targets, ...)
# then sits flat at the solution root.  Recursively walk every directory scope
# defined so far and fold anything that doesn't already have a FOLDER (leaving
# GLFW's own grouping untouched) under one "ThirdParty" folder.  Run before
# add_subdirectory(shaders)/(src) so our own targets are left ungrouped at the root.
function(_group_third_party_targets dir folder)
    get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_tgt IN LISTS _targets)
        get_target_property(_existing "${_tgt}" FOLDER)
        if(NOT _existing)
            set_target_properties("${_tgt}" PROPERTIES FOLDER "${folder}")
        endif()
    endforeach()

    get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(_sub IN LISTS _subdirs)
        _group_third_party_targets("${_sub}" "${folder}")
    endforeach()
endfunction()

_group_third_party_targets("${CMAKE_SOURCE_DIR}" "ThirdParty")

# ── Project subdirectories ────────────────────────────────────────────────────
add_subdirectory(shaders)   # device_programs OBJECT library (PTX)
add_subdirectory(src)       # OptixRaytracer executable

# Set OptixRaytracer as the default startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT OptixRaytracer)
