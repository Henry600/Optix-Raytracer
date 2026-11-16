# Compile device_programs.cu (which includes the headers under device/) to a single PTX module for OptiX JIT compilation at runtime.
# CUDA_PTX_COMPILATION ON  → outputs .ptx instead of .obj
# 75-virtual              → PTX only, no embedded cubin (required for OptiX)
# CUDA_SEPARABLE_COMPILATION OFF → PTX programs are self-contained entry points

add_library(device_programs OBJECT
    device_programs.cu
    device_math.h
    color_math.h
    launch_params.h
    scene_data.h
    device/dev_common.h
    device/path_payload.h
    device/rng_sampling.h
    device/pbr.h
    device/environment.h
    device/emissive_lights.h
    device/display.h
    device/eon_diffuse.h
    device/splat_gather.h
    device/path_state.h
    device/shadow_ray.h
    device/camera_ray.h
    device/path_events.h
    device/shadow_catcher.h
    device/surface_context.h
    device/coat_lobe.h
    device/specular_lobe.h
    device/diffuse_lobe.h
    device/transmission_lobe.h
    device/base_lobe.h
    device/raygen_render.h
    device/miss_shadow_programs.h
    device/material.h
    device/hit_mesh.h
    device/implicit_shapes.h
    device/hit_implicit.h
    device/splat_programs.h
    device/pick_programs.h
)

source_group("Header Files" FILES device_math.h color_math.h launch_params.h scene_data.h)
source_group("Header Files\\device" FILES
    device/dev_common.h
    device/path_payload.h
    device/rng_sampling.h
    device/pbr.h
    device/environment.h
    device/emissive_lights.h
    device/display.h
    device/eon_diffuse.h
    device/splat_gather.h
    device/path_state.h
    device/shadow_ray.h
    device/camera_ray.h
    device/path_events.h
    device/shadow_catcher.h
    device/surface_context.h
    device/coat_lobe.h
    device/specular_lobe.h
    device/diffuse_lobe.h
    device/transmission_lobe.h
    device/base_lobe.h
    device/raygen_render.h
    device/miss_shadow_programs.h
    device/material.h
    device/hit_mesh.h
    device/implicit_shapes.h
    device/hit_implicit.h
    device/splat_programs.h
    device/pick_programs.h
)
source_group("Source Files" FILES device_programs.cu)

set_property(TARGET device_programs PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET device_programs PROPERTY CUDA_ARCHITECTURES "75-virtual")
set_property(TARGET device_programs PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_include_directories(device_programs PRIVATE
    ${CMAKE_SOURCE_DIR}/shaders
    ${OptiX_INCLUDE_DIR}
)

# VS IntelliSense reads IncludePath, not CudaCompile/Include, so inject a props
# file that adds the OptiX directory to IncludePath for all configurations.
if(MSVC)
    configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/cuda_intellisense.props.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cuda_intellisense.props"
        @ONLY
    )
    set_target_properties(device_programs PROPERTIES
        VS_USER_PROPS "${CMAKE_CURRENT_BINARY_DIR}/cuda_intellisense.props"
    )
endif()
