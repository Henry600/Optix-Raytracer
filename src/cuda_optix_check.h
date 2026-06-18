#ifndef OPTIX_RAYTRACER_CUDA_OPTIX_CHECK_H
#define OPTIX_RAYTRACER_CUDA_OPTIX_CHECK_H

#include <cuda_runtime.h>
#include <optix.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t rc = (call);                                               \
        if (rc != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error in " __FILE__     \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + cudaGetErrorString(rc));                                     \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult rc = (call);                                               \
        if (rc != OPTIX_SUCCESS) {                                             \
            throw std::runtime_error(std::string("OptiX error in " __FILE__    \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + optixGetErrorString(rc));                                    \
        }                                                                      \
    } while (0)

#endif // OPTIX_RAYTRACER_CUDA_OPTIX_CHECK_H
