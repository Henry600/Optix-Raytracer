#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// RAII wrapper around a CUDA device allocation (cudaMalloc / cudaFree).
// Non-copyable; movable.  Mirrors the lifetime of the object — the
// destructor always frees the buffer if one is allocated.
class GPUBuffer
{
public:
    GPUBuffer()  = default;
    ~GPUBuffer() { free(); }

    GPUBuffer(const GPUBuffer&)            = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    GPUBuffer(GPUBuffer&& o) noexcept
        : m_ptr(o.m_ptr), m_size(o.m_size)
    {
        o.m_ptr  = 0;
        o.m_size = 0;
    }

    GPUBuffer& operator=(GPUBuffer&& o) noexcept
    {
        if (this != &o)
        {
            free();
            m_ptr    = o.m_ptr;
            m_size   = o.m_size;
            o.m_ptr  = 0;
            o.m_size = 0;
        }
        return *this;
    }

    // Free any existing allocation, then allocate `bytes` on the device.
    void alloc(size_t bytes)
    {
        free();
        cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&m_ptr), bytes);
        if (rc != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("GPUBuffer::alloc failed: ") + cudaGetErrorString(rc));
        }
        m_size = bytes;
    }

    // Free the allocation.  No-op if not currently allocated.
    void free()
    {
        if (m_ptr)
        {
            cudaFree(reinterpret_cast<void*>(m_ptr));
            m_ptr  = 0;
            m_size = 0;
        }
    }

    // Copy `bytes` from `src` (host) into the device buffer.
    void upload(const void* src, size_t bytes)
    {
        cudaError_t rc = cudaMemcpy(
            reinterpret_cast<void*>(m_ptr), src, bytes, cudaMemcpyHostToDevice);
        if (rc != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("GPUBuffer::upload failed: ") + cudaGetErrorString(rc));
        }
    }

    // Copy `bytes` from the device buffer into `dst` (host).
    void download(void* dst, size_t bytes) const
    {
        cudaError_t rc = cudaMemcpy(
            dst, reinterpret_cast<const void*>(m_ptr), bytes, cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("GPUBuffer::download failed: ") + cudaGetErrorString(rc));
        }
    }

    // Zero the entire allocation.
    void clear()
    {
        if (m_ptr)
        {
            cudaMemset(reinterpret_cast<void*>(m_ptr), 0, m_size);
        }
    }

    // Raw device pointer — cast to CUdeviceptr for OptiX / CUDA driver APIs.
    CUdeviceptr ptr()   const { return m_ptr; }
    size_t      size()  const { return m_size; }
    bool        valid() const { return m_ptr != 0; }

    // Typed device pointer for CUDA runtime and struct assignments.
    template<typename T>
    T* typedPtr() const { return reinterpret_cast<T*>(m_ptr); }

private:
    CUdeviceptr m_ptr  = 0;
    size_t      m_size = 0;
};
