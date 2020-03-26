#include <cstdlib>
#include <string>

#include <arbor/arbexcept.hpp>

#include "util.hpp"

#ifdef ARB_HAVE_GPU

#include <backends/gpu/gpu_api.hpp>

#define HANDLE_GPU_ERROR(error, msg)\
throw arbor_exception("CUDA memory:: "+std::string(__func__)+" "+std::string((msg))+": "+gpu_error_string(error));

namespace arb {
namespace memory {

using std::to_string;

void cuda_memcpy_d2d(void* dest, const void* src, std::size_t n) {
    if (auto error = gpu_memcpy(dest, src, n, gpuMemcpyDeviceToDevice)) {
        HANDLE_GPU_ERROR(error, "n="+to_string(n));
    }
}

void cuda_memcpy_d2h(void* dest, const void* src, std::size_t n) {
    if (auto error = gpu_memcpy(dest, src, n, gpuMemcpyDeviceToHost)) {
        HANDLE_GPU_ERROR(error, "n="+to_string(n));
    }
}

void cuda_memcpy_h2d(void* dest, const void* src, std::size_t n) {
    if (auto error = gpu_memcpy(dest, src, n, gpuMemcpyHostToDevice)) {
        HANDLE_GPU_ERROR(error, "n="+to_string(n));
    }
}

void* cuda_host_register(void* ptr, std::size_t size) {
    if (auto error = gpu_host_register(ptr, size, gpuHostRegisterPortable)) {
        HANDLE_GPU_ERROR(error, "unable to register host memory");
    }
    return ptr;
}

void cuda_host_unregister(void* ptr) {
    gpu_host_unregister(ptr);
}

void* cuda_malloc(std::size_t n) {
    void* ptr;

    if (auto error = gpu_malloc(&ptr, n)) {
        HANDLE_GPU_ERROR(error, "unable to allocate "+to_string(n)+" bytes");
    }
    return ptr;
}

void cuda_free(void* ptr) {
    if (auto error = gpu_free(ptr)) {
        HANDLE_GPU_ERROR(error, "");
    }
}

} // namespace memory
} // namespace arb

#else

#define NOCUDA \
LOG_ERROR("memory:: "+std::string(__func__)+"(): no CUDA support")

namespace arb {
namespace memory {

void cuda_memcpy_d2d(void* dest, const void* src, std::size_t n) {
    NOCUDA;
}

void cuda_memcpy_d2h(void* dest, const void* src, std::size_t n) {
    NOCUDA;
}

void cuda_memcpy_h2d(void* dest, const void* src, std::size_t n) {
    NOCUDA;
}

void* cuda_host_register(void* ptr, std::size_t size) {
    NOCUDA;
    return 0;
}

void cuda_host_unregister(void* ptr) {
    NOCUDA;
}

void* cuda_malloc(std::size_t n) {
    NOCUDA;
    return 0;
}

void cuda_free(void* ptr) {
    NOCUDA;
}

} // namespace memory
} // namespace arb

#endif // def ARB_HAVE_GPU

