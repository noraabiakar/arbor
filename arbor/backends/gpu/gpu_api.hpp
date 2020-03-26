#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using DeviceProp = cudaDeviceProp;

constexpr auto Success = cudaSuccess;
constexpr auto ErrorInvalidDevice = cudaErrorInvalidDevice;
constexpr auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr auto gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
constexpr auto gpuHostRegisterPortable = cudaHostRegisterPortable;

template <typename... ARGS>
inline auto get_device_properities(ARGS&&... args) -> cudaError_t {
  return cudaGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto gpu_error_string(ARGS&&... args) -> const char* {
    return cudaGetErrorString(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_device(ARGS&&... args) -> cudaError_t {
  return cudaSetDevice(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto gpu_memcpy(ARGS&&... args) -> cudaError_t {
    return cudaMemcpy(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto gpu_host_register(ARGS&&... args) -> cudaError_t {
    return cudaHostRegister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto gpu_host_unregister(ARGS&&... args) -> cudaError_t {
    return cudaHostUnregister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto gpu_malloc(ARGS&&... args) -> cudaError_t {
    return cudaMalloc(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto gpu_free(ARGS&&... args) -> cudaError_t {
    return cudaFree(std::forward<ARGS>(args)...);
}

