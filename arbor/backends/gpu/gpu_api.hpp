#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using DeviceProp = cudaDeviceProp;

constexpr cudaError_t Success = cudaSuccess;
constexpr cudaError_t ErrorInvalidDevice = cudaErrorInvalidDevice;

template <typename... ARGS>
inline auto get_device_properities(ARGS&&... args) -> cudaError_t {
  return cudaGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_device(ARGS&&... args) -> cudaError_t {
  return cudaSetDevice(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto get_error_name(ARGS&&... args) -> cudaError_t {
  return cudaGetErrorName(std::forward<ARGS>(args)...);
}

