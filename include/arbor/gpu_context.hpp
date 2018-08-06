#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <arbor/version.hpp>

#ifdef ARB_GPU_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace arb {

enum gpu_flags {
    no_sync = 0,
    has_atomic_double = 1
};

struct gpu_context {
    bool has_gpu_;
    size_t attributes_ = 0;
#ifdef ARB_GPU_ENABLED
    unsigned num_streams_;
    cudaStream_t* streams_;
#endif

    gpu_context(): has_gpu_(false), attributes_(get_attributes()) {
        set_cuda_streams();
    };
    gpu_context(bool has_gpu): has_gpu_(has_gpu), attributes_(get_attributes()) {
        set_cuda_streams();
    };

#ifdef ARB_GPU_ENABLED
    ~gpu_context() {
        delete[] streams_;
    }
    unsigned get_stream_id() {
        return stream_id_++ % num_streams_;
    }
#endif


private:
    int stream_id_;
    size_t get_attributes() {
        size_t attributes = 0;
#ifdef ARB_GPU_ENABLED
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if(prop.concurrentManagedAccess)
            attributes|= (1<<gpu_flags::no_sync);
        if(prop.major*100 + prop.minor >= 600)
            attributes|= (1<<gpu_flags::has_atomic_double);
#endif
        return attributes;
    };

    void set_cuda_streams(unsigned size = 4) {
#ifdef ARB_GPU_ENABLED
        stream_id_ = 0;
        num_streams_ = size;
        streams_ = new cudaStream_t[num_streams_];
        for (unsigned i = 0; i < num_streams_; i++) {
            cudaStreamCreate(&streams_[i]);
        }
#endif
    }
};

}
