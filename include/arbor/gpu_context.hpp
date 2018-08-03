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
    std::vector<cudaStream_t> streams_;
#endif

    gpu_context(): has_gpu_(false), attributes_(get_attributes()) {
        set_cuda_streams();
    };
    gpu_context(bool has_gpu): has_gpu_(has_gpu), attributes_(get_attributes()) {
        set_cuda_streams();
    };

private:
    size_t get_attributes() {
        size_t attributes = 0;
#ifdef ARB_GPU_ENABLED
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if(prop.concurrentManagedAccess)
        //if(prop.major*100 + prop.minor >= 600)
            attributes|= (1<<gpu_flags::no_sync);
        if(prop.major*100 + prop.minor >= 600)
            attributes|= (1<<gpu_flags::has_atomic_double);
#endif
        return attributes;
    };

    void set_cuda_streams(unsigned nstreams=8) {
#ifdef ARB_GPU_ENABLED
        streams_.resize(nstreams);
        for (unsigned i = 0; i < nstreams; i++) {
            cudaStreamCreate(&streams_[i]);
        }
#endif
    }
};

}
