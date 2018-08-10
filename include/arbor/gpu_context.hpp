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

namespace threading {
    class task_system;
}
using task_system_handle = std::shared_ptr<threading::task_system>;

enum gpu_flags {
    no_sync = 0,
    has_atomic_double = 1
};

struct gpu_context {
    size_t attributes_ = 0;
#ifdef ARB_GPU_ENABLED
    unsigned num_streams_;
    cudaStream_t* streams_;
    std::unordered_map<std::thread::id, std::size_t> thread_to_stream_;
#endif

    gpu_context(task_system_handle& ts): attributes_(get_attributes()) {
        set_cuda_streams(ts);
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

    void set_cuda_streams(task_system_handle& ts) {
#ifdef ARB_GPU_ENABLED
        stream_id_ = 0;
        num_streams_ = ts->get_num_threads();
        thread_to_stream_ = ts->get_thread_ids();
        streams_ = new cudaStream_t[num_streams_];
        for (unsigned i = 0; i < num_streams_; i++) {
            cudaStreamCreate(&streams_[i]);
        }
#endif
    }
};

}
