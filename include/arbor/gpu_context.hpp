#pragma once

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
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

std::unordered_map<std::thread::id, std::size_t> get_map(task_system_handle& ts);

#ifndef ARB_GPU_ENABLED
struct gpu_context {
    gpu_context(task_system_handle& ts) {};
};
#else

enum gpu_flags {
no_sync = 0,
has_atomic_double = 1
};

struct gpu_context {
    size_t attributes_ = 0;
    cudaStream_t* streams_;
    std::unordered_map<std::thread::id, std::size_t> thread_to_stream_;

    gpu_context(task_system_handle& ts): attributes_(get_attributes()) {
        get_cuda_streams(ts);
    };

    ~gpu_context() {
        delete[] streams_;
    }

    cudaStream_t* get_thread_stream(std::thread::id thread_id) {
        std::size_t stream_id = thread_to_stream_[thread_id];
        return &streams_[stream_id];
    }

private:

    size_t get_attributes() {
        size_t attributes = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if(prop.concurrentManagedAccess)
            attributes|= (1<<gpu_flags::no_sync);
        if(prop.major*100 + prop.minor >= 600)
            attributes|= (1<<gpu_flags::has_atomic_double);
        return attributes;
    };

    void get_cuda_streams(task_system_handle& ts) {
        thread_to_stream_ = get_map(ts);
        int num_streams = thread_to_stream_.size();
        streams_ = new cudaStream_t[num_streams];
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams_[i]);
        }
    }
};

#endif
};
