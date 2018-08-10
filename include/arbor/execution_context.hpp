#pragma once

#include <memory>
#include <string>

#include <arbor/distributed_context.hpp>
#include <arbor/gpu_context.hpp>
#include <arbor/util/pp_util.hpp>


namespace arb {
namespace threading {
    class task_system;
}
using task_system_handle = std::shared_ptr<threading::task_system>;
using distributed_context_handle = std::shared_ptr<distributed_context>;
using gpu_context_handle = std::shared_ptr<gpu_context>;

task_system_handle make_thread_pool();
task_system_handle make_thread_pool(int nthreads);


struct execution_context {
    distributed_context_handle distributed;
    task_system_handle thread_pool;
    gpu_context_handle gpu;

    execution_context(): distributed(std::make_shared<distributed_context>()),
                         thread_pool(arb::make_thread_pool()),
                         gpu(std::make_shared<gpu_context>(thread_pool)) {};
};

}
