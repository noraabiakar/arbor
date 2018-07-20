#pragma once

#include <iostream>
#include <type_traits>

#include <thread>
#include <mutex>
#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <vector>
#include <type_traits>
#include <functional>
#include <condition_variable>
#include <utility>
#include <unordered_map>
#include <deque>
#include <atomic>
#include <type_traits>

#include <cstdlib>
#include "arbor/execution_context.hpp"

namespace arb {
inline threading::task_system* get_task_system(const task_system_handle* h) {
    return (*h).get();
}
namespace threading {

// Forward declare task_group at bottom of this header
class task_group;

using std::mutex;
using lock = std::unique_lock<mutex>;
using std::condition_variable;
using task = std::function<void()>;

struct task_box {
    std::vector<task> tb;
};

namespace impl {
class notification_queue {
private:
    // FIFO of pending tasks.
    std::deque<task> q_tasks_;

    // Lock and signal on task availability change this is the crucial bit.
    mutex q_mutex_;
    condition_variable q_tasks_available_;

    // Flag to handle exit from all threads.
    bool quit_ = false;

public:
    // Pops a task from the task queue returns false when queue is empty.
    task try_pop();
    task pop();

    // Pushes a task into the task queue and increases task group counter.
    void push(task&& tsk); // TODO: need to use value?
    void push4(task&& tsk0, task&& tsk1, task&& tsk2, task&& tsk3);
    void push_x(std::vector<task>&& tsk);
    bool try_push(task& tsk);
    bool try_push4(task& tsk0, task& tsk1, task& tsk2, task& tsk3);
    bool try_push_x(std::vector<task>& tsk);

    // Finish popping all waiting tasks on queue then stop trying to pop new tasks
    void quit();
};
}// namespace impl

class task_system {
private:
    unsigned count_;

    std::vector<std::thread> threads_;

    // queue of tasks
    std::vector<impl::notification_queue> q_;

    // threads -> index
    std::unordered_map<std::thread::id, std::size_t> thread_ids_;

    // total number of tasks pushed in all queues
    std::atomic<unsigned> index_{0};

public:
    // Create nthreads-1 new c std threads
    task_system(int nthreads);

    // task_system is a singleton.
    task_system(const task_system&) = delete;
    task_system& operator=(const task_system&) = delete;

    ~task_system();

    // Pushes tasks into notification queue.
    void async(task tsk);
    void async4(task tsk0, task tsk1, task tsk2, task tsk3);
    void async_x(std::vector<task> v_tsk);

    // Runs tasks until quit is true.
    void run_tasks_loop(int i);

    // Request that the task_system attempts to find and run a _single_ task.
    // Will return without executing a task if no tasks available.
    void try_run_task();

    // Includes master thread.
    int get_num_threads();

    // Get a stable integer for the current thread that is [0, nthreads).
    std::size_t get_current_thread();
};

///////////////////////////////////////////////////////////////////////
// types
///////////////////////////////////////////////////////////////////////

template <typename T>
class enumerable_thread_specific {
    task_system* global_task_system = nullptr;

    using storage_class = std::vector<T>;
    storage_class data;

public:
    using iterator = typename storage_class::iterator;
    using const_iterator = typename storage_class::const_iterator;

    enumerable_thread_specific(const task_system_handle* ts):
        global_task_system{get_task_system(ts)},
        data{std::vector<T>(global_task_system->get_num_threads())}
    {}

    enumerable_thread_specific(const T& init, const task_system_handle* ts):
        global_task_system{get_task_system(ts)},
        data{std::vector<T>(global_task_system->get_num_threads(), init)}
    {}

    T& local() {
        return data[global_task_system->get_current_thread()];
    }
    const T& local() const {
        return data[global_task_system->get_current_thread()];
    }

    auto size() const { return data.size(); }

    iterator begin() { return data.begin(); }
    iterator end()   { return data.end(); }

    const_iterator begin() const { return data.begin(); }
    const_iterator end()   const { return data.end(); }

    const_iterator cbegin() const { return data.cbegin(); }
    const_iterator cend()   const { return data.cend(); }
};

inline std::string description() {
    return "CThread Pool";
}

constexpr bool multithreaded() { return true; }

class task_group {
private:
    std::atomic<std::size_t> in_flight_{0};
    task_system* task_system_;

public:
    task_group(task_system* ts):
        task_system_{ts}
    {}

    task_group(const task_group&) = delete;
    task_group& operator=(const task_group&) = delete;

    template <typename F>
    class wrap {
        F f;
        std::atomic<std::size_t>& counter;

    public:

        // Construct from a compatible function and atomic counter
        template <typename F2>
        explicit wrap(F2&& other, std::atomic<std::size_t>& c):
                f(std::forward<F2>(other)),
                counter(c)
        {}

        wrap(wrap&& other):
                f(std::move(other.f)),
                counter(other.counter)
        {}

        // std::function is not guaranteed to not copy the contents on move construction
        // But the class is safe because we don't call operator() more than once on the same wrapped task
        wrap(const wrap& other):
                f(other.f),
                counter(other.counter)
        {}

        void operator()() {
            f();
            --counter;
        }
    };

    template <typename F>
    using callable = typename std::decay<F>::type;

    template <typename F>
    wrap<callable<F>> make_wrapped_function(F&& f, std::atomic<std::size_t>& c) {
        return wrap<callable<F>>(std::forward<F>(f), c);
    }

    template<typename F>
    void run(F&& f) {
        ++in_flight_;
        task_system_->async(make_wrapped_function(std::forward<F>(f), in_flight_));
    }

    template<typename F>
    void run4(F&& f0, F&& f1, F&& f2, F&& f3) {
        in_flight_+=4;
        wrap<callable<F>> w0 = make_wrapped_function(std::forward<F>(f0), in_flight_);
        wrap<callable<F>> w1 = make_wrapped_function(std::forward<F>(f1), in_flight_);
        wrap<callable<F>> w2 = make_wrapped_function(std::forward<F>(f2), in_flight_);
        wrap<callable<F>> w3 = make_wrapped_function(std::forward<F>(f3), in_flight_);
        task_system_->async4(w0, w1, w2, w3);
    }

    template<typename F>
    void run_x(std::vector<F> v) {
        in_flight_+= v.size();
        std::vector<task> w;
        for (unsigned i = 0; i < v.size(); i++) {
            w.push_back(make_wrapped_function(std::forward<F>(v[i]), in_flight_));
        }
        task_system_->async_x(w);
    }

    // wait till all tasks in this group are done
    void wait() {
        while (in_flight_) {
            task_system_->try_run_task();
        }
    }

    // Make sure that all tasks are done before clean up
    ~task_group() {
        wait();
    }
};

///////////////////////////////////////////////////////////////////////
// algorithms
///////////////////////////////////////////////////////////////////////
struct parallel_for {
    template <typename F>
    static void apply(int left, int right, task_system* ts, F f) {
        int size = 4;
        task_group g(ts);
        for (int i = left; i < right; i+=size) {
            if (i + size > right) {
                for (int j = i; j < right; j++) {
                    g.run([=] { f(j); });
                }
            }
            else {
                std::vector<task> v;
                for (int j = i; j < i + size ; j++)
                    v.push_back([=] {f(j);});
                g.run_x<task>(v);
            }
        }
        g.wait();
    }
};
} // namespace threading
} // namespace arb
