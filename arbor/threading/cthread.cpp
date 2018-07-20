#include <atomic>
#include <cassert>
#include <cstring>
#include <exception>
#include <iostream>
#include <regex>

#include "cthread.hpp"
#include "threading.hpp"
#include "arbor/execution_context.hpp"

using namespace arb::threading::impl;
using namespace arb::threading;
using namespace arb;

task notification_queue::try_pop() {
    task tsk;
    lock q_lock{q_mutex_, std::try_to_lock};
    if (q_lock && !q_tasks_.empty()) {
        tsk = std::move(q_tasks_.front());
        q_tasks_.pop_front();
    }
    return tsk;
}

task notification_queue::pop() {
    task tsk;
    lock q_lock{q_mutex_};
    while (q_tasks_.empty() && !quit_) {
        q_tasks_available_.wait(q_lock);
    }
    if (!q_tasks_.empty()) {
        tsk = std::move(q_tasks_.front());
        q_tasks_.pop_front();
    }
    return tsk;
}

bool notification_queue::try_push(task& tsk) {
    {
        lock q_lock{q_mutex_, std::try_to_lock};
        if (!q_lock) return false;
        q_tasks_.push_back(std::move(tsk));
        tsk = 0;
    }
    q_tasks_available_.notify_all();
    return true;
}

bool notification_queue::try_push4(task& tsk0, task& tsk1, task& tsk2, task& tsk3) {
    {
        lock q_lock{q_mutex_, std::try_to_lock};
        if (!q_lock) return false;
        q_tasks_.push_back(std::move(tsk0));
        q_tasks_.push_back(std::move(tsk1));
        q_tasks_.push_back(std::move(tsk2));
        q_tasks_.push_back(std::move(tsk3));
        tsk0 = 0; tsk1 = 0; tsk2 = 0; tsk3 = 0;
    }
    q_tasks_available_.notify_all();
    return true;
}

bool notification_queue::try_push_x(std::vector<task>& v_tsk) {
    {
        lock q_lock{q_mutex_, std::try_to_lock};
        if (!q_lock) return false;
        for (unsigned i = 0; i < v_tsk.size(); i++) {
            q_tasks_.push_back(std::move(v_tsk[i]));
            v_tsk[i] = 0;
        }
    }
    q_tasks_available_.notify_all();
    return true;
}

void notification_queue::push(task&& tsk) {
    {
        lock q_lock{q_mutex_};
        q_tasks_.push_back(std::move(tsk));
    }
    q_tasks_available_.notify_all();
}

void notification_queue::push4(task&& tsk0, task&& tsk1, task&& tsk2, task&& tsk3) {
    {
        lock q_lock{q_mutex_};
        q_tasks_.push_back(std::move(tsk0));
        q_tasks_.push_back(std::move(tsk1));
        q_tasks_.push_back(std::move(tsk2));
        q_tasks_.push_back(std::move(tsk3));
    }
    q_tasks_available_.notify_all();
}

void notification_queue::push_x(std::vector<task>&& v_tsk) {
    {
        lock q_lock{q_mutex_};
        for (unsigned i = 0; i < v_tsk.size(); i++)
            q_tasks_.push_back(std::move(v_tsk[i]));
    }
    q_tasks_available_.notify_all();
}

void notification_queue::quit() {
    {
        lock q_lock{q_mutex_};
        quit_ = true;
    }
    q_tasks_available_.notify_all();
}

void task_system::run_tasks_loop(int i){
    while (true) {
        task tsk;
        for (unsigned n = 0; n != count_; n++) {
            tsk = q_[(i + n) % count_].try_pop();
            if (tsk) break;
        }
        if (!tsk) tsk = q_[i].pop();
        if (!tsk) break;
        tsk();
    }
}

void task_system::try_run_task() {
    auto nthreads = get_num_threads();
    task tsk;
    for (int n = 0; n != nthreads; n++) {
        tsk = q_[n % nthreads].try_pop();
        if (tsk) {
            tsk();
            break;
        }
    }
}

task_system::task_system(int nthreads) : count_(nthreads), q_(nthreads) {
    assert( nthreads > 0);

    // now for the main thread
    auto tid = std::this_thread::get_id();
    thread_ids_[tid] = 0;

    for (unsigned i = 1; i < count_; i++) {
        threads_.emplace_back([this, i]{run_tasks_loop(i);});
        tid = threads_.back().get_id();
        thread_ids_[tid] = i;
    }
}

task_system::~task_system() {
    for (auto& e: q_) e.quit();
    for (auto& e: threads_) e.join();
}

void task_system::async(task tsk) {
    auto i = index_++;

    for (unsigned n = 0; n != count_; n++) {
        if (q_[(i + n) % count_].try_push(tsk)) return;
    }
    q_[i % count_].push(std::move(tsk));
}

void task_system::async4(task tsk0, task tsk1, task tsk2, task tsk3) {
    auto i = index_++;

    for (unsigned n = 0; n != count_; n++) {
        if (q_[(i + n) % count_].try_push4(tsk0, tsk1, tsk2, tsk3)) return;
    }
    q_[i % count_].push4(std::move(tsk0), std::move(tsk1), std::move(tsk2), std::move(tsk3));
}

void task_system::async_x(std::vector<task> v_tsk) {
    auto i = index_++;

    for (unsigned n = 0; n != count_; n++) {
        if (q_[(i + n) % count_].try_push_x(v_tsk)) return;
    }
    q_[i % count_].push_x(std::move(v_tsk));
}

int task_system::get_num_threads() {
    return threads_.size() + 1;
}

std::size_t task_system::get_current_thread() {
    std::thread::id tid = std::this_thread::get_id();
    return thread_ids_[tid];
}

task_system_handle arb::make_ts(int nthreads) {
    return task_system_handle(new task_system(nthreads));
}


