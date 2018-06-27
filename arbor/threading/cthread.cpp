#include <atomic>
#include <cassert>
#include <cstring>
#include <exception>
#include <iostream>
#include <regex>

#include "cthread.hpp"
#include "threading.hpp"

using namespace arb::threading::impl;
using namespace arb;

bool notification_queue::try_pop(task& tsk) {
    lock q_lock{q_mutex_, std::try_to_lock};
    if (!q_lock || q_tasks_.empty()) return false;
    std::swap(tsk, q_tasks_.front());
    q_tasks_.pop_front();
    return true;
}

bool notification_queue::pop(task& tsk) {
    lock q_lock{q_mutex_};
    while (q_tasks_.empty() && !quit_) {
        q_tasks_available_.wait(q_lock);
    }
    if(q_tasks_.empty()) {
        return false;
    }
    std::swap(tsk, q_tasks_.front());
    q_tasks_.pop_front();
    return true;
}

template<typename B>
bool notification_queue::pop_if_not(task& tsk, B finished) {
    lock q_lock{q_mutex_};
    while (q_tasks_.empty() && !quit_ && ! finished) {
        q_tasks_available_.wait(q_lock);
    }
    if(q_tasks_.empty()) {
        return false;
    }
    std::swap(tsk, q_tasks_.front());
    q_tasks_.pop_front();
    return true;
}

bool notification_queue::try_push(const task& tsk) {
    {
        lock q_lock{q_mutex_, std::try_to_lock};
        if(!q_lock) return false;
        q_tasks_.push_back(std::move(tsk));
        tsk.second->inc_in_flight();
    }
    q_tasks_available_.notify_all();
    return true;
}

void notification_queue::push(task&& tsk) {
    {
        lock q_lock{q_mutex_};
        q_tasks_.push_back(std::move(tsk));
        tsk.second->inc_in_flight();
    }
    q_tasks_available_.notify_all();
}

void notification_queue::push(const task& tsk) {
    {
        lock q_lock{q_mutex_};
        q_tasks_.push_back(tsk);
        tsk.second->inc_in_flight();
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

template<typename B>
void task_system::run_tasks_loop(B finished ){
    size_t i = get_current_thread();
    while (true) {
        task tsk;
        for(unsigned n = 0; n != count_; n++) {
            if(q_[(i + n) % count_].try_pop(tsk)) break;
        }
        if(!tsk.first && !q_[i].pop(tsk)) break;
        tsk.first();
        tsk.second->dec_in_flight();
    }
}

void task_system::run_tasks_while(task_group* g) {
    run_tasks_loop([=] {return ! g->get_in_flight();});
}

void task_system::run_tasks_forever() {
    run_tasks_loop([] {return false;});
}

task_system::task_system(int nthreads) : count_(nthreads), q_(nthreads) {
    assert( nthreads > 0);

    // now for the main thread
    auto tid = std::this_thread::get_id();
    thread_ids_[tid] = 0;

    // and go from there
    lock thread_ids_lock{thread_ids_mutex_};
    for (std::size_t i = 1; i < count_; i++) {
        threads_.emplace_back([&]{run_tasks_forever();});
        auto tid = threads_.back().get_id();
        thread_ids_[tid] = i;
    }
}

task_system::~task_system() {
    for (auto& e : q_) e.quit();
    for (auto& e : threads_) e.join();
}

void task_system::async_(task&& tsk) {
    auto i = index_++;

    for(unsigned n = 0; n != count_; n++) {
        if(q_[(i + n) % count_].try_push(tsk)) return;
    }
    q_[i % count_].push(tsk);
}

void task_system::wait(task_group* g) {
    run_tasks_while(g);
}

int task_system::get_num_threads() {
    return threads_.size() + 1;
}

std::size_t task_system::get_current_thread() {
    lock thread_ids_lock{thread_ids_mutex_};
    std::thread::id tid = std::this_thread::get_id();
    return thread_ids_[tid];
}

task_system& task_system::get_global_task_system() {
    auto num_threads = threading::num_threads();
    static task_system global_task_system(num_threads);
    return global_task_system;
}

void task_group::wait() {
    while(get_in_flight()) {
        size_t i = global_task_system.get_current_thread();
        task tsk;
        for(unsigned n = 0; n != global_task_system.get_num_threads(); n++) {
            if(global_task_system.q_[(i + n) % global_task_system.get_num_threads()].try_pop(tsk)) {
                tsk.first();
                tsk.second->dec_in_flight();
                break;
            }
        }
    }
}