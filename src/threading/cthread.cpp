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

int count_pop = 0;
int count_push = 0;

// RAII owner for a task in flight
/*struct task_pool::run_task {
    task_pool& pool;
    lock& lck;
    task tsk;

    run_task(task_pool&, lock&);
    ~run_task();
};

// Own a task in flight
// lock should be passed locked,
// and will be unlocked after call
task_pool::run_task::run_task(task_pool& pool, lock& lck):
    pool{pool},
    lck{lck},
    tsk{}
{
    std::swap(tsk, pool.tasks_.front());
    pool.tasks_.pop_front();

    lck.unlock();
    pool.tasks_available_.notify_all();
}

// Release task
// Call unlocked, returns unlocked
task_pool::run_task::~run_task() {
    lck.lock();
    tsk.second->in_flight--;

    lck.unlock();
    pool.tasks_available_.notify_all();
}*/

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
    while (q_tasks_.empty() && !quit_ && ! finished()) {
        q_tasks_available_.wait(q_lock);
    }
    if(q_tasks_.empty()) {
        return false;
    }
    std::swap(tsk, q_tasks_.front());
    q_tasks_.pop_front();
    return true;
}

void notification_queue::remove_from_task_group(task &tsk) {
    {
        lock q_lock{q_mutex_};
        tsk.second->in_flight--;
    }
    q_tasks_available_.notify_all();
}

template<typename F>
void notification_queue::push(task&& tsk) {
    {
        lock q_lock{q_mutex_};
        q_tasks_.push_back(std::move(tsk));
        tsk.second->in_flight++;
    }
    q_tasks_available_.notify_all();
}

void notification_queue::push(const task& tsk) {
    {
        lock q_lock{q_mutex_};
        q_tasks_.push_back(tsk);
        tsk.second->in_flight++;
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
    //checking finished without a lock
    //should be okay if we don't add tasks to
    //a task_group while executing tasks in the task_group
    size_t i = get_current_thread();
    while (true) {
        task tsk;
        if (!q_[i].pop_if_not(tsk, finished))
            break;
        tsk.first();
        q_[i].remove_from_task_group(tsk);
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
    for (std::size_t i = 1; i < nthreads; i++) {
        threads_.emplace_back([&]{run_tasks_forever();});
        tid = threads_.back().get_id();
        thread_ids_[tid] = i;
    }
}

task_system::~task_system() {
    for (auto& e : q_) e.quit();
    for (auto& e : threads_) e.join();
}

void task_system::async_(task&& tsk) {
    auto i = index_++;
    q_[i % count_].push(tsk);
}

void task_system::wait(task_group* g) {
    run_tasks_while(g);
}

int task_system::get_num_threads() {
    return threads_.size() + 1;
}

std::size_t task_system::get_current_thread() {
    return thread_ids_[std::this_thread::get_id()];
}

task_system& task_system::get_global_task_system() {
    auto num_threads = threading::num_threads();
    static task_system global_task_system(num_threads);
    return global_task_system;
}
/*template<typename B>
void task_pool::run_tasks_loop(B finished) {
    lock lck{tasks_mutex_, std::defer_lock};
    while (true) {
        lck.lock();

        while (! quit_ && tasks_.empty() && ! finished()) {
            tasks_available_.wait(lck);
        }
        if (quit_ || finished()) {
            return;
        }

        run_task run{*this, lck};
        run.tsk.first();
    }
}

// runs forever until quit is true
void task_pool::run_tasks_forever() {
    run_tasks_loop([] {return false;});
}

// run until out of tasks for a group
void task_pool::run_tasks_while(task_group* g) {
    run_tasks_loop([=] {return ! g->in_flight;});
}

// Create pool and threads
// new threads are nthreads-1
task_pool::task_pool(std::size_t nthreads):
    tasks_mutex_{},
    tasks_available_{},
    tasks_{},
    threads_{}
{
    assert(nthreads > 0);

    // now for the main thread
    auto tid = std::this_thread::get_id();
    thread_ids_[tid] = 0;

    // and go from there
    for (std::size_t i = 1; i < nthreads; i++) {
        threads_.emplace_back([this]{run_tasks_forever();});
        tid = threads_.back().get_id();
        thread_ids_[tid] = i;
    }
}

task_pool::~task_pool() {
    {
        lock lck{tasks_mutex_};
        quit_ = true;
    }
    tasks_available_.notify_all();

    for (auto& thread: threads_) {
        thread.join();
    }
}

// push a task into pool
void task_pool::run(const task& tsk) {
    {
        lock lck{tasks_mutex_};
        tasks_.push_back(tsk);
        tsk.second->in_flight++;
    }
    tasks_available_.notify_all();
}

void task_pool::run(task&& tsk) {
  {
      lock lck{tasks_mutex_};
      tasks_.push_back(std::move(tsk));
      tsk.second->in_flight++;
  }
  tasks_available_.notify_all();
}

// call on main thread
// uses this thread to run tasks
// and waits until the entire task
// queue is cleared
void task_pool::wait(task_group* g) {
    run_tasks_while(g);
}

task_pool& task_pool::get_global_task_pool() {
    auto num_threads = threading::num_threads();
    static task_pool global_task_pool(num_threads);
    return global_task_pool;
}
*/