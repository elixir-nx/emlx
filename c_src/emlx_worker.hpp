#pragma once

// emlx::Worker — one OS thread + one mlx::core::Stream.
//
// Each Worker owns a dedicated std::thread that pulls Jobs from a
// FIFO queue and runs them. The thread sets the worker's MLX Stream
// as its current-thread default on startup, so any MLX op invoked
// from inside a Job (mx::eval, mx::synchronize, etc.) dispatches to
// that stream.
//
// This is the dispatch primitive that backs both the application
// default worker (held in :persistent_term, see lib/emlx/application.ex)
// and per-context EMLX.CommandQueue instances created by
// Nx.Serving partitions or user code.
//
// The class is header-only because it is co-located with c_src/emlx_nif.cpp
// (single translation unit per the project's c_src layout convention).

#include "erl_nif.h"
#include "mlx/mlx.h"
#include "mlx/scheduler.h"
#include "mlx/stream.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace emlx {

class Worker {
public:
  using Job = std::function<void()>;

  // Spawns the worker thread. The fresh mlx::core::Stream is allocated
  // *inside* the worker thread — MLX 0.31.2 has thread-local state for
  // GPU streams (mlx-lm#1090, mlx-lm#1179: "There is no Stream(gpu, N)
  // in current thread"), so a stream created on thread A cannot be
  // synchronized or dispatched to from thread B. We block here until
  // the worker thread has created the stream and signalled ready, so
  // that callers can rely on stream() / device() being valid the
  // moment the constructor returns.
  explicit Worker(mlx::core::Device device)
      : device_(device), stream_(/*placeholder index*/ -1, device) {
    std::promise<mlx::core::Stream> stream_promise;
    auto stream_future = stream_promise.get_future();
    thread_ = std::thread(&Worker::thread_main, this, std::move(stream_promise));
    // Blocks (or rethrows) until the worker thread has registered its
    // stream and signalled ready.
    //
    // If stream_future.get() throws (worker thread set an exception),
    // the thread has already exited cleanly. We must join before
    // re-throwing, because ~thread() on a joinable thread calls
    // std::terminate() — which is the "terminate called without an
    // active exception" crash seen on Linux with no GPU.
    try {
      stream_ = stream_future.get();
    } catch (...) {
      if (thread_.joinable()) {
        thread_.join();
      }
      throw;
    }
  }

  // Sets the stop flag, drains any in-flight jobs already past the
  // pop, then joins the OS thread. Pending jobs in the queue at stop
  // time are still executed (so callers awaiting an enif_send reply
  // receive it). Jobs posted *after* the destructor begins are
  // rejected by post().
  ~Worker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  Worker(const Worker &) = delete;
  Worker &operator=(const Worker &) = delete;
  Worker(Worker &&) = delete;
  Worker &operator=(Worker &&) = delete;

  // Pushes a job onto the queue. Throws std::runtime_error if the
  // worker is already stopping (the caller should translate this to
  // an `{:error, "worker stopped"}` NIF return).
  void post(Job job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_.load(std::memory_order_acquire)) {
        throw std::runtime_error("Worker is stopping; cannot post job");
      }
      queue_.push_back(std::move(job));
    }
    cv_.notify_one();
  }

  mlx::core::Device device() const { return device_; }
  mlx::core::Stream stream() const { return stream_; }

private:
  void thread_main(std::promise<mlx::core::Stream> stream_promise) {
    // Allocate the stream on THIS thread (MLX 0.31.2 thread-locality
    // requirement — see constructor comment). Pin all MLX ops issued
    // from this thread to our stream by making it the per-thread
    // default. Graph-construction NIFs continue to run on BEAM
    // scheduler threads (and use those threads' defaults); only ops
    // invoked *inside* a posted job (currently mx::eval and
    // mx::synchronize) inherit this binding.
    try {
      mlx::core::Stream stream = mlx::core::new_stream(device_);
      mlx::core::set_default_stream(stream);
      stream_promise.set_value(stream);
    } catch (...) {
      stream_promise.set_exception(std::current_exception());
      return;
    }

    while (true) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
          return !queue_.empty() || stop_.load(std::memory_order_acquire);
        });
        if (queue_.empty() && stop_.load(std::memory_order_acquire)) {
          break;
        }
        job = std::move(queue_.front());
        queue_.pop_front();
      }

      // Jobs are responsible for their own exception handling and for
      // delivering a reply to the calling BEAM process via enif_send.
      // We never let an exception escape into the std::thread wrapper
      // (which would call std::terminate).
      try {
        job();
      } catch (...) {
        // Swallow. The job contract is: catch your own errors and
        // turn them into a {:error, _} reply. Anything that escapes
        // here would orphan the calling process's receive.
      }
    }

    // Release any per-thread MLX resources we hold (the StreamThread
    // for stream_ in the global scheduler will be torn down when the
    // last reference to its index drops).
    mlx::core::clear_streams();
  }

  mlx::core::Device device_;
  mlx::core::Stream stream_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Job> queue_;
  std::atomic<bool> stop_{false};
};

} // namespace emlx
