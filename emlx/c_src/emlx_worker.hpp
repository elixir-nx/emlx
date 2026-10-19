#pragma once

// emlx::Worker — one OS thread + one mlx::core::Stream.
//
// Each Worker owns a dedicated OS thread that pulls Jobs from a
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
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <pthread.h>
#include <stdexcept>

namespace emlx {

class Worker;

// Points at the `Worker` whose dedicated OS thread is currently running,
// for the entire lifetime of that thread (set once in `thread_main`, not
// per-job — unlike `emlx::g_current_caller_pid`, which changes per job).
// Lets code running deep inside a job (e.g. `invoke_runtime_call` in
// emlx_runtime_call_bridge.hpp) find its own worker to pump its queue
// while blocked, without threading a `Worker*` through every call site.
inline thread_local Worker *g_current_worker = nullptr;

class Worker {
public:
  using Job = std::function<void()>;

  // Spawns the worker thread. The fresh mlx::core::Stream is allocated
  // *inside* the worker thread — MLX has thread-local command encoders
  // for GPU (0.31+) and CPU (0.32+, ml-explore/mlx#3537), so a stream
  // created on thread A cannot be synchronized or dispatched to from
  // thread B ("There is no Stream(gpu|cpu, N) in current thread"). We
  // block here until the worker thread has created the stream and
  // signalled ready, so callers can rely on stream() / device() being
  // valid the moment the constructor returns.
  //
  // Mixed-device graphs (e.g. GPU tensor + Nx scalar on the default CPU
  // backend) are handled in Elixir by promoting operands onto the op's
  // target device in `prepare_tensors!/1` before the NIF runs.
  explicit Worker(mlx::core::Device device)
      : device_(device), stream_(/*placeholder index*/ -1, device) {
    auto stream_future = stream_promise_.get_future();
    start_thread();
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
      join();
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
    join();
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

  // Called *from inside* a currently-executing job, on this worker's own
  // OS thread — e.g. `emlx::native::invoke_runtime_call`
  // (emlx_runtime_call_bridge.hpp), blocked inside `EMLXRuntimeCall::
  // eval_cpu`/`eval_gpu` waiting for the Elixir side's reply. Rather than
  // parking this thread on a bare wait (which would starve the rest of
  // this worker's queue — including any job the pending reply's own
  // real callback needs, e.g. `EMLX.Quantization`'s callbacks reentering
  // EMLX on this very worker/device to run the actual op), keep draining
  // and running this worker's own queue until `done()` returns true.
  // Safe to do because we're still on the one OS thread that owns this
  // worker's MLX stream thread-local state (see the class comment above
  // and emlx_async.hpp's header comment) — no cross-thread MLX dispatch
  // occurs, we're just running the *next* queued job a little earlier
  // than `thread_main`'s own loop would have. Jobs run this way nest:
  // if one of them itself blocks on another runtime_call, it recurses
  // into another `pump_until` a level deeper, which is why
  // `CallerPidGuard` (emlx_async.hpp) restores the *previous* caller pid
  // on exit rather than unconditionally clearing it.
  //
  // `done` may be polled many times and must be cheap and never block.
  void pump_until(const std::function<bool()> &done) {
    while (!done()) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        // Bounded wait: re-checks `done()` even if no job ever arrives
        // (e.g. the reply is delivered without going through this
        // worker's queue at all).
        cv_.wait_for(lock, std::chrono::milliseconds(5), [this] {
          return !queue_.empty() || stop_.load(std::memory_order_acquire);
        });
        if (queue_.empty()) {
          continue;
        }
        job = std::move(queue_.front());
        queue_.pop_front();
      }

      // Same contract as thread_main's loop below: a job must never let
      // an exception escape.
      try {
        job();
      } catch (...) {
      }
    }
  }

private:
  // MLX compiles a graph with recursion as deep as the graph (compile_dfs).
  // std::thread gets the default stack of a secondary thread, which is
  // 512 KB on macOS, and a compiled chain of 10,000 ops overflows it with
  // SIGBUS. 64 MB holds a chain of 200,000 ops. The OS maps stack pages
  // only when they are used, so a deep compile is the only cost.
  static constexpr size_t kStackSize = 64 * 1024 * 1024;

  void start_thread() {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, kStackSize);
    int rc = pthread_create(&thread_, &attr, &Worker::thread_entry, this);
    pthread_attr_destroy(&attr);
    if (rc != 0) {
      throw std::runtime_error("could not start the EMLX worker thread");
    }
    joinable_ = true;
  }

  static void *thread_entry(void *worker) {
    static_cast<Worker *>(worker)->thread_main();
    return nullptr;
  }

  void join() {
    if (joinable_) {
      pthread_join(thread_, nullptr);
      joinable_ = false;
    }
  }

  void thread_main() {
    // Allocate the stream on THIS thread (MLX thread-locality requirement —
    // see constructor comment). Pin all MLX ops issued from this thread to
    // our stream by making it the per-thread default. Only ops invoked
    // *inside* a posted job inherit this binding.
    try {
      mlx::core::Stream stream = mlx::core::new_stream(device_);
      mlx::core::set_default_stream(stream);
      stream_promise_.set_value(stream);
    } catch (...) {
      stream_promise_.set_exception(std::current_exception());
      return;
    }

    g_current_worker = this;

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
      // We never let an exception escape the thread function (which
      // would call std::terminate).
      try {
        job();
      } catch (...) {
        // Swallow. The job contract is: catch your own errors and
        // turn them into a {:error, _} reply. Anything that escapes
        // here would orphan the calling process's receive.
      }
    }

    g_current_worker = nullptr;

    // Release any per-thread MLX resources we hold (the StreamThread
    // for stream_ in the global scheduler will be torn down when the
    // last reference to its index drops).
    mlx::core::clear_streams();
  }

  mlx::core::Device device_;
  mlx::core::Stream stream_;
  std::promise<mlx::core::Stream> stream_promise_;
  pthread_t thread_;
  bool joinable_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Job> queue_;
  std::atomic<bool> stop_{false};
};

} // namespace emlx
