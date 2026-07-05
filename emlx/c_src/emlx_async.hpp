// Async NIF dispatch built on top of emlx::Worker.
//
// MLX 0.31.2 makes both Metal CommandEncoders (mlx/backend/metal/device.cpp:
// `static thread_local std::unordered_map<int, CommandEncoder> encoders;`)
// and the per-device default Stream (mlx/stream.cpp: `static thread_local
// auto default_streams = ...`) thread-local. Because `mlx::core::eval` walks
// the tape and calls `gpu::eval(arr)` *directly* on the calling thread (it
// is NOT trampolined to a `scheduler::StreamThread`; see
// mlx/transforms.cpp:eval_impl), every op for a given GPU stream — both
// graph construction AND eval — must happen on the OS thread that called
// `mlx::core::new_stream(d)` for that stream. Otherwise the eval thread's
// thread-local encoder map will not contain an entry for the stream's
// index, producing the runtime error
//   "There is no Stream(gpu, N) in current thread."
//
// Consequence for EMLX: every NIF that touches the MLX graph must run on
// the worker thread that owns the stream. We achieve this without
// rewriting each NIF body by:
//
//   1. Defining each "sync" NIF (e.g. `add`, `reshape`, `eval`, ...) as
//      a normal C++ function with the ERTS NIF signature.
//   2. Registering an `_async` wrapper in `nif_funcs[]` whose arity is
//      `original_arity + 1` (worker is `argv[0]`).
//   3. The wrapper extracts the worker, copies `argv[1..]` into a
//      process-independent `msg_env`, captures the caller pid + a fresh
//      ref, and posts a lambda to `worker->post(...)`.
//   4. The worker thread runs the original sync NIF body against
//      `msg_env` + the shifted argv, takes its `{:ok, _}` / `{:error, _}`
//      tagged tuple result, wraps it as `{ref, payload}`, and
//      `enif_send`s it back to the caller.
//
// The worker's `thread_main` calls `mlx::core::set_default_stream(stream)`
// before signalling ready, so any sync NIF body that resolves a
// `StreamOrDevice` from a `:cpu` / `:gpu` device atom (via `DEVICE_PARAM`)
// picks up the worker's stream automatically through MLX's
// `to_stream(s, default_) -> default_stream(default_)` lookup. No
// per-NIF code change is required.
//
// Lifetime invariants this helper relies on:
//
//   * `enif_self(env, &caller)` MUST be called on the BEAM scheduler
//     thread. We capture the resulting `ErlNifPid` by value into the
//     lambda; the worker thread (a non-scheduler OS thread) MUST NOT
//     call `enif_self` itself (the BEAM has no scheduler context for it).
//
//   * `enif_make_copy(msg_env, term)` for a resource ref bumps the
//     resource's ERTS refcount, so the resource (and the embedded MLX
//     array, function, or worker it backs) stays alive at least until
//     `msg_env` is freed at the end of the lambda. We do not need
//     additional `enif_keep_resource` bumps.
//
//   * `enif_send` with a non-NULL `msg_env` does not transfer ownership
//     of the env object itself. We always `enif_free_env(msg_env)` after
//     `enif_send` returns, regardless of success/failure. (Successful
//     `enif_send` invalidates the terms in `msg_env` but the env handle
//     itself remains owned by the caller.)
//
//   * If `worker->post` throws (worker is stopping), we must reclaim
//     `msg_env` and propagate the error to the BEAM caller synchronously.

#pragma once

#include "emlx_runtime_call_bridge.hpp"
#include "emlx_worker.hpp"
#include "erl_nif.h"
#include "nx_nif_utils.hpp"

#include <exception>
#include <vector>

namespace emlx {

// Build an `{:error, "<message>"}` tuple in `msg_env`. Uses
// `enif_make_string` to mirror nx::nif::error so the Elixir side can
// `List.to_string/1` it uniformly.
inline ERL_NIF_TERM make_error_term(ErlNifEnv *msg_env, const char *what) {
  return enif_make_tuple2(msg_env, enif_make_atom(msg_env, "error"),
                          enif_make_string(msg_env, what, ERL_NIF_LATIN1));
}

// Build an error tuple from the currently-thrown exception (must be
// called from inside a `catch` block).
inline ERL_NIF_TERM error_from_current_exception(ErlNifEnv *msg_env) {
  try {
    throw;
  } catch (const std::exception &e) {
    return make_error_term(msg_env, e.what());
  } catch (...) {
    return make_error_term(msg_env, "Unknown error");
  }
}

// Run `SyncOp(msg_env, op_argc, op_argv)` on `worker`'s thread and
// `enif_send` its tagged result back to the calling Elixir process.
// Returns the job ref synchronously for the caller to `receive` on.
//
// `argv[0]` MUST be the worker resource ref. `argv[1..argc-1]` are the
// op's actual arguments and are forwarded (after `enif_make_copy` into
// `msg_env`) to `SyncOp`.
//
// `SyncOp` is an existing sync-style NIF function that returns either
// `{:ok, value}` or `{:error, reason}`. The wrapper does not introspect
// the tuple — it is forwarded as-is as the second element of
// `{job_ref, payload}`.
template <ERL_NIF_TERM (*SyncOp)(ErlNifEnv *, int, const ERL_NIF_TERM *)>
ERL_NIF_TERM async_dispatch(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  if (argc < 1) {
    return enif_make_badarg(env);
  }

  emlx::Worker *worker;
  if (!enif_get_resource(env, argv[0], resource_object<emlx::Worker>::type,
                         (void **)&worker)) {
    return nx::nif::error(env, "Invalid command queue ref");
  }

  ErlNifPid caller_pid;
  enif_self(env, &caller_pid);

  ErlNifEnv *msg_env = enif_alloc_env();
  if (!msg_env) {
    return nx::nif::error(env, "Failed to allocate msg env");
  }

  ERL_NIF_TERM job_ref_msg = enif_make_ref(msg_env);
  ERL_NIF_TERM job_ref_caller = enif_make_copy(env, job_ref_msg);

  // Copy the op's arguments (everything past argv[0]) into msg_env.
  // For resource refs this also bumps the resource's ERTS refcount,
  // keeping the underlying MLX array / function / worker alive for the
  // duration of the lambda.
  // We need to do this because the worker is async and might outlive
  // the NIF env.
  int op_argc = argc - 1;
  std::vector<ERL_NIF_TERM> op_argv;
  op_argv.reserve(op_argc);
  for (int i = 0; i < op_argc; ++i) {
    op_argv.push_back(enif_make_copy(msg_env, argv[i + 1]));
  }

  try {
    worker->post([msg_env, job_ref_msg, caller_pid,
                  op_argv = std::move(op_argv)]() mutable {
      ERL_NIF_TERM payload;
      try {
        // Make this job's caller pid available to emlx::native's
        // EMLXRuntimeCall primitive for the duration of the call — see
        // emlx_runtime_call_bridge.hpp. Jobs *can* nest on a worker's one
        // dedicated OS thread: a blocked runtime_call pumps this same
        // worker's queue while waiting (Worker::pump_until,
        // emlx_worker.hpp), so a job posted by that call's own real
        // callback (e.g. EMLX.Quantization reentering EMLX on this same
        // worker) may run to completion *before* this outer job does.
        // Restore the *previous* pid on exit (not unconditionally
        // nullptr) so unwinding back out of a nested job doesn't clobber
        // the outer job's still-in-flight caller pid.
        struct CallerPidGuard {
          ErlNifPid pid;
          ErlNifPid *previous;
          explicit CallerPidGuard(ErlNifPid p) : pid(p) {
            previous = emlx::g_current_caller_pid;
            emlx::g_current_caller_pid = &pid;
          }
          ~CallerPidGuard() { emlx::g_current_caller_pid = previous; }
        } caller_pid_guard(caller_pid);

        payload = SyncOp(msg_env, static_cast<int>(op_argv.size()),
                         op_argv.data());
      } catch (...) {
        // The sync NIF should normally translate its own C++ exceptions
        // into `{:error, _}` via the `CATCH()` macro, but defensively
        // wrap anything that escapes so the caller's `receive` never
        // hangs.
        payload = error_from_current_exception(msg_env);
      }

      ERL_NIF_TERM reply =
          enif_make_tuple2(msg_env, job_ref_msg, payload);
      ErlNifPid pid = caller_pid;
      enif_send(NULL, &pid, msg_env, reply);
      enif_free_env(msg_env);
    });
  } catch (const std::exception &e) {
    // Worker is stopping or rejected the job; reclaim msg_env and
    // surface the error synchronously so the caller's wrapper can
    // raise without ever entering its `receive`.
    enif_free_env(msg_env);
    return nx::nif::error(env, e.what());
  } catch (...) {
    enif_free_env(msg_env);
    return nx::nif::error(env, "Unknown error posting to worker");
  }

  return nx::nif::ok(env, job_ref_caller);
}

} // namespace emlx
