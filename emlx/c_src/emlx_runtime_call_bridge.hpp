#pragma once

// emlx_runtime_call_bridge.hpp — blocking bridge between the worker OS
// thread executing a compiled program's `EMLXRuntimeCall` primitive
// (emlx_compiler.cpp) and the BEAM process that actually runs the real
// Elixir callback for `Nx.runtime_call/4` (see EMLX.await_worker/2's
// `:emlx_runtime_call` receive clause).
//
// Architecture mirrors EXLA's `nx/exla/c_src/exla/custom_calls/
// runtime_callback_bridge.{h,cc}`: a `PendingRuntimeCall` resource carries a
// mutex/condvar pair plus pre-allocated output destinations. After
// `enif_send`ing a request, the worker thread does NOT simply park on the
// condvar — it calls `Worker::pump_until` (emlx_worker.hpp) to keep
// draining and running its own job queue while waiting. This matters
// because the real Elixir callback that will eventually call
// `resolve_runtime_call/3` may itself need to run further work on this
// exact worker (EMLX.Quantization's callbacks, for one, reenter EMLX on
// the same device/worker they were called from) — parking this, the
// worker's one dedicated OS thread, on a bare wait would deadlock against
// that reentrant job sitting behind it in the very queue it can no longer
// service. See emlx_worker.hpp's `pump_until` doc for the full argument.
//
// `g_current_caller_pid` is set by `emlx::async_dispatch<SyncOp>`
// (emlx_async.hpp) for the duration of one worker-thread job — including
// any `mlx::core::eval(...)` call the job body makes — so it is in scope
// whenever `EMLXRuntimeCall::eval_cpu`/`eval_gpu` actually fires. One
// `EMLX.CommandQueue` worker owns exactly one dedicated OS thread (see
// emlx_async.hpp's header comment), so a plain `thread_local` is enough to
// disambiguate multiple workers evaluating concurrently.

#include "emlx_worker.hpp"
#include "erl_nif.h"
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "nx_nif_utils.hpp"

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace emlx {

// Set by `async_dispatch<SyncOp>` around the call to `SyncOp` (e.g.
// `eval_program`) — see emlx_async.hpp.
inline thread_local ErlNifPid *g_current_caller_pid = nullptr;

namespace native {

// Per-call state shared between the worker thread blocked inside
// `EMLXRuntimeCall::eval()` and the BEAM process that delivers the reply via
// `resolve_runtime_call/3`. Exposed as an ERTS resource so it can ride as an
// opaque handle inside the `{:emlx_runtime_call, pending, callback_index,
// args}` message instead of an integer id.
struct PendingRuntimeCall {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  bool ok = false;
  std::string error;
  // (destination pointer, byte size) per output, in order — already
  // allocated by `invoke_runtime_call` before the message is sent, per
  // `mlx::core::Primitive`'s "the evaluation function is responsible for
  // allocating space for the array" contract.
  std::vector<std::pair<uint8_t *, size_t>> outputs;
};

// Blocks the calling (worker) thread until the Elixir side invokes
// `EMLX.NIF.resolve_runtime_call/3` for this call. Called from
// `EMLXRuntimeCall::eval()` (emlx_compiler.cpp). Throws `std::runtime_error`
// on any failure (no caller pid in scope, send failure, or a callback that
// errored) — the caller (`eval_cpu`/`eval_gpu`, reached through
// `eval_program`'s `CATCH()` macro) turns that into `{:error, _}`.
inline void invoke_runtime_call(int64_t callback_index,
                                const std::vector<mlx::core::array> &inputs,
                                std::vector<mlx::core::array> &outputs) {
  if (g_current_caller_pid == nullptr) {
    throw std::runtime_error(
        "emlx::native: runtime_call primitive fired with no caller pid in "
        "scope (a compiled program containing a runtime_call must be "
        "force-evaluated inside eval_program's async job body)");
  }
  ErlNifPid caller_pid = *g_current_caller_pid;

  auto *pending = static_cast<PendingRuntimeCall *>(enif_alloc_resource(
      resource_object<PendingRuntimeCall>::type, sizeof(PendingRuntimeCall)));
  if (pending == nullptr) {
    throw std::runtime_error(
        "emlx::native: failed to allocate runtime_call pending resource");
  }
  new (pending) PendingRuntimeCall();

  // Pre-allocate every output buffer now so the destination pointers are
  // stable before we hand the request to the Elixir side.
  pending->outputs.reserve(outputs.size());
  for (auto &out : outputs) {
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    pending->outputs.emplace_back(out.data<uint8_t>(), out.nbytes());
  }

  ErlNifEnv *msg_env = enif_alloc_env();
  if (msg_env == nullptr) {
    enif_release_resource(pending);
    throw std::runtime_error("emlx::native: failed to allocate msg env for runtime_call");
  }

  std::vector<ERL_NIF_TERM> arg_terms;
  arg_terms.reserve(inputs.size());
  for (const auto &in : inputs) {
    size_t nbytes = in.nbytes();
    ErlNifBinary bin;
    if (!enif_alloc_binary(nbytes, &bin)) {
      enif_release_resource(pending);
      enif_free_env(msg_env);
      throw std::runtime_error(
          "emlx::native: failed to allocate binary for runtime_call arg");
    }
    if (nbytes > 0) {
      std::memcpy(bin.data, in.data<uint8_t>(), nbytes);
    }
    arg_terms.push_back(enif_make_binary(msg_env, &bin));
  }

  ERL_NIF_TERM pending_term = enif_make_resource(msg_env, pending);
  // enif_make_resource/enif_make_copy below keep their own references; drop
  // ours now — the resource stays alive via the message term (and later,
  // whatever term resolve_runtime_call's caller decodes it into) until its
  // own destructor runs.
  enif_release_resource(pending);

  ERL_NIF_TERM args_list = enif_make_list_from_array(
      msg_env, arg_terms.data(), static_cast<unsigned>(arg_terms.size()));

  ERL_NIF_TERM msg =
      enif_make_tuple4(msg_env, enif_make_atom(msg_env, "emlx_runtime_call"),
                       pending_term, enif_make_int64(msg_env, callback_index),
                       args_list);

  ErlNifPid send_pid = caller_pid;
  int sent = enif_send(NULL, &send_pid, msg_env, msg);
  enif_free_env(msg_env);

  if (!sent) {
    throw std::runtime_error(
        "emlx::native: failed to send runtime_call request (target process not alive?)");
  }

  auto is_done = [pending] {
    std::lock_guard<std::mutex> lock(pending->mu);
    return pending->done;
  };

  // g_current_worker is set for the whole lifetime of this OS thread (see
  // emlx_worker.hpp's thread_main) — always non-null here, since this
  // function only ever runs from inside a job posted to some worker (via
  // EMLXRuntimeCall::eval_cpu/eval_gpu, reached through eval_program).
  // The plain-condvar fallback exists only in case that invariant is ever
  // violated (e.g. a future caller that isn't worker-dispatched) rather
  // than silently deadlocking with no queue to pump.
  if (Worker *worker = g_current_worker) {
    worker->pump_until(is_done);
  } else {
    std::unique_lock<std::mutex> lock(pending->mu);
    pending->cv.wait(lock, [pending] { return pending->done; });
  }

  std::lock_guard<std::mutex> lock(pending->mu);
  if (!pending->ok) {
    throw std::runtime_error("emlx::native: runtime_call callback failed: " + pending->error);
  }
}

// resolve_runtime_call/3 NIF — called from the BEAM process that ran the
// real Elixir callback (EMLX.await_worker/2's `:emlx_runtime_call` receive
// clause) to deliver the reply and wake the blocked worker thread.
//
// argv[0] : pending resource ref
// argv[1] : status atom (:ok | :error)
// argv[2] : on :ok, a list of binaries (one per output, in declared order,
//           each exactly `out.nbytes()` long); on :error, a binary error
//           message.
inline ERL_NIF_TERM resolve_runtime_call(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  (void)argc;
  PendingRuntimeCall *pending;
  if (!enif_get_resource(env, argv[0], resource_object<PendingRuntimeCall>::type,
                         reinterpret_cast<void **>(&pending))) {
    return nx::nif::error(env, "Invalid runtime_call pending resource");
  }

  char status_buf[8];
  if (enif_get_atom(env, argv[1], status_buf, sizeof(status_buf), ERL_NIF_LATIN1) <= 0) {
    return nx::nif::error(env, "Invalid runtime_call status atom");
  }
  bool is_ok = std::string(status_buf) == "ok";

  bool decode_ok = true;
  std::string message;

  {
    std::lock_guard<std::mutex> lock(pending->mu);

    if (is_ok) {
      unsigned n;
      if (!enif_get_list_length(env, argv[2], &n)) {
        decode_ok = false;
        message = "runtime_call reply must be a list of binaries";
      } else if (static_cast<size_t>(n) != pending->outputs.size()) {
        decode_ok = false;
        message = "runtime_call reply has " + std::to_string(n) +
                  " outputs, expected " + std::to_string(pending->outputs.size());
      } else {
        ERL_NIF_TERM head, tail, list = argv[2];
        size_t i = 0;
        while (decode_ok && enif_get_list_cell(env, list, &head, &tail)) {
          ErlNifBinary bin;
          if (!enif_inspect_binary(env, head, &bin)) {
            decode_ok = false;
            message = "runtime_call reply element " + std::to_string(i) + " is not a binary";
            break;
          }
          auto &[dst, size] = pending->outputs[i];
          if (bin.size != size) {
            decode_ok = false;
            message = "runtime_call reply output " + std::to_string(i) +
                      " has unexpected byte size (" + std::to_string(bin.size) +
                      ", expected " + std::to_string(size) + ")";
            break;
          }
          if (size > 0) {
            std::memcpy(dst, bin.data, size);
          }
          list = tail;
          i++;
        }
      }
    } else {
      ErlNifBinary bin;
      if (!enif_inspect_binary(env, argv[2], &bin)) {
        decode_ok = false;
        message = "runtime_call error reply is not a binary";
      } else {
        message = std::string(reinterpret_cast<const char *>(bin.data), bin.size);
      }
    }

    pending->ok = is_ok && decode_ok;
    pending->error = message;
    pending->done = true;
  }

  pending->cv.notify_one();
  return nx::nif::ok(env);
}

} // namespace native
} // namespace emlx
