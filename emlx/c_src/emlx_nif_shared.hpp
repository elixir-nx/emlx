#pragma once

// Shared infrastructure used by both emlx_nif.cpp and emlx_fast.cpp.

#include "emlx_async.hpp"
#include "emlx_worker.hpp"
#include "erl_nif.h"
#include "mlx/mlx.h"
#include "nx_nif_utils.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace mlx::core;
using namespace mlx::core::fast;

// MLX 0.31+ uses Shape = SmallVector<int> and Strides = SmallVector<long long>
// which no longer accept implicit construction from std::vector.
static inline mlx::core::Shape to_shape(const std::vector<int> &v) {
  return mlx::core::Shape(v.begin(), v.end());
}
static inline mlx::core::Strides to_strides(const std::vector<int64_t> &v) {
  return mlx::core::Strides(v.begin(), v.end());
}

inline mlx::core::Device string2device(const std::string &atom) {
  if (atom == "cpu") {
    return mlx::core::Device(mlx::core::Device::DeviceType::cpu, 0);
  } else if (atom == "gpu") {
    return mlx::core::Device(mlx::core::Device::DeviceType::gpu, 0);
  }
  throw std::runtime_error("Unknown device: " + atom);
}

// Class to manage the refcount of MLX tensors
class TensorP {
public:
  TensorP(ErlNifEnv *env, const ERL_NIF_TERM arg) : ptr(nullptr) {
    // setup
    if (!enif_get_resource(env, arg, resource_object<mlx::core::array>::type,
                           (void **)&ptr)) {
      error_message_ = "Unable to get tensor param in NIF";
      err = nx::nif::error(env, error_message_.c_str());
      return;
    }

    refcount = (std::atomic<int> *)(ptr + 1);
    deleted = (std::atomic_flag *)(refcount + 1);

    if (refcount->load() == 0) {
      // already deallocated
      ptr = nullptr;
      error_message_ = "Tensor has been deallocated";
      err = nx::nif::error(env, error_message_.c_str());
      return;
    }

    if (is_valid()) {
      // increase reference count
      ++(*refcount);
    }
  }

  // Movable (needed so a TensorP can be constructed inside a fine::Decoder
  // and returned by value into the async-dispatched NIF's argument list —
  // see the TensorArg bridge below), but never copyable: a bitwise copy
  // would double-decrement the atomic refcount on destruction.
  TensorP(TensorP &&other) noexcept
      : ptr(other.ptr), refcount(other.refcount), deleted(other.deleted),
        err(other.err), error_message_(std::move(other.error_message_)) {
    other.ptr = nullptr;
  }
  TensorP &operator=(TensorP &&other) noexcept {
    if (this != &other) {
      ptr = other.ptr;
      refcount = other.refcount;
      deleted = other.deleted;
      err = other.err;
      error_message_ = std::move(other.error_message_);
      other.ptr = nullptr;
    }
    return *this;
  }
  TensorP(const TensorP &) = delete;
  TensorP &operator=(const TensorP &) = delete;

  ~TensorP() {
    if (is_valid()) {
      // decrease reference count
      if (refcount->fetch_sub(1) == 0) {
        ptr->~array(); // Call MLX tensor destructor
      }
    }
  }

  bool deallocate() {
    if (is_valid() && atomic_flag_test_and_set(deleted) == false) {
      --(*refcount);
      return true;
    } else {
      return false;
    }
  }

  mlx::core::array *data() const { return ptr; }

  // Raw ERTS resource pointer for use with enif_make_resource_binary.
  void *resource_ptr() const { return static_cast<void *>(ptr); }

  bool is_valid() const { return ptr != nullptr; }

  ERL_NIF_TERM error() { return err; }
  const std::string &error_message() const { return error_message_; }

private:
  mlx::core::array *ptr;
  std::atomic<int> *refcount;
  std::atomic_flag *deleted;
  ERL_NIF_TERM err;
  std::string error_message_;
};

#define CATCH()                                                              \
  catch (const std::exception &e) {                                          \
    std::ostringstream msg;                                                  \
    msg << e.what() << " in NIF." << __func__ << "/" << argc;                \
    return nx::nif::error(env, msg.str().c_str());                           \
  }                                                                          \
  catch (...) {                                                              \
    return nx::nif::error(env, "Unknown error occurred");                    \
  }

#define TENSOR(A)                                                            \
  try {                                                                      \
    return nx::nif::ok(env, create_tensor_resource(env, A));                 \
  }                                                                          \
  CATCH()

#define NIF(NAME)                                                            \
  ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

// One-line async wrapper: declare `NIF(OP) { ... }` then `ASYNC_NIF(OP)`.
// Register in `nif_funcs[]` at `original_arity + 1` (command queue is argv[0]).
// Example: {"add", 4, add_async}  // was {"add", 3, add}
#define ASYNC_NIF(OP)                                                        \
  ERL_NIF_TERM OP##_async(ErlNifEnv *env, int argc,                          \
                          const ERL_NIF_TERM argv[]) {                       \
    return emlx::async_dispatch<OP>(env, argc, argv);                        \
  }

#define TENSOR_PARAM(ARGN, VAR)                                              \
  TensorP VAR##_tp(env, argv[ARGN]);                                         \
  mlx::core::array *VAR;                                                     \
  if (!VAR##_tp.is_valid()) {                                                \
    return VAR##_tp.error();                                                 \
  } else {                                                                   \
    VAR = VAR##_tp.data();                                                   \
  }

// Optional tensor argument: the Elixir caller passes the atom `nil` when the
// tensor is absent (e.g. biases for a microscaled quantization mode, or
// sinks for a plain SDPA call); any other term is decoded as a tensor
// resource. VAR is `nullptr` when absent.
#define OPTIONAL_TENSOR_PARAM(ARGN, VAR)                                     \
  std::optional<TensorP> VAR##_tp;                                          \
  mlx::core::array *VAR = nullptr;                                          \
  {                                                                         \
    std::string VAR##_nil_check;                                           \
    bool VAR##_is_nil = nx::nif::get_atom(env, argv[ARGN], VAR##_nil_check) &&\
                        VAR##_nil_check == "nil";                           \
    if (!VAR##_is_nil) {                                                   \
      VAR##_tp.emplace(env, argv[ARGN]);                                   \
      if (!VAR##_tp->is_valid()) {                                         \
        return VAR##_tp->error();                                          \
      }                                                                    \
      VAR = VAR##_tp->data();                                              \
    }                                                                      \
  }

// Forward declaration — defined in emlx_nif.cpp, used in emlx_fast.cpp and
// emlx/compiler.cpp.
ERL_NIF_TERM create_tensor_resource(ErlNifEnv *env, mlx::core::array tensor);

// Dtype name ↔ mlx::core::Dtype mapping — shared across emlx_nif.cpp and
// emlx/compiler.cpp.
inline const std::map<std::string, mlx::core::Dtype> &dtype_map() {
  static const std::map<std::string, mlx::core::Dtype> table = {
      {"bool", mlx::core::bool_},         {"uint8", mlx::core::uint8},
      {"uint16", mlx::core::uint16},      {"uint32", mlx::core::uint32},
      {"uint64", mlx::core::uint64},      {"int8", mlx::core::int8},
      {"int16", mlx::core::int16},        {"int32", mlx::core::int32},
      {"int64", mlx::core::int64},        {"float16", mlx::core::float16},
      {"float32", mlx::core::float32},    {"bfloat16", mlx::core::bfloat16},
      {"complex64", mlx::core::complex64}};
  return table;
}

inline mlx::core::Dtype string2dtype(const std::string &atom) {
  const auto &table = dtype_map();
  auto it = table.find(atom);
  if (it != table.end())
    return it->second;
  throw std::runtime_error("Unknown dtype: " + atom);
}

inline const std::string *dtype2string(const mlx::core::Dtype dtype) {
  for (const auto &pair : dtype_map()) {
    if (pair.second == dtype)
      return &pair.first;
  }
  return nullptr;
}

// ─── `fine` bridging ─────────────────────────────────────────────────────
//
// `fine::nif()`/`FINE_NIF` (the library's own typed-dispatch entry points)
// translate a thrown C++ exception into a *raised* Elixir exception via
// `enif_raise_exception`. That only makes sense as the return value of a
// live NIF call serviced directly by the BEAM scheduler. EMLX's ASYNC_NIF /
// `emlx::async_dispatch` convention instead runs the sync NIF body on a
// worker thread and ships its `{:ok, _}` / `{:error, _}` tagged-tuple
// result back to the caller as an ordinary term via `enif_send` (see
// emlx_async.hpp) — `enif_raise_exception`'s special return sentinel is
// meaningless in that context (there is no live NIF call for the BEAM to
// attach the exception to), so `fine::nif()` cannot be used as-is here.
//
// Bridge: reuse `fine`'s `Decoder<Args>...`/`Encoder<Return>` machinery
// (typed arg decode, exception-safe dispatch) via our own dispatcher that
// funnels failures through the *existing* `nx::nif::error(env, msg)`
// tuple convention instead of `enif_raise_exception`. This keeps every
// existing async/registration/error-surfacing mechanism (and the
// `EMLX.NIFError` Elixir-side contract) unchanged; `fine` supplies argument
// marshalling only.
//
// Tensor resources are bridged the same way: fine's `Decoder`/`Encoder`
// are specialized below for `TensorArg`/`mlx::core::array` against the
// *existing* `TensorP` class and `create_tensor_resource` — not
// `fine::ResourcePtr`. `fine::ResourcePtr<T>` only wraps ERTS's own
// `enif_keep_resource`/`enif_release_resource` refcounting; `TensorP` adds
// a second, independent atomic refcount + "deleted" flag on top, used by
// the explicit `deallocate` NIF (emlx_nif.cpp) to free GPU memory ahead of
// BEAM GC. That's a real semantic layer `fine::ResourcePtr` does not
// provide — `TensorP` cannot be a mechanical `fine::ResourcePtr` swap and
// must stay as a bridged custom type.

#include <fine.hpp>

// Decoded tensor argument: owns a TensorP (RAII refcount bump/decrement for
// the duration of the call, mirroring TENSOR_PARAM's local today) plus the
// raw `array*` for ergonomic `*x` / `x->...` use in NIF bodies.
struct TensorArg {
  TensorP tp;
  mlx::core::array *ptr;

  mlx::core::array &operator*() const { return *ptr; }
  mlx::core::array *operator->() const { return ptr; }
};

namespace fine {

template <> struct Decoder<TensorArg> {
  static TensorArg decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    TensorP tp(env, term);
    if (!tp.is_valid()) {
      throw std::invalid_argument(tp.error_message());
    }
    auto *ptr = tp.data();
    return TensorArg{std::move(tp), ptr};
  }
};

// `std::optional<TensorArg>` mirrors OPTIONAL_TENSOR_PARAM: the Elixir
// caller passes the atom `nil` when the tensor is absent.
template <> struct Decoder<std::optional<TensorArg>> {
  static std::optional<TensorArg> decode(ErlNifEnv *env,
                                         const ERL_NIF_TERM &term) {
    std::string atom_val;
    if (nx::nif::get_atom(env, term, atom_val) && atom_val == "nil") {
      return std::nullopt;
    }
    return Decoder<TensorArg>::decode(env, term);
  }
};

// Device atom (`:cpu` / `:gpu`) -> mlx::core::Device, matching DEVICE_PARAM.
template <> struct Decoder<mlx::core::Device> {
  static mlx::core::Device decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    std::string atom_val;
    if (!nx::nif::get_atom(env, term, atom_val)) {
      throw std::invalid_argument("Unable to get device atom in NIF");
    }
    return string2device(atom_val);
  }
};

// Dtype atom (`:float32`, `:bool`, …) -> mlx::core::Dtype. Lets NIF argument
// types (e.g. emlx/compiler.hpp's `Program::constants`) decode dtypes
// directly instead of going through an int<->dtype lookup table shared with
// Elixir (see EMLX.Native.to_mlx_type/1 on the Elixir side).
template <> struct Decoder<mlx::core::Dtype> {
  static mlx::core::Dtype decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    std::string atom_val;
    if (!nx::nif::get_atom(env, term, atom_val)) {
      throw std::invalid_argument("Unable to get dtype atom in NIF");
    }
    return string2dtype(atom_val);
  }
};

// Plain mlx::core::array argument (as opposed to TensorArg, which also keeps
// the TensorP RAII wrapper alive for the duration of the call) — a straight
// copy of the resource's array value, matching the pre-`fine` LIST_PARAM(...,
// std::vector<mlx::core::array>, ...) semantics used e.g. for compile_program's
// captures.
template <> struct Decoder<mlx::core::array> {
  static mlx::core::array decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    return *Decoder<TensorArg>::decode(env, term);
  }
};

// `int` — fine only ships int64_t/uint64_t decoders; match the existing
// nx::nif::get(..., int*)/enif_get_int semantics exactly (used for e.g.
// RoPE `dims`/`offset` params).
template <> struct Decoder<int> {
  static int decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    int value;
    if (!enif_get_int(env, term, &value)) {
      throw std::invalid_argument("Unable to get int param in NIF");
    }
    return value;
  }
};

// Tensor return value -> the existing create_tensor_resource (unchanged
// resource type / refcount scheme, not fine::ResourcePtr).
template <> struct Encoder<mlx::core::array> {
  static ERL_NIF_TERM encode(ErlNifEnv *env, const mlx::core::array &value) {
    return create_tensor_resource(env, value);
  }
};

} // namespace fine

namespace emlx_fine {

// Mirrors fine::__private__::nif_impl, but on failure returns EMLX's own
// `{:error, msg}` tuple (nx::nif::error) instead of raising — see the
// rationale comment above.
template <typename Return, typename... Args, std::size_t... Indices>
ERL_NIF_TERM nif_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                      Return (*fun)(ErlNifEnv *, Args...), const char *name,
                      std::index_sequence<Indices...>) {
  try {
    auto result = fun(env, fine::decode<Args>(env, argv[Indices])...);
    return nx::nif::ok(env, fine::encode(env, result));
  } catch (const std::exception &e) {
    std::ostringstream msg;
    msg << e.what() << " in NIF." << name << "/" << argc;
    return nx::nif::error(env, msg.str().c_str());
  } catch (...) {
    return nx::nif::error(env, "Unknown error occurred");
  }
}

template <typename Return, typename... Args>
ERL_NIF_TERM nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                 Return (*fun)(ErlNifEnv *, Args...), const char *name) {
  if (static_cast<int>(sizeof...(Args)) != argc) {
    return nx::nif::error(env, "wrong number of arguments");
  }
  return nif_impl(env, argc, argv, fun, name,
                  std::make_index_sequence<sizeof...(Args)>());
}

} // namespace emlx_fine

// Declares `NAME##_impl(ErlNifEnv*, <typed args>...)` (typed body, written
// by the caller right above this macro) as a `fine`-dispatched sync NIF
// named `NAME`, then wires it into the existing ASYNC_NIF/nif_funcs[]
// machinery exactly as `NIF(NAME) { ... } ASYNC_NIF(NAME)` did — the
// generated `NAME`/`NAME_async` symbols and registered arity are
// unchanged, so no emlx_nif.cpp registration-table edits are needed.
#define FINE_ASYNC_NIF(NAME)                                                 \
  ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {    \
    return emlx_fine::nif(env, argc, argv, NAME##_impl, #NAME);              \
  }                                                                          \
  ASYNC_NIF(NAME)
