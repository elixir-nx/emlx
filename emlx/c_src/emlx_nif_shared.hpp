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
      err = nx::nif::error(env, "Unable to get tensor param in NIF");
      return;
    }

    refcount = (std::atomic<int> *)(ptr + 1);
    deleted = (std::atomic_flag *)(refcount + 1);

    if (refcount->load() == 0) {
      // already deallocated
      ptr = nullptr;
      err = nx::nif::error(env, "Tensor has been deallocated");
      return;
    }

    if (is_valid()) {
      // increase reference count
      ++(*refcount);
    }
  }

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

private:
  mlx::core::array *ptr;
  std::atomic<int> *refcount;
  std::atomic_flag *deleted;
  ERL_NIF_TERM err;
};

#define CATCH()                                                                \
  catch (const std::exception &e) {                                            \
    std::ostringstream msg;                                                    \
    msg << e.what() << " in NIF." << __func__ << "/" << argc;                  \
    return nx::nif::error(env, msg.str().c_str());                             \
  }                                                                            \
  catch (...) {                                                                \
    return nx::nif::error(env, "Unknown error occurred");                      \
  }

#define TENSOR(A)                                                              \
  try {                                                                        \
    return nx::nif::ok(env, create_tensor_resource(env, A));                   \
  }                                                                            \
  CATCH()

#define NIF(NAME)                                                              \
  ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

// One-line async wrapper: declare `NIF(OP) { ... }` then `ASYNC_NIF(OP)`.
// Register in `nif_funcs[]` at `original_arity + 1` (command queue is argv[0]).
// Example: {"add", 4, add_async}  // was {"add", 3, add}
#define ASYNC_NIF(OP)                                                          \
  ERL_NIF_TERM OP##_async(ErlNifEnv *env, int argc,                            \
                          const ERL_NIF_TERM argv[]) {                         \
    return emlx::async_dispatch<OP>(env, argc, argv);                          \
  }

#define TENSOR_PARAM(ARGN, VAR)                                                \
  TensorP VAR##_tp(env, argv[ARGN]);                                           \
  mlx::core::array *VAR;                                                       \
  if (!VAR##_tp.is_valid()) {                                                  \
    return VAR##_tp.error();                                                   \
  } else {                                                                     \
    VAR = VAR##_tp.data();                                                     \
  }

// Optional tensor argument: the Elixir caller passes the atom `nil` when the
// tensor is absent (e.g. biases for a microscaled quantization mode, or
// sinks for a plain SDPA call); any other term is decoded as a tensor
// resource. VAR is `nullptr` when absent.
#define OPTIONAL_TENSOR_PARAM(ARGN, VAR)                                       \
  std::optional<TensorP> VAR##_tp;                                            \
  mlx::core::array *VAR = nullptr;                                            \
  {                                                                           \
    std::string VAR##_nil_check;                                             \
    bool VAR##_is_nil = nx::nif::get_atom(env, argv[ARGN], VAR##_nil_check) &&\
                        VAR##_nil_check == "nil";                             \
    if (!VAR##_is_nil) {                                                     \
      VAR##_tp.emplace(env, argv[ARGN]);                                     \
      if (!VAR##_tp->is_valid()) {                                           \
        return VAR##_tp->error();                                            \
      }                                                                      \
      VAR = VAR##_tp->data();                                                \
    }                                                                        \
  }

// Forward declaration — defined in emlx_nif.cpp, used in emlx_fast.cpp and
// emlx_compiler.cpp.
ERL_NIF_TERM create_tensor_resource(ErlNifEnv *env, mlx::core::array tensor);

// Dtype name ↔ mlx::core::Dtype mapping — shared across emlx_nif.cpp and
// emlx_compiler.cpp.
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
