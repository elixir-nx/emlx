#include "emlx/compiler.hpp"
#include "emlx_nif_shared.hpp"
#include "emlx/plugin/registry.hpp"
#include "emlx_nif_lifecycle.hpp"
#include "emlx/plugin/build_compat.hpp"

#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <cstdio>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

std::map<const std::string, const uint8_t> dtype_sizes = {
    {"bool", mlx::core::bool_.size()},
    {"uint8", mlx::core::uint8.size()},
    {"uint16", mlx::core::uint16.size()},
    {"uint32", mlx::core::uint32.size()},
    {"uint64", mlx::core::uint64.size()},
    {"int8", mlx::core::int8.size()},
    {"int16", mlx::core::int16.size()},
    {"int32", mlx::core::int32.size()},
    {"int64", mlx::core::int64.size()},
    {"float16", mlx::core::float16.size()},
    {"float32", mlx::core::float32.size()},
    {"bfloat16", mlx::core::bfloat16.size()},
    {"complex64", mlx::core::complex64.size()}};


ERL_NIF_TERM
create_tensor_resource(ErlNifEnv *env, mlx::core::array tensor) {
  ERL_NIF_TERM ret;
  mlx::core::array *tensorPtr;
  std::atomic<int> *refcount;

  tensorPtr = (mlx::core::array *)enif_alloc_resource(
      resource_object<mlx::core::array>::type, sizeof(mlx::core::array) +
                                                   sizeof(std::atomic<int>) +
                                                   sizeof(std::atomic_flag));
  if (tensorPtr == NULL)
    return enif_make_badarg(env);

  new (tensorPtr) mlx::core::array(std::move(tensor));
  refcount = new (tensorPtr + 1) std::atomic<int>(1);
  new (refcount + 1) std::atomic_flag();

  ret = enif_make_resource(env, tensorPtr);
  enif_release_resource(tensorPtr);

  return ret;
}

NIF(deallocate) {
  TensorP t(env, argv[0]);
  if (t.deallocate()) {
    return nx::nif::ok(env);
  } else {
    return nx::nif::atom(env, "already_deallocated");
  }
}

NIF(scalar_type) {
  TENSOR_PARAM(0, t);

  const std::string *type_name = dtype2string(t->dtype());

  if (type_name != nullptr)
    return nx::nif::ok(env, enif_make_atom(env, type_name->c_str()));
  else
    return nx::nif::error(env, "Could not determine tensor type.");
}

NIF(shape) {
  TENSOR_PARAM(0, t);

  std::vector<ERL_NIF_TERM> sizes;
  for (int64_t dim = 0; dim < t->ndim(); dim++)
    sizes.push_back(nx::nif::make(env, static_cast<int64_t>(t->shape()[dim])));

  return nx::nif::ok(
      env, enif_make_tuple_from_array(env, sizes.data(), sizes.size()));
}

NIF(ones) {
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::ones(to_shape(shape), type, device));
}

NIF(reshape) {
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::reshape(*t, to_shape(shape), device));
}

NIF(astype) {
  TENSOR_PARAM(0, t);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::astype(*t, type, device));
}

NIF(copy) {
  TENSOR_PARAM(0, t);
  DEVICE_PARAM(1, device);

  TENSOR(mlx::core::copy(*t, device));
}

// Builds the resource binary for `to_blob` in `out_env`. Used by both the
// legacy direct-call NIF and `command_queue_post_to_blob` (which builds the
// term inside the worker thread for delivery via enif_send). May throw
// std::runtime_error on resource allocation failure.
//
// `out_env` is the env the binary will live in (the caller env for the
// legacy path; a process-independent msg_env for the worker path).
static ERL_NIF_TERM to_blob_term(ErlNifEnv *out_env, mlx::core::array *t,
                                 size_t byte_size) {
  if (t->flags().row_contiguous) {
    // Zero-copy: alias the MLX buffer via the existing tensor resource.
    // Invariant: lib/emlx.ex calls eval(tensor) before this NIF, so
    // data<void>() is guaranteed non-null and stable (MLX arrays are immutable
    // once materialised). enif_make_resource_binary keeps the resource alive
    // until the binary is GC'd, decoupling the binary lifetime from Elixir GC
    // of the tensor term.
    return enif_make_resource_binary(out_env, static_cast<void *>(t),
                                     t->data<void>(), byte_size);
  }

  // Non-contiguous: materialise a fresh row-major copy, wrap it in a minimal
  // ERTS resource, and alias that buffer zero-copy.
  // The resource holds only sizeof(mlx::core::array) — no TensorP refcount/
  // deleted-flag tail — because it is never exposed to TensorP or the Elixir
  // side; only the binary holds a reference and default_dtor<array>
  // (~array()) is the sole destructor path.
  auto ct = mlx::core::contiguous(*t);
  mlx::core::eval(ct);

  auto *ct_ptr = static_cast<mlx::core::array *>(enif_alloc_resource(
      resource_object<mlx::core::array>::type, sizeof(mlx::core::array)));
  if (!ct_ptr) {
    throw std::runtime_error("Unable to allocate contiguous-copy resource");
  }

  new (ct_ptr) mlx::core::array(std::move(ct));
  ERL_NIF_TERM resource_bin = enif_make_resource_binary(
      out_env, ct_ptr, ct_ptr->data<void>(), byte_size);
  enif_release_resource(ct_ptr);
  return resource_bin;
}

NIF(to_blob) {
  TENSOR_PARAM(0, t);

  size_t byte_size = t->nbytes();
  if (argc == 2) {
    PARAM(1, int, param_limit);
    byte_size = static_cast<size_t>(param_limit) * t->itemsize();
  }

  try {
    return nx::nif::ok(env, to_blob_term(env, t, byte_size));
  }
  CATCH()
}

uint64_t elem_count(std::vector<int> shape) {
  return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>{});
}

NIF(from_blob) {
  BINARY_PARAM(0, blob);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, type);
  // DEVICE_PARAM(3, device);

  if (blob.size / dtype_sizes[type_atom] < elem_count(shape))
    return nx::nif::error(env,
                          "Binary size is too small for the requested shape");

  try {
    // Allocate MLX buffer and copy data from blob
    size_t byte_size = blob.size;
    allocator::Buffer mlx_buf = allocator::malloc(byte_size);
    void *buf_ptr = mlx_buf.raw_ptr();

    // Copy binary data to MLX buffer
    std::memcpy(buf_ptr, blob.data, byte_size);

    // Create deleter for the buffer
    auto deleter = [](allocator::Buffer buf) { allocator::free(buf); };

    // Create MLX array from the buffer
    TENSOR(mlx::core::array(mlx_buf, to_shape(shape), type, deleter));
  } catch (const std::exception &e) {
    return nx::nif::error(env, e.what());
  } catch (...) {
    return nx::nif::error(env,
                          "Unknown error creating tensor from binary data");
  }
}

NIF(scalar_tensor) {
  SCALAR_PARAM(0, scalar, is_complex);
  TYPE_PARAM(1, type);
  // DEVICE_PARAM(2, device);

  if (is_complex) {
    TENSOR(mlx::core::array(complex_scalar, type))
  } else {
    TENSOR(mlx::core::array(scalar, type))
  }
}

NIF(full) {
  SCALAR_PARAM(0, scalar, is_complex);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, type);
  DEVICE_PARAM(3, device);

  if (is_complex) {
    TENSOR(mlx::core::full(to_shape(shape), complex_scalar, type, device));
  } else {
    TENSOR(mlx::core::full(to_shape(shape), scalar, type, device));
  }
}

NIF(arange) {
  PARAM(0, int, start);
  PARAM(1, int, stop);
  PARAM(2, int, step);
  PARAM(3, bool, integer);
  DEVICE_PARAM(4, device);

  if (integer) {
    TENSOR(mlx::core::arange(start, stop, step, device));
  } else {
    TENSOR(mlx::core::arange(static_cast<double>(start),
                             static_cast<double>(stop),
                             static_cast<double>(step), device));
  }
}

NIF(eye) {
  PARAM(0, int, m);
  PARAM(1, int, n);
  TYPE_PARAM(2, type);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::eye(m, n, 0, type, device));
}

NIF(broadcast_to) {
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);
  DEVICE_PARAM(2, device);

  auto result = mlx::core::broadcast_to(*t, to_shape(shape), device);

  TENSOR(result);
}

NIF(tensordot) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  LIST_PARAM(2, std::vector<int>, axes1);
  LIST_PARAM(3, std::vector<int>, axes2);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::tensordot(*a, *b, axes1, axes2, device));
}

NIF(einsum) {
  // Variadic operand count (2+), e.g. for a 3-operand contraction like
  // "ij,jk,kl->il" — see stack/concatenate above for the same
  // list-of-tensor-resources decode pattern.
  LIST_PARAM(0, std::vector<mlx::core::array>, arrays);

  std::string spec_string;
  if (!nx::nif::get(env, argv[1], spec_string)) {
    return nx::nif::error(env, "Unable to get spec_string param.");
  }

  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::einsum(spec_string, arrays, device));
}

NIF(tri_inv) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, bool, upper);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::linalg::tri_inv(*tensor, upper, device));
}

NIF(linalg_lu) {
  TENSOR_PARAM(0, tensor);
  DEVICE_PARAM(1, device);

  try {
    auto result = mlx::core::linalg::lu(*tensor, device);
    return nx::nif::ok(env, nx::nif::make_list(env, result));
  }
  CATCH()
}

NIF(linalg_qr) {
  TENSOR_PARAM(0, tensor);
  DEVICE_PARAM(1, device);

  try {
    auto [q, r] = mlx::core::linalg::qr(*tensor, device);
    return nx::nif::ok(env, enif_make_tuple2(
      env,
      create_tensor_resource(env, q),
      create_tensor_resource(env, r)));
  }
  CATCH()
}

// MLX's `svd`/`eigh` primitives are CPU-only — they throw on a GPU stream
// (mirroring `emlx/compiler.cpp`'s `k_linalg_cpu`, which pins the same ops
// for the compiled path). Both NIFs below ignore the caller's `device` and
// pin to CPU instead: unified memory means the array's data doesn't need to
// move, only the stream that executes on it, so this is free and lets
// `EMLX.Backend.block/4` call these directly for `:gpu` tensors too instead
// of falling back to `Nx.LinAlg`'s much slower eager `default_expr`.
static const mlx::core::Device kLinalgCpuDevice(mlx::core::Device::DeviceType::cpu, 0);

NIF(linalg_svd) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, bool, compute_uv);
  DEVICE_PARAM(2, device);
  (void)device;

  try {
    auto result = mlx::core::linalg::svd(*tensor, compute_uv, kLinalgCpuDevice);
    return nx::nif::ok(env, nx::nif::make_list(env, result));
  }
  CATCH()
}

NIF(linalg_cholesky) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, bool, upper);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::linalg::cholesky(*tensor, upper, device));
}

NIF(linalg_eigh) {
  TENSOR_PARAM(0, tensor);
  ATOM_PARAM(1, uplo);
  DEVICE_PARAM(2, device);
  (void)device;

  try {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eigh(*tensor, uplo, kLinalgCpuDevice);
    return nx::nif::ok(env, enif_make_tuple2(
      env,
      create_tensor_resource(env, eigenvalues),
      create_tensor_resource(env, eigenvectors)));
  }
  CATCH()
}

NIF(linalg_inv) {
  TENSOR_PARAM(0, tensor);
  DEVICE_PARAM(1, device);

  TENSOR(mlx::core::linalg::inv(*tensor, device));
}

NIF(linalg_pinv) {
  TENSOR_PARAM(0, tensor);
  DEVICE_PARAM(1, device);

  TENSOR(mlx::core::linalg::pinv(*tensor, device));
}

NIF(linalg_solve) {
  TENSOR_PARAM(0, tensorA);
  TENSOR_PARAM(1, tensorB);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::linalg::solve(*tensorA, *tensorB, device));
}

NIF(linalg_solve_triangular) {
  TENSOR_PARAM(0, tensorA);
  TENSOR_PARAM(1, tensorB);
  PARAM(2, bool, upper);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::linalg::solve_triangular(*tensorA, *tensorB, upper, device));
}

NIF(conv_general) {
  TENSOR_PARAM(0, tensor_input);
  TENSOR_PARAM(1, tensor_kernel);
  LIST_PARAM(2, std::vector<int>, strides);
  LIST_PARAM(3, std::vector<int>, padding_low);
  LIST_PARAM(4, std::vector<int>, padding_high);
  LIST_PARAM(5, std::vector<int>, kernel_dilation);
  LIST_PARAM(6, std::vector<int>, input_dilation);
  PARAM(7, int, feature_group_count);
  DEVICE_PARAM(8, device);

  TENSOR(mlx::core::conv_general(
      *tensor_input, *tensor_kernel, strides, padding_low, padding_high,
      kernel_dilation, input_dilation, feature_group_count, false, device));
}

NIF(transpose) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::transpose(*t, axes, device));
}

NIF(pad) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  LIST_PARAM(2, std::vector<int>, low_pad_size);
  LIST_PARAM(3, std::vector<int>, high_pad_size);
  TENSOR_PARAM(4, pad_value);
  DEVICE_PARAM(5, device);

  TENSOR(mlx::core::pad(*t, axes, to_shape(low_pad_size), to_shape(high_pad_size),
                        *pad_value, "constant", device))
};

NIF(sort) {
  TENSOR_PARAM(0, t);
  PARAM(1, int, axis);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::sort(*t, axis, device));
}

NIF(argsort) {
  TENSOR_PARAM(0, t);
  PARAM(1, int, axis);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::argsort(*t, axis, device));
}

NIF(eval) {
  TENSOR_PARAM(0, t);
  mlx::core::eval(*t);
  mlx::core::synchronize();
  return nx::nif::ok(env);
}

// Evaluates several tensors in one round trip instead of one `eval` NIF
// call per ref. `eval_program` itself returns lazy refs (see its comment);
// this batches their materialization. Currently unused from Elixir (kept
// as a general-purpose multi-ref eval primitive) — the earlier in-graph
// `:host_callback` round-trip opcode that motivated it was removed in
// favor of graph-splitting on bare `Nx.runtime_call` (see `EMLX.__compile__/3`
// and `EMLX.Defn.Tree`'s `:__EMLX__` metadata handling in `expr.ex`).
NIF(eval_many) {
  LIST_PARAM(0, std::vector<mlx::core::array>, inputs);
  mlx::core::eval(inputs);
  mlx::core::synchronize();
  return nx::nif::ok(env);
}

NIF(to_device) {
  TENSOR_PARAM(0, t);
  DEVICE_PARAM(1, device);
  TENSOR(mlx::core::contiguous(*t, false, device));
}
ASYNC_NIF(to_device)

NIF(stack) {
  LIST_PARAM(0, std::vector<mlx::core::array>, arrays);
  PARAM(1, int, axis);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::stack(arrays, axis, device));
}

NIF(where) {
  TENSOR_PARAM(0, condition);
  TENSOR_PARAM(1, x);
  TENSOR_PARAM(2, y);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::where(*condition, *x, *y, device));
}

NIF(concatenate) {
  LIST_PARAM(0, std::vector<mlx::core::array>, arrays);
  PARAM(1, int, axis);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::concatenate(arrays, axis, device));
}

NIF(take_along_axis) {
  TENSOR_PARAM(0, t);
  TENSOR_PARAM(1, indices);
  PARAM(2, int, axis);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::take_along_axis(*t, *indices, axis, device));
}

NIF(take) {
  TENSOR_PARAM(0, t);
  TENSOR_PARAM(1, indices);
  PARAM(2, int, axis);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::take(*t, *indices, axis, device));
}

NIF(gather) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<mlx::core::array>, indices);
  LIST_PARAM(2, std::vector<int>, axes);
  LIST_PARAM(3, std::vector<int>, slice_sizes);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::gather(*t, indices, axes, to_shape(slice_sizes), device));
}

NIF(scatter_add) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<mlx::core::array>, indices);
  TENSOR_PARAM(2, tensor_updates);
  LIST_PARAM(3, std::vector<int>, axes);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::scatter_add(*t, indices, *tensor_updates, axes, device));
}

NIF(scatter) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<mlx::core::array>, indices);
  TENSOR_PARAM(2, tensor_updates);
  LIST_PARAM(3, std::vector<int>, axes);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::scatter(*t, indices, *tensor_updates, axes, device));
}

/* Reduction Ops */

#define REDUCTION_AXES_OP(OP) REDUCTION_AXES_OP2(OP, OP)

#define REDUCTION_AXES_OP2(OP, NATIVE_OP)                                      \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, tensor);                                                   \
    LIST_PARAM(1, std::vector<int>, axes);                                     \
    PARAM(2, bool, keep_dims);                                                 \
    DEVICE_PARAM(3, device);                                                   \
                                                                               \
    if (axes.empty()) {                                                        \
      for (int i = 0; i < tensor->ndim(); ++i) {                               \
        axes.push_back(i);                                                     \
      }                                                                        \
    }                                                                          \
    TENSOR(mlx::core::NATIVE_OP(*tensor, axes, keep_dims, device));            \
  }

#define REDUCTION_AXIS_OP(OP) REDUCTION_AXIS_OP2(OP, OP)

#define REDUCTION_AXIS_OP2(OP, NATIVE_OP)                                      \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, tensor);                                                   \
    if (argc == 3) {                                                           \
      PARAM(1, bool, keep_dims);                                               \
      DEVICE_PARAM(2, device);                                                 \
      TENSOR(mlx::core::NATIVE_OP(*tensor, keep_dims, device));                \
    } else {                                                                   \
      PARAM(1, int, axis);                                                     \
      PARAM(2, bool, keep_dims);                                               \
      DEVICE_PARAM(3, device);                                                 \
      TENSOR(mlx::core::NATIVE_OP(*tensor, axis, keep_dims, device));          \
    }                                                                          \
  }

#define REDUCTION_AXIS_REVERSIBLE_OP(OP) REDUCTION_AXIS_REVERSIBLE_OP2(OP, OP)

#define REDUCTION_AXIS_REVERSIBLE_OP2(OP, NATIVE_OP)                           \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, tensor);                                                   \
    PARAM(1, int, axis);                                                       \
    PARAM(2, bool, keep_dims);                                                 \
    DEVICE_PARAM(3, device);                                                   \
                                                                               \
    TENSOR(mlx::core::NATIVE_OP(*tensor, axis, keep_dims, device));            \
  }

REDUCTION_AXES_OP(all)
REDUCTION_AXES_OP(any)
REDUCTION_AXES_OP(sum)
REDUCTION_AXES_OP2(product, prod)
REDUCTION_AXIS_OP(argmax)
REDUCTION_AXIS_OP(argmin)

NIF(cumulative_sum) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, int, axis);
  PARAM(2, bool, reverse);
  PARAM(3, bool, inclusive);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::cumsum(*tensor, axis, reverse, inclusive, device));
}

NIF(cumulative_product) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, int, axis);
  PARAM(2, bool, reverse);
  PARAM(3, bool, inclusive);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::cumprod(*tensor, axis, reverse, inclusive, device));
}

NIF(cumulative_max) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, int, axis);
  PARAM(2, bool, reverse);
  PARAM(3, bool, inclusive);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::cummax(*tensor, axis, reverse, inclusive, device));
}

NIF(cumulative_min) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, int, axis);
  PARAM(2, bool, reverse);
  PARAM(3, bool, inclusive);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::cummin(*tensor, axis, reverse, inclusive, device));
}

/* Unary Ops */

#define UNARY_OP(OP) UNARY_OP2(OP, OP)

#define UNARY_OP2(OP, NATIVE_OP)                                               \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, tensor);                                                   \
    DEVICE_PARAM(1, device);                                                   \
                                                                               \
    TENSOR(mlx::core::NATIVE_OP(*tensor, device));                             \
  }

/* Binary Ops */

#define BINARY_OP(OP) BINARY_OP2(OP, OP)

#define BINARY_OP2(OP, NATIVE_OP)                                              \
  NIF(OP) {                                                                    \
    TENSOR_PARAM(0, a);                                                        \
    TENSOR_PARAM(1, b);                                                        \
    DEVICE_PARAM(2, device);                                                   \
                                                                               \
    TENSOR(mlx::core::NATIVE_OP(*a, *b, device));                              \
  }

// Generate a unique POSIX shared-memory name of the form /emlx_<16hex>.
// Names must begin with '/' and be short enough for shm_open on all platforms.
// Uses thread_local state so concurrent NIF calls don't race on the RNG.
static std::string generate_shm_name() {
  thread_local std::mt19937_64 gen(std::random_device{}());
  thread_local std::uniform_int_distribution<uint64_t> dist;
  char buf[32];
  snprintf(buf, sizeof(buf), "/emlx_%016llx", (unsigned long long)dist(gen));
  return std::string(buf);
}

// ── IPC shared-memory interop ─────────────────────────────────────────────────
//
// These two NIFs implement the :ipc mode of Nx.Backend.to_pointer / from_pointer
// using POSIX shared memory (shm_open + mmap).  MLX arrays are immutable, so
// the sender *copies* tensor data into the shm segment — there is no zero-copy
// path here (documented as copy semantics in the Elixir layer).
//
// Lifecycle:
//   Sender  (tensor_to_shm): creates shm, memcpy, munmap+close fd.  Name persists
//   Receiver (array_from_shm): shm_open, mmap, shm_unlink immediately (keeps object
//   alive via fd), creates mlx::array with a deleter that munmap+closes on GC.

// Creates a POSIX shm segment containing a contiguous copy of the tensor's data.
// argv[0]: tensor_ref   (must already be eval'd — Elixir calls eval before this NIF)
// argv[1]: permissions  (mode_t expressed as uint64, e.g. 0o400 = 256)
// Returns: {:ok, {name_binary, byte_size}} on success.
NIF(tensor_to_shm) {
  TENSOR_PARAM(0, t);
  PARAM(1, size_t, permissions);

  if (t->data<void>() == nullptr) {
    return nx::nif::error(env,
        "Tensor not evaluated; call EMLX.eval/1 before to_pointer with mode: :ipc");
  }

  // Ensure contiguous layout before exposing to shared memory.
  size_t byte_size;
  void *src_ptr;

  // Use optional to avoid default-constructing mlx::core::array.
  std::optional<mlx::core::array> ct_opt;
  if (t->flags().row_contiguous) {
    byte_size = t->nbytes();
    src_ptr = t->data<void>();
  } else {
    ct_opt.emplace(mlx::core::contiguous(*t));
    mlx::core::eval(*ct_opt);
    byte_size = ct_opt->nbytes();
    src_ptr = ct_opt->data<void>();
  }

  // O_EXCL ensures we create a fresh segment; retry on the rare collision.
  int fd = -1;
  std::string shm_name;
  for (int attempt = 0; attempt < 10; ++attempt) {
    shm_name = generate_shm_name();
    fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, (mode_t)permissions);
    if (fd != -1 || errno != EEXIST) break;
  }
  if (fd == -1) {
    return nx::nif::error(env, "shm_open failed in tensor_to_shm");
  }

  if (ftruncate(fd, (off_t)byte_size) == -1) {
    close(fd);
    shm_unlink(shm_name.c_str());
    return nx::nif::error(env, "ftruncate failed in tensor_to_shm");
  }

  void *ptr = mmap(NULL, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    shm_unlink(shm_name.c_str());
    return nx::nif::error(env, "mmap failed in tensor_to_shm");
  }

  std::memcpy(ptr, src_ptr, byte_size);

  munmap(ptr, byte_size);
  close(fd);
  // shm object persists under shm_name until the receiver calls shm_unlink.

  // Return the shm name as a binary (not a charlist) so %Nx.Pointer{handle: ...}
  // holds a conventional Elixir binary that other backends can consume.
  // ERTS API: enif_make_new_binary(env, size, &term) returns the writable ptr.
  ERL_NIF_TERM name_term;
  unsigned char *bin_data = enif_make_new_binary(env, shm_name.size(), &name_term);
  std::memcpy(bin_data, shm_name.data(), shm_name.size());
  ERL_NIF_TERM size_term = nx::nif::make(env, byte_size);
  return nx::nif::ok(env, enif_make_tuple2(env, name_term, size_term));
}

// Opens an existing POSIX shm segment and wraps it as an MLX array.
// The shm is unlinked immediately after mmap so cleanup is automatic:
// when the returned MLX array is GC'd, its deleter calls munmap + close.
// argv[0]: name_binary  (POSIX shm name string)
// argv[1]: shape        (tuple of ints)
// argv[2]: dtype        (atom, e.g. :float32)
// argv[3]: byte_size    (uint64, validated against computed size)
// Returns: {:ok, tensor_ref} on success.
NIF(array_from_shm) {
  std::string shm_name;
  if (!nx::nif::get(env, argv[0], shm_name))
    return nx::nif::error(env, "Unable to get shm name param");

  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, dtype);
  PARAM(3, size_t, byte_size);

  if (shm_name.empty()) {
    return nx::nif::error(env, "Empty shm name");
  }

  // Try read-write first; fall back to read-only if permission denied.
  int writable = 1;
  int fd = shm_open(shm_name.c_str(), O_RDWR, 0);
  if (fd == -1 && errno == EACCES) {
    fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
    writable = 0;
  }
  if (fd == -1) {
    return nx::nif::error(env, "shm_open failed in array_from_shm");
  }

  int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
  void *ptr = mmap(NULL, byte_size, prot, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    return nx::nif::error(env, "mmap failed in array_from_shm");
  }

  // Unlink immediately: the name is removed, but the object lives as long as
  // this fd (and thus this mmap) is open.  The deleter owns cleanup.
  shm_unlink(shm_name.c_str());

  try {
    // Capture fd and byte_size in the deleter; MLX calls it exactly once.
    auto deleter = [fd, byte_size](void *p) {
      munmap(p, byte_size);
      close(fd);
    };
    auto arr = mlx::core::array(ptr, to_shape(shape), dtype, deleter);
    return nx::nif::ok(env, create_tensor_resource(env, std::move(arr)));
  }
  CATCH()
}

// Unlinks a POSIX shm segment by name.  Call this if the receiver never opens
// the pointer returned by tensor_to_shm — otherwise the shm name persists in
// /dev/shm until the next reboot.
// argv[0]: name binary (the handle from %Nx.Pointer{kind: :ipc, handle: name})
NIF(shm_unlink_handle) {
  std::string shm_name;
  if (!nx::nif::get(env, argv[0], shm_name))
    return nx::nif::error(env, "Unable to get shm name param");

  if (shm_unlink(shm_name.c_str()) == -1 && errno != ENOENT) {
    return nx::nif::error(env, "shm_unlink failed");
  }
  return nx::nif::ok(env);
}

// Returns the raw data pointer of an evaluated tensor as a {address, byte_size}
// tuple of uint64 values. The Elixir caller must call EMLX.eval/1 first so
// that data<void>() is non-null and stable (MLX arrays are immutable once
// materialised). On Apple Silicon the pointer is accessible from both CPU and
// GPU due to unified memory. Primary use case: sharing tensors with Python MLX
// via Pythonx using the Nx.Backend.to_pointer/from_pointer protocol.
NIF(tensor_data_ptr) {
  TENSOR_PARAM(0, t);

  if (t->data<void>() == nullptr) {
    return nx::nif::error(
        env, "Tensor has not been evaluated; call EMLX.eval/1 before to_pointer");
  }

  size_t addr = reinterpret_cast<size_t>(t->data<void>());
  size_t byte_size = t->nbytes();

  ERL_NIF_TERM addr_term = nx::nif::make(env, addr);
  ERL_NIF_TERM size_term = nx::nif::make(env, byte_size);
  return nx::nif::ok(env, enif_make_tuple2(env, addr_term, size_term));
}

// Wraps an external raw pointer as an MLX array with a no-op deleter.
// The caller is responsible for keeping the backing buffer alive for the
// duration of use (see include/emlx.h for the lifetime contract).
// argv[0]: address  (uint64 / size_t)
// argv[1]: shape    (tuple of ints)
// argv[2]: dtype    (atom, e.g. :float32)
// argv[3]: byte_size (uint64, validated but not used by the MLX ctor)
// argv[4]: deleter  (reserved / ignored; pass nil)
NIF(array_from_ptr) {
  PARAM(0, size_t, raw_addr);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, dtype);
  // argv[3] byte_size and argv[4] deleter are accepted but deferred.

  if (raw_addr == 0) {
    return nx::nif::error(env, "Null pointer passed to array_from_ptr");
  }

  try {
    void *ptr = reinterpret_cast<void *>(raw_addr);
    // No-op deleter: the caller owns the buffer.
    auto arr =
        mlx::core::array(ptr, to_shape(shape), dtype, [](void *) {});
    return nx::nif::ok(env, create_tensor_resource(env, std::move(arr)));
  }
  CATCH()
}

// ─── Worker / EMLX.CommandQueue NIFs ────────────────────────────────────────
//
// Lifecycle for posted jobs:
//   1. Caller's NIF env (`env`) is short-lived — we cannot hand its terms
//      to the worker thread.
//   2. We allocate a process-independent ErlNifEnv (`msg_env`) per job.
//      The job_ref and the reply tuple live in `msg_env`.
//   3. The job_ref is also enif_make_copy'd into the caller's `env` so
//      the wrapper in lib/emlx.ex can `receive {^job_ref, _}`.
//   4. The lambda posted to the worker captures `t_ptr`, `t_refcount`,
//      `caller_pid`, `msg_env`, and `job_ref_msg` by value. Tensor
//      lifetime across the post boundary is held two ways:
//        - enif_keep_resource on t_ptr (ERTS resource refcount)
//        - ++(*t_refcount) on the embedded TensorP refcount
//      Both are dropped at the end of the lambda.
//   5. After enif_send, msg_env is freed inside the lambda.
//
// Stop semantics: if the Worker is destroyed (NIF resource refcount drops
// to zero), pending jobs already in the queue at destructor time still
// run and still send their reply. Jobs posted *after* the destructor
// begins throw on post() and the NIF returns {:error, _} synchronously.

NIF(command_queue_new) {
  ATOM_PARAM(0, device_atom);

  try {
    mlx::core::Device device = string2device(device_atom);

    // Guard before spawning any threads: mlx::core::new_stream on an
    // unavailable device (e.g. gpu on Linux/CPU-only libmlx) does not
    // throw — it calls std::terminate() internally, which we cannot
    // catch. Check availability here and surface a clean {:error, _}
    // so EMLX.Application can skip the GPU worker on unsupported hosts.
    if (!mlx::core::is_available(device)) {
      return nx::nif::error(env, "device not available");
    }

    auto *worker_ptr = static_cast<emlx::Worker *>(enif_alloc_resource(
        resource_object<emlx::Worker>::type, sizeof(emlx::Worker)));
    if (!worker_ptr) {
      return enif_make_badarg(env);
    }

    try {
      new (worker_ptr) emlx::Worker(device);
    } catch (...) {
      // Placement new failed before constructor completed; the resource
      // memory is uninitialised so we must NOT call ~Worker. Just release
      // the bare allocation and re-throw to the outer handler.
      enif_release_resource(worker_ptr);
      throw;
    }

    ERL_NIF_TERM ref = enif_make_resource(env, worker_ptr);
    enif_release_resource(worker_ptr);
    return nx::nif::ok(env, ref);
  }
  CATCH()
}

// Posts a no-op barrier job that calls mx::synchronize(stream). The
// Elixir wrapper blocks in `receive` until the reply lands, which only
// happens after every preceding job on this worker has completed AND
// MLX has flushed the GPU command buffer.
NIF(command_queue_synchronize) {
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

  mlx::core::Stream stream = worker->stream();

  try {
    worker->post([stream, caller_pid, msg_env, job_ref_msg]() mutable {
      ERL_NIF_TERM result;
      try {
        mlx::core::synchronize(stream);
        result = enif_make_tuple2(msg_env, enif_make_atom(msg_env, "ok"),
                                  enif_make_atom(msg_env, "ok"));
      } catch (const std::exception &e) {
        result = enif_make_tuple2(
            msg_env, enif_make_atom(msg_env, "error"),
            enif_make_string(msg_env, e.what(), ERL_NIF_LATIN1));
      } catch (...) {
        result = enif_make_tuple2(
            msg_env, enif_make_atom(msg_env, "error"),
            enif_make_string(msg_env, "Unknown error in synchronize",
                             ERL_NIF_LATIN1));
      }

      ERL_NIF_TERM msg = enif_make_tuple2(msg_env, job_ref_msg, result);
      ErlNifPid pid = caller_pid;
      enif_send(NULL, &pid, msg_env, msg);
      enif_free_env(msg_env);
    });
  } catch (const std::exception &e) {
    enif_free_env(msg_env);
    return nx::nif::error(env, e.what());
  }

  return nx::nif::ok(env, job_ref_caller);
}

// `eval` and `to_blob` are now worker-routed via `ASYNC_NIF` (see the
// async wrapper block near `nif_funcs[]`). The bespoke
// `command_queue_post_eval` / `command_queue_post_to_blob` NIFs were
// removed in favour of that uniform dispatch.

static int open_resources(ErlNifEnv *env) {
  const char *mod = "EMLX";
  if (!open_resource<mlx::core::array>(env, mod, "MLXArray")) {
    return -1;
  }

  // emlx::Worker — backs EMLX.CommandQueue and the application default
  // worker. Default destructor (~Worker) signals stop, drains pending
  // jobs, and joins the OS thread.
  if (!open_resource<emlx::Worker>(env, mod, "CommandQueue")) {
    return -1;
  }

  // emlx::native::Expr — opaque compiled program resource for the defn compiler.
  if (!open_resource<emlx::native::Expr>(env, mod, "NativeProgram")) {
    return -1;
  }

  // emlx::native::PendingRuntimeCall — opaque handle for one in-flight
  // Nx.runtime_call/4 round trip (see emlx_runtime_call_bridge.hpp).
  if (!open_resource<emlx::native::PendingRuntimeCall>(env, mod, "PendingRuntimeCall")) {
    return -1;
  }

  return 0;
}

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  (void)priv_data;
  (void)load_info;
  return emlx_initialize_nif_runtime(env, open_resources,
                                     EMLX_EXPECTED_MLX_BUILD_ID);
}

int upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data, ERL_NIF_TERM load_info) {
  return emlx_upgrade_nif_runtime(env, priv_data, old_priv_data, load_info);
}

UNARY_OP(abs)
UNARY_OP(ceil)
UNARY_OP(conjugate)
UNARY_OP(floor)
UNARY_OP2(negate, negative)
UNARY_OP(round)
UNARY_OP(sign)
UNARY_OP(real)
UNARY_OP(imag)
UNARY_OP2(is_nan, isnan)
UNARY_OP2(is_infinity, isinf)
UNARY_OP(logical_not)
UNARY_OP(sigmoid)

UNARY_OP2(asin, arcsin)
UNARY_OP2(asinh, arcsinh)
UNARY_OP2(acos, arccos)
UNARY_OP2(acosh, arccosh)
UNARY_OP2(atan, arctan)
UNARY_OP2(atanh, arctanh)
UNARY_OP(cos)
UNARY_OP(cosh)
UNARY_OP(erf)
UNARY_OP2(erf_inv, erfinv)
UNARY_OP(exp)
UNARY_OP(expm1)
UNARY_OP(log)
UNARY_OP(log1p)
UNARY_OP(rsqrt)
UNARY_OP(sin)
UNARY_OP(sinh)
UNARY_OP(sqrt)
UNARY_OP(tan)
UNARY_OP(tanh)

BINARY_OP(add)
BINARY_OP(subtract)
BINARY_OP(multiply)
BINARY_OP2(pow, power)
BINARY_OP2(remainder, remainder)
BINARY_OP2(divide, divide)
BINARY_OP2(atan2, arctan2)
BINARY_OP2(minimum, minimum)
BINARY_OP2(maximum, maximum)
BINARY_OP2(quotient, floor_divide)
BINARY_OP(bitwise_and)
BINARY_OP(bitwise_or)
BINARY_OP(bitwise_xor)
NIF(bitwise_not) {
  TENSOR_PARAM(0, a);
  DEVICE_PARAM(1, device);

  auto dtype = (*a).dtype();
  auto mask = mlx::core::full({}, 0xFFFFFFFFFFFFFFFF, dtype, device);
  TENSOR(mlx::core::subtract(mask, *a, device));
}
BINARY_OP(left_shift)
BINARY_OP(right_shift)
BINARY_OP(equal)
BINARY_OP(not_equal)
BINARY_OP(greater)
BINARY_OP(less)
BINARY_OP(greater_equal)
BINARY_OP(less_equal)
BINARY_OP(logical_and)
BINARY_OP(logical_or)
NIF(logical_xor) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  DEVICE_PARAM(2, device);

  auto t1 = mlx::core::logical_or(*a, *b, device);
  auto t2 =
      mlx::core::logical_not(mlx::core::logical_and(*a, *b, device), device);
  TENSOR(mlx::core::logical_and(t1, t2, device));
}
NIF(allclose) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  PARAM(2, double, rtol);
  PARAM(3, double, atol);
  PARAM(4, bool, equal_nan);
  DEVICE_PARAM(5, device);

  TENSOR(mlx::core::allclose(*a, *b, rtol, atol, equal_nan, device));
}
NIF(isclose) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  PARAM(2, double, rtol);
  PARAM(3, double, atol);
  PARAM(4, bool, equal_nan);
  DEVICE_PARAM(5, device);

  TENSOR(mlx::core::isclose(*a, *b, rtol, atol, equal_nan, device));
}

NIF(item) {
  TENSOR_PARAM(0, t);
  mlx::core::eval(*t);

  // Fix for MLX scalar layout bug: Use the correct type when calling item<T>()
  // to avoid reading wrong number of bytes from potentially invalid memory
  // layouts.
  auto dtype = t->dtype();

  // Handle integer and boolean types with proper dtype matching
  if (dtype == mlx::core::bool_) {
    bool value = t->item<bool>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<int64_t>(value)));
  } else if (dtype == mlx::core::uint8) {
    uint8_t value = t->item<uint8_t>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<int64_t>(value)));
  } else if (dtype == mlx::core::uint16) {
    uint16_t value = t->item<uint16_t>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<int64_t>(value)));
  } else if (dtype == mlx::core::uint32) {
    uint32_t value = t->item<uint32_t>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<int64_t>(value)));
  } else if (dtype == mlx::core::uint64) {
    uint64_t value = t->item<uint64_t>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<int64_t>(value)));
  } else if (dtype == mlx::core::int8) {
    int8_t value = t->item<int8_t>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<int64_t>(value)));
  } else if (dtype == mlx::core::int16) {
    int16_t value = t->item<int16_t>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<int64_t>(value)));
  } else if (dtype == mlx::core::int32) {
    int32_t value = t->item<int32_t>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<int64_t>(value)));
  } else if (dtype == mlx::core::int64) {
    int64_t value = t->item<int64_t>();
    return nx::nif::ok(env, nx::nif::make(env, value));
  } else if (dtype == mlx::core::float16 || dtype == mlx::core::bfloat16) {
    // MLX handles float16/bfloat16 conversion internally
    float value = t->item<float>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<double>(value)));
  } else if (dtype == mlx::core::float32) {
    float value = t->item<float>();
    return nx::nif::ok(env, nx::nif::make(env, static_cast<double>(value)));
  } else if (dtype == mlx::core::complex64) {
    // Complex types need special handling - not supported via item()
    return nx::nif::error(env,
                          "Complex scalar extraction not supported via item()");
  } else {
    // Fallback for any other types
    double value = t->item<double>();
    return nx::nif::ok(env, nx::nif::make(env, value));
  }
}

NIF(slice) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, starts);
  LIST_PARAM(2, std::vector<int>, stops);
  LIST_PARAM(3, std::vector<int>, strides);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::slice(*t, to_shape(starts), to_shape(stops), to_shape(strides), device));
}

NIF(slice_update) {
  TENSOR_PARAM(0, t);
  TENSOR_PARAM(1, tensor_updates);
  LIST_PARAM(2, std::vector<int>, starts);
  LIST_PARAM(3, std::vector<int>, stops);
  DEVICE_PARAM(4, device);
  TENSOR(mlx::core::slice_update(*t, *tensor_updates, to_shape(starts), to_shape(stops), device));
}

NIF(squeeze) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  DEVICE_PARAM(2, device);
  TENSOR(mlx::core::squeeze(*t, axes, device));
}

NIF(emlx_fft) {
  TENSOR_PARAM(0, t);
  PARAM(1, int, n);
  PARAM(2, int, axis);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::fft::fft(*t, n, axis, mlx::core::fft::FFTNorm::Backward, device));
}

NIF(ifft) {
  TENSOR_PARAM(0, t);
  PARAM(1, int, n);
  PARAM(2, int, axis);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::fft::ifft(*t, n, axis, mlx::core::fft::FFTNorm::Backward, device));
}

NIF(emlx_fft2) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, n);
  LIST_PARAM(2, std::vector<int>, axes);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::fft::fft2(*t, to_shape(n), axes, mlx::core::fft::FFTNorm::Backward, device));
}

NIF(ifft2) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, n);
  LIST_PARAM(2, std::vector<int>, axes);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::fft::ifft2(*t, to_shape(n), axes, mlx::core::fft::FFTNorm::Backward, device));
}

NIF(view) {
  TENSOR_PARAM(0, t);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);
  TENSOR(mlx::core::view(*t, type, device));
}

NIF(max) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  PARAM(2, bool, keep_axes);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::max(*t, axes, keep_axes, device));
}

NIF(min) {
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int>, axes);
  PARAM(2, bool, keep_axes);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::min(*t, axes, keep_axes, device));
}

NIF(clip) {
  TENSOR_PARAM(0, t);
  TENSOR_PARAM(1, min);
  TENSOR_PARAM(2, max);
  DEVICE_PARAM(3, device);
  TENSOR(mlx::core::clip(*t, *min, *max, device));
}

NIF(memory_info) {
  size_t active = mlx::core::get_active_memory();
  size_t peak = mlx::core::get_peak_memory();
  size_t cache = mlx::core::get_cache_memory();

  ERL_NIF_TERM keys[] = {
    enif_make_atom(env, "active_memory"),
    enif_make_atom(env, "peak_memory"),
    enif_make_atom(env, "cache_memory")
  };
  ERL_NIF_TERM values[] = {
    enif_make_uint64(env, active),
    enif_make_uint64(env, peak),
    enif_make_uint64(env, cache)
  };

  ERL_NIF_TERM map;
  enif_make_map_from_arrays(env, keys, values, 3, &map);
  return nx::nif::ok(env, map);
}

NIF(clear_cache) {
  mlx::core::clear_cache();
  return nx::nif::ok(env);
}

NIF(reset_peak_memory) {
  mlx::core::reset_peak_memory();
  return nx::nif::ok(env);
}

NIF(set_memory_limit) {
  ErlNifUInt64 limit;
  if (!enif_get_uint64(env, argv[0], &limit))
    return nx::nif::error(env, "Unable to get limit param.");
  uint64_t prev = static_cast<uint64_t>(mlx::core::set_memory_limit(static_cast<size_t>(limit)));
  return nx::nif::ok(env, enif_make_uint64(env, prev));
}

NIF(set_cache_limit) {
  ErlNifUInt64 limit;
  if (!enif_get_uint64(env, argv[0], &limit))
    return nx::nif::error(env, "Unable to get limit param.");
  uint64_t prev = static_cast<uint64_t>(mlx::core::set_cache_limit(static_cast<size_t>(limit)));
  return nx::nif::ok(env, enif_make_uint64(env, prev));
}

// Starts a Metal GPU frame capture, writing the trace to `path` (a
// `.gputrace` bundle). Requires `MTL_CAPTURE_ENABLED=1` in the process
// environment *before* the BEAM (and therefore the Metal device) started —
// Apple's capture gate is read once at process startup and cannot be
// enabled lazily. On MLX 0.31.2, missing this precondition throws (caught
// by CATCH() below and surfaced as a normal EMLX.NIFError), not a silent
// no-op. See EMLX.metal_start_capture/1's moduledoc for details.
NIF(metal_start_capture) {
  std::string path;
  if (!nx::nif::get(env, argv[0], path)) {
    return nx::nif::error(env, "Unable to get path param.");
  }
  try {
    mlx::core::metal::start_capture(path);
    return nx::nif::ok(env);
  }
  CATCH()
}

NIF(metal_stop_capture) {
  try {
    mlx::core::metal::stop_capture();
    return nx::nif::ok(env);
  }
  CATCH()
}

NIF(strides) {
  TENSOR_PARAM(0, t);

  auto raw = t->strides();
  std::vector<int64_t> strides_vec(raw.begin(), raw.end());
  return nx::nif::ok(env, nx::nif::make_list(env, strides_vec));
}

NIF(as_strided) {
  TENSOR_PARAM(0, t);
  TUPLE_PARAM(1, std::vector<int>, shape);
  LIST_PARAM(2, std::vector<int64_t>, strides);
  PARAM(3, int, offset);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::as_strided(*t, to_shape(shape), to_strides(strides), offset, device));
}

// ============================================================================
// Quantization Operations (for 4-bit model support)
// ============================================================================

// quantized_matmul - Multiplies x with a quantized weight matrix w
// This is the key operation for efficient 4-bit inference
// MLX API: quantized_matmul(x, w, scales, biases, transpose, group_size, bits, stream)
// mode: "affine" (default, real biases) or a microscaled variant
// ("mxfp4"/"mxfp8"/"nvfp4" — no biases; mx::fp_quantize returns only
// (wq, scales)). biases is `nil` from Elixir for microscaled modes.
NIF(quantized_matmul) {
  TENSOR_PARAM(0, x);       // Input tensor [batch, seq, hidden]
  TENSOR_PARAM(1, w);       // Quantized weights [out/8, in] (uint32 packed)
  TENSOR_PARAM(2, scales);  // Scales [out/group_size, in] (bfloat16, or u8 for microscaled)
  OPTIONAL_TENSOR_PARAM(3, biases); // Biases (bfloat16); nil for microscaled modes
  PARAM(4, bool, transpose);
  PARAM(5, int, group_size);
  PARAM(6, int, bits);
  std::string mode;
  if (!nx::nif::get(env, argv[7], mode)) {
    return nx::nif::error(env, "Unable to get mode param.");
  }
  DEVICE_PARAM(8, device);

  std::optional<mlx::core::array> biases_opt =
      biases ? std::make_optional(*biases) : std::nullopt;

  TENSOR(mlx::core::quantized_matmul(
      *x, *w, *scales, biases_opt, transpose, group_size, bits, mode, device));
}

// dequantize - Converts quantized weights back to float
// Useful for debugging and verification
// MLX API: dequantize(w, scales, biases, group_size, bits, mode, stream)
NIF(dequantize) {
  TENSOR_PARAM(0, w);       // Quantized weights (uint32 packed)
  TENSOR_PARAM(1, scales);  // Scales (bfloat16, or u8 for microscaled)
  OPTIONAL_TENSOR_PARAM(2, biases); // Biases (bfloat16); nil for microscaled modes
  PARAM(3, int, group_size);
  PARAM(4, int, bits);
  std::string mode;
  if (!nx::nif::get(env, argv[5], mode)) {
    return nx::nif::error(env, "Unable to get mode param.");
  }
  DEVICE_PARAM(6, device);

  std::optional<mlx::core::array> biases_opt =
      biases ? std::make_optional(*biases) : std::nullopt;

  TENSOR(mlx::core::dequantize(*w, *scales, biases_opt, group_size, bits, mode, std::nullopt, std::nullopt, device));
}

// quantize - Quantizes a float tensor to packed format
// Returns a 3-tuple {weights, scales, biases}; biases is the atom `nil` for
// microscaled modes ("mxfp4"/"mxfp8"/"nvfp4"), which don't produce a biases
// array (mx::quantize's `result` vector has 2 elements instead of 3 there).
// MLX API: quantize(w, group_size, bits, mode, stream) -> vector<array>
NIF(quantize) {
  TENSOR_PARAM(0, w);       // Float weights to quantize
  PARAM(1, int, group_size);
  PARAM(2, int, bits);
  std::string mode;
  if (!nx::nif::get(env, argv[3], mode)) {
    return nx::nif::error(env, "Unable to get mode param.");
  }
  DEVICE_PARAM(4, device);

  try {
    auto result = mlx::core::quantize(*w, group_size, bits, mode, std::nullopt, device);

    ERL_NIF_TERM weights_term = create_tensor_resource(env, result[0]);
    ERL_NIF_TERM scales_term = create_tensor_resource(env, result[1]);
    ERL_NIF_TERM biases_term = result.size() > 2
        ? create_tensor_resource(env, result[2])
        : enif_make_atom(env, "nil");

    return nx::nif::ok(env, enif_make_tuple3(env, weights_term, scales_term, biases_term));
  }
  CATCH()
}

ASYNC_NIF(quantized_matmul)
ASYNC_NIF(dequantize)
ASYNC_NIF(quantize)

// fast_* and kv_cache_* NIFs are defined in emlx_fast.cpp.

// Forward declarations for the async wrappers defined in emlx_fast.cpp.
ERL_NIF_TERM fast_rms_norm_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_rope_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_sdpa_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_sdpa_masked_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_layer_norm_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_layer_norm_no_bias_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_rope_ids_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_rope_with_freqs_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_rope_positions_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_sdpa_causal_key_masked_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_sdpa_causal_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM fast_swiglu_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM kv_cache_attention_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM kv_cache_attention_masked_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
ERL_NIF_TERM kv_cache_sdpa_update_async(ErlNifEnv *, int, const ERL_NIF_TERM []);
// ─── Async wrappers ────────────────────────────────────────────────────────

// Build a sliding window view of a padded tensor.
// padded: [...] of ndim n; window/strides: per-axis lists of length n.
// Returns a view of shape [o0,...,on-1, w0,...,wn-1] where
// oi = (padded_shape[i] - window[i]) / strides[i] + 1.
static mlx::core::array sliding_window_view_cpp(
    const mlx::core::array &padded,
    const std::vector<int> &window,
    const std::vector<int> &strides,
    const mlx::core::Device &device) {
  int n = padded.ndim();
  auto ps = padded.shape();  // SmallVector<int>

  // Doubled element strides: output dims share the same strides as window dims.
  auto orig_strides = padded.strides();
  std::vector<int64_t> view_strides(orig_strides.begin(), orig_strides.end());
  for (auto s : orig_strides) view_strides.push_back(s);

  // view_shape = [ps[i]-window[i]+1, ..., w0, ..., wn-1]
  std::vector<int> view_shape;
  for (int i = 0; i < n; ++i) view_shape.push_back(ps[i] - window[i] + 1);
  for (int w : window) view_shape.push_back(w);

  auto strided = mlx::core::as_strided(padded, to_shape(view_shape),
                                       to_strides(view_strides), 0, device);

  // Slice: strides=[strides..., 1...], stops=view_shape
  std::vector<int> starts(2 * n, 0);
  std::vector<int> stops = view_shape;
  std::vector<int> slstrides = strides;
  for (int i = 0; i < n; ++i) slstrides.push_back(1);

  return mlx::core::slice(strided, to_shape(starts), to_shape(stops),
                          to_shape(slstrides), device);
}

// Shared implementation for window_scatter_max/min.
// When scatter_max=true: first-occurrence argmax.
// When scatter_max=false: last-occurrence argmin via mask*arange trick.
static mlx::core::array window_scatter_impl(
    const mlx::core::array &tensor_t,
    const mlx::core::array &tensor_source,
    const mlx::core::array &tensor_init_value,
    const std::vector<int> &window,
    const std::vector<int> &low_pad,
    const std::vector<int> &high_pad,
    const std::vector<int> &strides,
    bool scatter_max,
    const mlx::core::Device &device) {
  int n = tensor_t.ndim();

  // 1. Cast init_value to the input dtype.
  auto init_casted =
      mlx::core::astype(tensor_init_value, tensor_t.dtype(), device);

  // 2. Pad input with init_value on all axes.
  std::vector<int> all_axes(n);
  std::iota(all_axes.begin(), all_axes.end(), 0);
  auto padded =
      mlx::core::pad(tensor_t, all_axes, to_shape(low_pad), to_shape(high_pad),
                     init_casted, "constant", device);

  auto padded_shape = padded.shape();
  std::vector<int> padded_shape_vec(padded_shape.begin(), padded_shape.end());

  // 3. Sliding window view: [o0,...,on-1, w0,...,wn-1].
  auto window_view =
      sliding_window_view_cpp(padded, window, strides, device);

  // out_shape = first n dims of window_view
  std::vector<int> out_shape(window_view.shape().begin(),
                             window_view.shape().begin() + n);

  // K = product of window dims
  int K = 1;
  for (int w : window) K *= w;

  // 4. Flatten window dims: [..., K]
  std::vector<int> flat_shape = out_shape;
  flat_shape.push_back(K);
  auto windows_flat =
      mlx::core::reshape(window_view, to_shape(flat_shape), device);

  // 5. Find argmax / tie-broken argmin over last axis.
  auto arg_idx = [&]() -> mlx::core::array {
    if (scatter_max) {
      return mlx::core::argmax(windows_flat, n, false, device);
    }
    // Tie-broken argmin (last-occurrence):
    // m = min over last axis (keepdims), mask where equal, argmax(mask*arange).
    auto m = mlx::core::min(windows_flat, std::vector<int>{n}, true, device);
    auto mask = mlx::core::astype(
        mlx::core::equal(windows_flat, m, device), mlx::core::uint32, device);
    auto arange_k = mlx::core::astype(
        mlx::core::arange(0, K, 1, device), mlx::core::uint32, device);
    std::vector<int> arange_shape(n + 1, 1);
    arange_shape[n] = K;
    auto arange_k_nd =
        mlx::core::reshape(arange_k, to_shape(arange_shape), device);
    auto weighted = mlx::core::multiply(mask, arange_k_nd, device);
    return mlx::core::argmax(weighted, n, false, device);
  }();

  // 6. Expand arg_idx to [..., 1] for take_along_axis.
  std::vector<int> arg_exp_shape = out_shape;
  arg_exp_shape.push_back(1);
  auto arg_idx_exp =
      mlx::core::reshape(arg_idx, to_shape(arg_exp_shape), device);

  // 7. For each axis, compute absolute padded-tensor indices.
  std::vector<mlx::core::array> abs_indices;
  for (int a = 0; a < n; ++a) {
    // 1-D iota along axis a of the padded shape.
    auto arange_a = mlx::core::astype(
        mlx::core::arange(0, (int)padded_shape[a], 1, device),
        mlx::core::int32, device);

    // Reshape to [1,...,padded_shape[a],...,1] (size pd[a] at axis a).
    std::vector<int> iota_shape(n, 1);
    iota_shape[a] = (int)padded_shape[a];
    auto iota_nd =
        mlx::core::reshape(arange_a, to_shape(iota_shape), device);

    // Broadcast to full padded shape.
    // NOTE: assumes padded is contiguous (as returned by mlx::pad), so its
    // element strides are dense. The doubled-strides trick in
    // sliding_window_view_cpp works correctly on broadcast_to's zero strides:
    // for axis-a iota, the zero stride on non-a dims keeps the value constant,
    // which is exactly the intended iota semantics.
    auto iota_bc =
        mlx::core::broadcast_to(iota_nd, to_shape(padded_shape_vec), device);

    // Apply same sliding-window view + flatten.
    auto iota_view =
        sliding_window_view_cpp(iota_bc, window, strides, device);
    auto iota_flat =
        mlx::core::reshape(iota_view, to_shape(flat_shape), device);

    // Pick the element at arg_idx position.
    auto abs_a =
        mlx::core::take_along_axis(iota_flat, arg_idx_exp, n, device);
    // Squeeze last dim: [..., 1] → [o0,...,on-1]
    abs_indices.push_back(
        mlx::core::reshape(abs_a, to_shape(out_shape), device));
  }

  // 8. Scatter source into a buffer filled with init_value.
  //    MLX scatter_add requires: updates.ndim == array.ndim + indices[0].ndim.
  //    array.ndim = n (padded), indices[0].ndim = n (out_shape), so we need 2n.
  //    Reshape source [o0,...,on-1] → [o0,...,on-1, 1,...,1] (n trailing singletons).
  auto source_shape_2n = std::vector<int>(tensor_source.shape().begin(),
                                          tensor_source.shape().end());
  for (int i = 0; i < n; ++i) source_shape_2n.push_back(1);
  auto updates =
      mlx::core::reshape(tensor_source, to_shape(source_shape_2n), device);

  auto buffer = mlx::core::broadcast_to(
      mlx::core::reshape(init_casted, to_shape(std::vector<int>{}), device),
      to_shape(padded_shape_vec), device);

  std::vector<int> scatter_axes(n);
  std::iota(scatter_axes.begin(), scatter_axes.end(), 0);
  auto scattered = mlx::core::scatter_add(buffer, abs_indices, updates,
                                          scatter_axes, device);

  // 9. Slice back to original shape (strip padding).
  auto orig_shape = tensor_t.shape();
  std::vector<int> slice_starts = low_pad;
  std::vector<int> slice_stops(n);
  for (int i = 0; i < n; ++i)
    slice_stops[i] = low_pad[i] + (int)orig_shape[i];
  std::vector<int> slice_ones(n, 1);

  return mlx::core::slice(scattered, to_shape(slice_starts),
                          to_shape(slice_stops), to_shape(slice_ones), device);
}

NIF(window_scatter_max) {
  TENSOR_PARAM(0, tensor_t);
  TENSOR_PARAM(1, tensor_source);
  TENSOR_PARAM(2, tensor_init_value);
  LIST_PARAM(3, std::vector<int>, window);
  LIST_PARAM(4, std::vector<int>, low_pad);
  LIST_PARAM(5, std::vector<int>, high_pad);
  LIST_PARAM(6, std::vector<int>, strides);
  DEVICE_PARAM(7, device);

  TENSOR(window_scatter_impl(*tensor_t, *tensor_source, *tensor_init_value,
                             window, low_pad, high_pad, strides, true, device));
}

NIF(window_scatter_min) {
  TENSOR_PARAM(0, tensor_t);
  TENSOR_PARAM(1, tensor_source);
  TENSOR_PARAM(2, tensor_init_value);
  LIST_PARAM(3, std::vector<int>, window);
  LIST_PARAM(4, std::vector<int>, low_pad);
  LIST_PARAM(5, std::vector<int>, high_pad);
  LIST_PARAM(6, std::vector<int>, strides);
  DEVICE_PARAM(7, device);

  TENSOR(window_scatter_impl(*tensor_t, *tensor_source, *tensor_init_value,
                             window, low_pad, high_pad, strides, false,
                             device));
}

// ─── Async wrappers ────────────────────────────────────────────────────────
//
// MLX 0.31.2's thread-local Metal CommandEncoders + thread-local default
// streams force every graph-touching op for a given GPU stream to run on
// the OS thread that created the stream. Each NIF below is the existing
// sync body wrapped by `emlx::async_dispatch<sync>`: the wrapper extracts
// the worker from `argv[0]`, copies `argv[1..]` into a process-independent
// `msg_env`, posts a lambda to the worker thread, and `enif_send`s
// `{job_ref, payload}` back to the caller. The sync body runs unchanged
// on the worker thread; `DEVICE_PARAM` resolutions transparently pick up
// the worker's stream via MLX's `to_stream(s, default_) -> default_stream`
// thread-local lookup (the worker called `set_default_stream` for its
// stream during `thread_main`).
//
// Sync (non-routed) NIFs preserved below. These either touch no MLX graph state
// or read fields that are safe across threads (resource allocator, cache
// limits, evaluated buffer pointers).

ASYNC_NIF(eval)
ASYNC_NIF(eval_many)
ASYNC_NIF(to_blob)
ASYNC_NIF(tensor_to_shm)
ASYNC_NIF(item)
ASYNC_NIF(from_blob)
ASYNC_NIF(scalar_tensor)

ASYNC_NIF(ones)
ASYNC_NIF(full)
ASYNC_NIF(arange)
ASYNC_NIF(eye)
ASYNC_NIF(reshape)
ASYNC_NIF(copy)
ASYNC_NIF(astype)
ASYNC_NIF(view)
ASYNC_NIF(broadcast_to)
ASYNC_NIF(transpose)
ASYNC_NIF(pad)
ASYNC_NIF(sort)
ASYNC_NIF(argsort)
ASYNC_NIF(slice)
ASYNC_NIF(slice_update)
ASYNC_NIF(squeeze)
ASYNC_NIF(as_strided)

ASYNC_NIF(stack)
ASYNC_NIF(where)
ASYNC_NIF(concatenate)
ASYNC_NIF(take_along_axis)
ASYNC_NIF(take)
ASYNC_NIF(gather)
ASYNC_NIF(scatter_add)
ASYNC_NIF(scatter)

ASYNC_NIF(all)
ASYNC_NIF(any)
ASYNC_NIF(sum)
ASYNC_NIF(product)
ASYNC_NIF(argmax)
ASYNC_NIF(argmin)
ASYNC_NIF(cumulative_sum)
ASYNC_NIF(cumulative_product)
ASYNC_NIF(cumulative_max)
ASYNC_NIF(cumulative_min)
ASYNC_NIF(max)
ASYNC_NIF(min)
ASYNC_NIF(clip)

ASYNC_NIF(abs)
ASYNC_NIF(ceil)
ASYNC_NIF(conjugate)
ASYNC_NIF(floor)
ASYNC_NIF(negate)
ASYNC_NIF(round)
ASYNC_NIF(sign)
ASYNC_NIF(real)
ASYNC_NIF(imag)
ASYNC_NIF(is_nan)
ASYNC_NIF(is_infinity)
ASYNC_NIF(logical_not)
ASYNC_NIF(sigmoid)
ASYNC_NIF(asin)
ASYNC_NIF(asinh)
ASYNC_NIF(acos)
ASYNC_NIF(acosh)
ASYNC_NIF(cos)
ASYNC_NIF(cosh)
ASYNC_NIF(atan)
ASYNC_NIF(atanh)
ASYNC_NIF(erf)
ASYNC_NIF(erf_inv)
ASYNC_NIF(exp)
ASYNC_NIF(expm1)
ASYNC_NIF(log)
ASYNC_NIF(log1p)
ASYNC_NIF(rsqrt)
ASYNC_NIF(sin)
ASYNC_NIF(sinh)
ASYNC_NIF(sqrt)
ASYNC_NIF(tan)
ASYNC_NIF(tanh)

ASYNC_NIF(add)
ASYNC_NIF(subtract)
ASYNC_NIF(multiply)
ASYNC_NIF(pow)
ASYNC_NIF(remainder)
ASYNC_NIF(divide)
ASYNC_NIF(atan2)
ASYNC_NIF(bitwise_and)
ASYNC_NIF(bitwise_or)
ASYNC_NIF(bitwise_xor)
ASYNC_NIF(bitwise_not)
ASYNC_NIF(left_shift)
ASYNC_NIF(right_shift)
ASYNC_NIF(minimum)
ASYNC_NIF(maximum)
ASYNC_NIF(quotient)
ASYNC_NIF(equal)
ASYNC_NIF(not_equal)
ASYNC_NIF(greater)
ASYNC_NIF(less)
ASYNC_NIF(greater_equal)
ASYNC_NIF(less_equal)
ASYNC_NIF(logical_and)
ASYNC_NIF(logical_or)
ASYNC_NIF(logical_xor)

ASYNC_NIF(emlx_fft)
ASYNC_NIF(ifft)
ASYNC_NIF(emlx_fft2)
ASYNC_NIF(ifft2)
ASYNC_NIF(allclose)
ASYNC_NIF(isclose)
ASYNC_NIF(tri_inv)

ASYNC_NIF(linalg_lu)
ASYNC_NIF(linalg_qr)
ASYNC_NIF(linalg_svd)
ASYNC_NIF(linalg_cholesky)
ASYNC_NIF(linalg_eigh)
ASYNC_NIF(linalg_inv)
ASYNC_NIF(linalg_pinv)
ASYNC_NIF(linalg_solve)
ASYNC_NIF(linalg_solve_triangular)
ASYNC_NIF(conv_general)
ASYNC_NIF(einsum)
ASYNC_NIF(tensordot)

ASYNC_NIF(window_scatter_max)
ASYNC_NIF(window_scatter_min)

// ── Native compiler NIFs (logic lives in emlx/compiler.cpp) ──────────────────
//
// compile_program is defined directly in emlx/compiler.cpp via
// FINE_ASYNC_NIF(compile_program) (see emlx/compiler.hpp for the
// compile_program/compile_program_async declarations); referenced fully
// qualified in nif_funcs[] below, same as resolve_runtime_call.

NIF(eval_program) { return emlx::native::eval_program(env, argc, argv); }
ASYNC_NIF(eval_program)

static ErlNifFunc nif_funcs[] = {
    // No dirty-scheduler flag: eval/eval_many post to a dedicated Worker
    // OS thread via async_dispatch (emlx_async.hpp) and return immediately —
    // the calling BEAM scheduler never blocks on the actual MLX work, same
    // as the ~150 sibling ASYNC_NIF ops below (item 3.8 fold-in).
    {"eval", 2, eval_async},
    {"eval_many", 2, eval_many_async},
    {"to_device", 3, to_device_async},
    {"to_blob", 2, to_blob_async},
    {"to_blob", 3, to_blob_async},
    {"tensor_to_shm", 3, tensor_to_shm_async},
    {"item", 2, item_async},
    {"from_blob", 5, from_blob_async},
    {"scalar_tensor", 4, scalar_tensor_async},

    {"ones", 4, ones_async},
    {"full", 5, full_async},
    {"arange", 6, arange_async},
    {"eye", 5, eye_async},
    {"reshape", 4, reshape_async},
    {"copy", 3, copy_async},
    {"astype", 4, astype_async},
    {"view", 4, view_async},
    {"broadcast_to", 4, broadcast_to_async},
    {"transpose", 4, transpose_async},
    {"pad", 7, pad_async},
    {"sort", 4, sort_async},
    {"argsort", 4, argsort_async},
    {"slice", 6, slice_async},
    {"slice_update", 6, slice_update_async},
    {"squeeze", 4, squeeze_async},
    {"as_strided", 6, as_strided_async},

    {"stack", 4, stack_async},
    {"where", 5, where_async},
    {"concatenate", 4, concatenate_async},
    {"take_along_axis", 5, take_along_axis_async},
    {"take", 5, take_async},
    {"gather", 6, gather_async},
    {"scatter_add", 6, scatter_add_async},
    {"scatter", 6, scatter_async},

    {"all", 5, all_async},
    {"any", 5, any_async},
    {"sum", 5, sum_async},
    {"product", 5, product_async},
    {"argmax", 4, argmax_async},
    {"argmax", 5, argmax_async},
    {"argmin", 4, argmin_async},
    {"argmin", 5, argmin_async},
    {"cumulative_sum", 6, cumulative_sum_async},
    {"cumulative_product", 6, cumulative_product_async},
    {"cumulative_max", 6, cumulative_max_async},
    {"cumulative_min", 6, cumulative_min_async},
    {"max", 5, max_async},
    {"min", 5, min_async},
    {"clip", 5, clip_async},

    {"abs", 3, abs_async},
    {"ceil", 3, ceil_async},
    {"conjugate", 3, conjugate_async},
    {"floor", 3, floor_async},
    {"negate", 3, negate_async},
    {"round", 3, round_async},
    {"sign", 3, sign_async},
    {"real", 3, real_async},
    {"imag", 3, imag_async},
    {"is_nan", 3, is_nan_async},
    {"is_infinity", 3, is_infinity_async},
    {"logical_not", 3, logical_not_async},
    {"sigmoid", 3, sigmoid_async},
    {"asin", 3, asin_async},
    {"asinh", 3, asinh_async},
    {"acos", 3, acos_async},
    {"acosh", 3, acosh_async},
    {"cos", 3, cos_async},
    {"cosh", 3, cosh_async},
    {"atan", 3, atan_async},
    {"atanh", 3, atanh_async},
    {"erf", 3, erf_async},
    {"erf_inv", 3, erf_inv_async},
    {"exp", 3, exp_async},
    {"expm1", 3, expm1_async},
    {"log", 3, log_async},
    {"log1p", 3, log1p_async},
    {"rsqrt", 3, rsqrt_async},
    {"sin", 3, sin_async},
    {"sinh", 3, sinh_async},
    {"sqrt", 3, sqrt_async},
    {"tan", 3, tan_async},
    {"tanh", 3, tanh_async},

    {"add", 4, add_async},
    {"subtract", 4, subtract_async},
    {"multiply", 4, multiply_async},
    {"pow", 4, pow_async},
    {"remainder", 4, remainder_async},
    {"divide", 4, divide_async},
    {"atan2", 4, atan2_async},
    {"bitwise_and", 4, bitwise_and_async},
    {"bitwise_or", 4, bitwise_or_async},
    {"bitwise_xor", 4, bitwise_xor_async},
    {"bitwise_not", 3, bitwise_not_async},
    {"left_shift", 4, left_shift_async},
    {"right_shift", 4, right_shift_async},
    {"minimum", 4, minimum_async},
    {"maximum", 4, maximum_async},
    {"quotient", 4, quotient_async},
    {"equal", 4, equal_async},
    {"not_equal", 4, not_equal_async},
    {"greater", 4, greater_async},
    {"less", 4, less_async},
    {"greater_equal", 4, greater_equal_async},
    {"less_equal", 4, less_equal_async},
    {"logical_and", 4, logical_and_async},
    {"logical_or", 4, logical_or_async},
    {"logical_xor", 4, logical_xor_async},

    {"fft", 5, emlx_fft_async},
    {"ifft", 5, ifft_async},
    {"fft2", 5, emlx_fft2_async},
    {"ifft2", 5, ifft2_async},
    {"allclose", 7, allclose_async},
    {"isclose", 7, isclose_async},
    {"tri_inv", 4, tri_inv_async},

    {"linalg_lu", 3, linalg_lu_async},
    {"linalg_qr", 3, linalg_qr_async},
    {"linalg_svd", 4, linalg_svd_async},
    {"linalg_cholesky", 4, linalg_cholesky_async},
    {"linalg_eigh", 4, linalg_eigh_async},
    {"linalg_inv", 3, linalg_inv_async},
    {"linalg_pinv", 3, linalg_pinv_async},
    {"linalg_solve", 4, linalg_solve_async},
    {"linalg_solve_triangular", 5, linalg_solve_triangular_async},
    {"conv_general", 10, conv_general_async},
    {"einsum", 4, einsum_async},
    {"tensordot", 6, tensordot_async},

    {"window_scatter_max", 9, window_scatter_max_async},
    {"window_scatter_min", 9, window_scatter_min_async},

    // ── Sync (non-routed) NIFs.
    {"strides", 1, strides},
    {"scalar_type", 1, scalar_type},
    {"shape", 1, shape},
    {"deallocate", 1, deallocate},
    {"array_from_shm", 4, array_from_shm, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"shm_unlink_handle", 1, shm_unlink_handle},
    {"tensor_data_ptr", 1, tensor_data_ptr, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"array_from_ptr", 5, array_from_ptr, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"memory_info", 0, memory_info},
    {"clear_cache", 0, clear_cache},
    {"reset_peak_memory", 0, reset_peak_memory},
    {"set_memory_limit", 1, set_memory_limit},
    {"set_cache_limit", 1, set_cache_limit},
    {"metal_start_capture", 1, metal_start_capture},
    {"metal_stop_capture", 0, metal_stop_capture},

    // ── Worker control NIFs.
    {"command_queue_new", 1, command_queue_new},
    {"command_queue_synchronize", 1, command_queue_synchronize},
    // Quantization operations (async — must run on a worker thread)
    {"quantized_matmul", 10, quantized_matmul_async},
    {"dequantize", 8, dequantize_async},
    {"quantize", 6, quantize_async},

    // mlx::fast ops (worker arity includes queue ref as argv[0])
    {"fast_rms_norm", 5, fast_rms_norm_async},
    {"fast_rope", 8, fast_rope_async},
    {"fast_sdpa", 7, fast_sdpa_async},
    {"fast_sdpa_masked", 8, fast_sdpa_masked_async},
    {"fast_rope_ids", 8, fast_rope_ids_async},
    {"fast_rope_with_freqs", 8, fast_rope_with_freqs_async},
    {"fast_rope_positions", 8, fast_rope_positions_async},
    {"fast_sdpa_causal_key_masked", 9, fast_sdpa_causal_key_masked_async},
    {"fast_sdpa_causal", 7, fast_sdpa_causal_async},
    {"fast_layer_norm", 6, fast_layer_norm_async},
    {"fast_layer_norm_no_bias", 5, fast_layer_norm_no_bias_async},
    {"fast_swiglu", 4, fast_swiglu_async},
    {"kv_cache_attention", 9, kv_cache_attention_async},
    {"kv_cache_attention_masked", 10, kv_cache_attention_masked_async},
    {"kv_cache_sdpa_update", 9, kv_cache_sdpa_update_async},

    // ── Native compiler NIFs.
    // No dirty-scheduler flag on either: both are FINE_ASYNC_NIF/ASYNC_NIF-
    // wrapped (post to a Worker OS thread, return immediately) — same
    // reasoning as eval/eval_many above (item 3.8 fold-in).
    {"compile_program", 2, emlx::native::compile_program_async},
    {"eval_program", 3, eval_program_async},
    // resolve_runtime_call is NOT worker-routed: it only decodes a reply and
    // memcpy's it into pre-registered buffers/notifies a condvar — no MLX
    // graph work, so it can run directly on the calling BEAM scheduler.
    {"resolve_runtime_call", 3, emlx::native::resolve_runtime_call},

    // load_plugin `dlopen`s a named, standalone native plugin (see
    // emlx/plugin/registry.hpp); not worker-routed since it does no MLX
    // graph work.
    {"load_plugin", 2, load_plugin},
    {"load_plugin", 3, load_plugin_with_build_id},
    {"call_plugin", 6, call_plugin_async}};

ERL_NIF_INIT(Elixir.EMLX.NIF, nif_funcs, load, NULL, upgrade, NULL)
