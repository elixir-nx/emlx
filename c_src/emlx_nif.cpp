#include "erl_nif.h"
#include "mlx/mlx.h"
#include "nx_nif_utils.hpp"

#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <cstring>

using namespace mlx::core;

std::map<const std::string, const mlx::core::Dtype> dtypes = {
    {"bool", mlx::core::bool_},         {"uint8", mlx::core::uint8},
    {"uint16", mlx::core::uint16},      {"uint32", mlx::core::uint32},
    {"uint64", mlx::core::uint64},      {"int8", mlx::core::int8},
    {"int16", mlx::core::int16},        {"int32", mlx::core::int32},
    {"int64", mlx::core::int64},        {"float16", mlx::core::float16},
    {"float32", mlx::core::float32},    {"bfloat16", mlx::core::bfloat16},
    {"complex64", mlx::core::complex64}};

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

inline mlx::core::Dtype string2dtype(const std::string &atom) {
  auto it = dtypes.find(atom);
  if (it != dtypes.end()) {
    return it->second;
  }
  throw std::runtime_error("Unknown dtype: " + atom);
}

inline const std::string *dtype2string(const mlx::core::Dtype dtype) {
  for (const auto &pair : dtypes) {
    if (pair.second == dtype) {
      return &pair.first;
    }
  }
  return nullptr;
}

inline const mlx::core::Device string2device(const std::string &atom) {
  if (atom == "cpu") {
    return mlx::core::Device(mlx::core::Device::DeviceType::cpu, 0);
  } else if (atom == "gpu") {
    return mlx::core::Device(mlx::core::Device::DeviceType::gpu, 0);
  }
  throw std::runtime_error("Unknown device: " + atom);
}

// MLX 0.31+ uses Shape = SmallVector<int> and Strides = SmallVector<long long>
// which no longer accept implicit construction from std::vector.
static inline mlx::core::Shape to_shape(const std::vector<int> &v) {
  return mlx::core::Shape(v.begin(), v.end());
}
static inline mlx::core::Strides to_strides(const std::vector<int64_t> &v) {
  return mlx::core::Strides(v.begin(), v.end());
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

ERL_NIF_TERM create_function_resource(ErlNifEnv *env, emlx::function function) {
  ERL_NIF_TERM ret;
  std::atomic<int> *refcount;
  auto function_ptr = (emlx::function *)enif_alloc_resource(
      resource_object<emlx::function>::type,
      sizeof(std::function<std::vector<array>(const std::vector<array> &)>) +
          sizeof(std::atomic<int>) + sizeof(std::atomic_flag));

  if (function_ptr == NULL) {
    return enif_make_badarg(env);
  }

  new (function_ptr) emlx::function(function);
  refcount = new (function_ptr + 1) std::atomic<int>(1);
  new (refcount + 1) std::atomic_flag();

  ret = enif_make_resource(env, function_ptr);
  enif_release_resource(function_ptr);

  return ret;
}

#define NIF(NAME)                                                              \
  ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

#define PARAM(ARGN, TYPE, VAR)                                                 \
  TYPE VAR;                                                                    \
  GET(ARGN, VAR)

#define TENSOR_PARAM(ARGN, VAR)                                                \
  TensorP VAR##_tp(env, argv[ARGN]);                                           \
  mlx::core::array *VAR;                                                       \
  if (!VAR##_tp.is_valid()) {                                                  \
    return VAR##_tp.error();                                                   \
  } else {                                                                     \
    VAR = VAR##_tp.data();                                                     \
  }

#define LIST_PARAM(ARGN, TYPE, VAR)                                            \
  TYPE VAR;                                                                    \
  if (!nx::nif::get_list(env, argv[ARGN], VAR))                                \
    return nx::nif::error(env, "Unable to get " #VAR " list param.");

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

NIF(zeros) {
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::zeros(to_shape(shape), type, device));
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

NIF(to_blob) {
  TENSOR_PARAM(0, t);

  size_t byte_size = t->nbytes();
  if (argc == 2) {
    PARAM(1, int, param_limit);
    byte_size = static_cast<size_t>(param_limit) * t->itemsize();
  }

  ERL_NIF_TERM resource_bin;

  if (t->flags().row_contiguous) {
    // Zero-copy: alias the MLX buffer via the existing tensor resource.
    // Invariant: lib/emlx.ex calls eval(tensor) before this NIF, so
    // data<void>() is guaranteed non-null and stable (MLX arrays are immutable
    // once materialised). enif_make_resource_binary keeps the resource alive
    // until the binary is GC'd, decoupling the binary lifetime from Elixir GC
    // of the tensor term.
    resource_bin = enif_make_resource_binary(env, t_tp.resource_ptr(),
                                             t->data<void>(), byte_size);
  } else {
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
    if (!ct_ptr)
      return enif_make_badarg(env);

    new (ct_ptr) mlx::core::array(std::move(ct));
    resource_bin =
        enif_make_resource_binary(env, ct_ptr, ct_ptr->data<void>(), byte_size);
    enif_release_resource(ct_ptr);
  }
  return nx::nif::ok(env, resource_bin);
}

uint64_t elem_count(std::vector<int> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
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
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);

  std::string spec_string;
  if (!nx::nif::get(env, argv[2], spec_string)) {
    return nx::nif::error(env, "Unable to get spec_string param.");
  }

  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::einsum(spec_string, std::vector<mlx::core::array>({*a, *b}),
                           device));
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

NIF(linalg_svd) {
  TENSOR_PARAM(0, tensor);
  PARAM(1, bool, compute_uv);
  DEVICE_PARAM(2, device);

  try {
    auto result = mlx::core::linalg::svd(*tensor, compute_uv, device);
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

  try {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eigh(*tensor, uplo, device);
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
  return nx::nif::ok(env);
}

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

static int open_resources(ErlNifEnv *env) {
  const char *mod = "EMLX";
  if (!open_resource<mlx::core::array>(env, mod, "MLXArray")) {
    return -1;
  }

  if (!open_resource<emlx::function>(env, mod, "CompiledFunction")) {
    return -1;
  }

  return 0;
}

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  if (open_resources(env) != 0) {
    return -1;
  }

  return 0;
}

int upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data, ERL_NIF_TERM load_info) {
  // Silence "unused var" warnings.
  (void)(env);
  (void)(priv_data);
  (void)(old_priv_data);
  (void)(load_info);

  return 0;
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

static ErlNifFunc nif_funcs[] = {
    {"strides", 1, strides},
    {"as_strided", 5, as_strided},
    {"scalar_type", 1, scalar_type},
    {"eval", 1, eval},
    {"view", 3, view},
    {"stack", 3, stack},
    {"where", 4, where},
    {"concatenate", 3, concatenate},
    {"take_along_axis", 4, take_along_axis},
    {"take", 4, take},
    {"gather", 5, gather},
    {"scatter_add", 5, scatter_add},
    {"scatter", 5, scatter},
    {"slice", 5, slice},
    {"slice_update", 5, slice_update},
    {"squeeze", 3, squeeze},
    {"item", 1, item, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"all", 4, all},
    {"any", 4, any},
    {"sum", 4, sum},
    {"product", 4, product},
    {"argmax", 3, argmax},
    {"argmax", 4, argmax},
    {"argmin", 3, argmin},
    {"argmin", 4, argmin},
    {"cumulative_sum", 5, cumulative_sum},
    {"cumulative_product", 5, cumulative_product},
    {"cumulative_max", 5, cumulative_max},
    {"cumulative_min", 5, cumulative_min},
    {"shape", 1, shape},
    {"reshape", 3, reshape},
    {"astype", 3, astype},
    {"to_blob", 1, to_blob, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"to_blob", 2, to_blob, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"from_blob", 4, from_blob},
    {"scalar_tensor", 3, scalar_tensor},
    {"ones", 3, ones},
    {"full", 4, full},
    {"arange", 5, arange},
    {"eye", 4, eye},
    {"broadcast_to", 3, broadcast_to},
    {"tensordot", 5, tensordot},
    {"einsum", 4, einsum},
    {"conv_general", 9, conv_general},
    {"transpose", 3, transpose},
    {"pad", 6, pad},
    {"sort", 3, sort},
    {"argsort", 3, argsort},
    {"abs", 2, abs},
    {"ceil", 2, ceil},
    {"conjugate", 2, conjugate},
    {"floor", 2, floor},
    {"negate", 2, negate},
    {"round", 2, round},
    {"sign", 2, sign},
    {"real", 2, real},
    {"imag", 2, imag},
    {"is_nan", 2, is_nan},
    {"is_infinity", 2, is_infinity},
    {"logical_not", 2, logical_not},
    {"sigmoid", 2, sigmoid},
    {"asin", 2, asin},
    {"asinh", 2, asinh},
    {"acos", 2, acos},
    {"acosh", 2, acosh},
    {"cos", 2, cos},
    {"cosh", 2, cosh},
    {"atan", 2, atan},
    {"atanh", 2, atanh},
    {"erf", 2, erf},
    {"erf_inv", 2, erf_inv},
    {"exp", 2, exp},
    {"expm1", 2, expm1},
    {"log", 2, log},
    {"log1p", 2, log1p},
    {"rsqrt", 2, rsqrt},
    {"sin", 2, sin},
    {"sinh", 2, sinh},
    {"sqrt", 2, sqrt},
    {"tan", 2, tan},
    {"tanh", 2, tanh},
    {"add", 3, add},
    {"subtract", 3, subtract},
    {"multiply", 3, multiply},
    {"pow", 3, pow},
    {"remainder", 3, remainder},
    {"divide", 3, divide},
    {"atan2", 3, atan2},
    {"bitwise_and", 3, bitwise_and},
    {"bitwise_or", 3, bitwise_or},
    {"bitwise_xor", 3, bitwise_xor},
    {"bitwise_not", 2, bitwise_not},
    {"left_shift", 3, left_shift},
    {"right_shift", 3, right_shift},
    {"minimum", 3, minimum},
    {"maximum", 3, maximum},
    {"quotient", 3, quotient},
    {"equal", 3, equal},
    {"not_equal", 3, not_equal},
    {"greater", 3, greater},
    {"less", 3, less},
    {"greater_equal", 3, greater_equal},
    {"less_equal", 3, less_equal},
    {"logical_and", 3, logical_and},
    {"logical_or", 3, logical_or},
    {"logical_xor", 3, logical_xor},
    {"fft", 4, emlx_fft},
    {"ifft", 4, ifft},
    {"fft2", 4, emlx_fft2},
    {"ifft2", 4, ifft2},
    {"allclose", 6, allclose},
    {"isclose", 6, isclose},
    {"deallocate", 1, deallocate},
    {"max", 4, max},
    {"min", 4, min},
    {"clip", 4, clip},
    {"tri_inv", 3, tri_inv},
    {"linalg_lu", 2, linalg_lu},
    {"linalg_qr", 2, linalg_qr},
    {"linalg_svd", 3, linalg_svd},
    {"linalg_cholesky", 3, linalg_cholesky},
    {"linalg_eigh", 3, linalg_eigh},
    {"linalg_inv", 2, linalg_inv},
    {"linalg_pinv", 2, linalg_pinv},
    {"linalg_solve", 3, linalg_solve},
    {"linalg_solve_triangular", 4, linalg_solve_triangular},
    {"window_scatter_max", 8, window_scatter_max},
    {"window_scatter_min", 8, window_scatter_min},
    {"memory_info", 0, memory_info},
    {"clear_cache", 0, clear_cache},
    {"reset_peak_memory", 0, reset_peak_memory},
    {"set_memory_limit", 1, set_memory_limit},
    {"set_cache_limit", 1, set_cache_limit}
};

// Update the NIF initialization
ERL_NIF_INIT(Elixir.EMLX.NIF, nif_funcs, load, NULL, upgrade, NULL)
