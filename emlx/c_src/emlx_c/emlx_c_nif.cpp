// EMLX mlx-c lane: tensor ops implemented with the official mlx-c API
// (https://github.com/ml-explore/mlx-c). Binary in, evaluated binary out.
//
// This is the proposed replacement layer for direct mlx::core C++ calls in
// emlx NIFs. Built as its own .so with its own statically linked libmlx
// (via mlx-c's FetchContent pin) so it can run side-by-side with the
// existing C++ lane for differential testing and benchmarking:
//
//   libemlx_c.so   -> EMLX.C.NIF     (this file, mlx-c calls)
//   libemlx_ref.so -> EMLX.C.RefNIF  (direct mlx::core C++, reference)
//
// Copyright (c) 2026. MIT license (same as EMLX).

#include <erl_nif.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "mlx/c/mlx.h"

namespace {

struct Bin {
  unsigned char *data;
  size_t size;
};

const char *atom_c_str(ErlNifEnv *env, ERL_NIF_TERM term) {
  static char buf[64];
  if (!enif_get_atom(env, term, buf, sizeof(buf), ERL_NIF_LATIN1))
    return nullptr;
  return buf;
}

bool get_bin(ErlNifEnv *env, ERL_NIF_TERM term, Bin *out) {
  ErlNifBinary b;
  if (!enif_inspect_binary(env, term, &b))
    return false;
  out->data = b.data;
  out->size = b.size;
  return true;
}

bool get_int(ErlNifEnv *env, ERL_NIF_TERM term, int *out) {
  return enif_get_int(env, term, out) != 0;
}

bool get_int64(ErlNifEnv *env, ERL_NIF_TERM term, long *out) {
  return enif_get_long(env, term, out) != 0;
}

bool get_double(ErlNifEnv *env, ERL_NIF_TERM term, double *out) {
  return enif_get_double(env, term, out) != 0;
}

bool get_shape(ErlNifEnv *env, ERL_NIF_TERM list, std::vector<int> *out) {
  unsigned len;
  if (!enif_get_list_length(env, list, &len))
    return false;
  out->clear();
  ERL_NIF_TERM head, tail = list;
  while (!enif_is_empty_list(env, tail)) {
    if (!enif_get_list_cell(env, tail, &head, &tail))
      return false;
    int v;
    if (!get_int(env, head, &v))
      return false;
    out->push_back(v);
  }
  return out->size() == len;
}

ERL_NIF_TERM ok_bin(ErlNifEnv *env, const void *data, size_t size) {
  ErlNifBinary b;
  if (!enif_alloc_binary(size, &b))
    return enif_make_badarg(env);
  memcpy(b.data, data, size);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                          enif_make_binary(env, &b));
}

ERL_NIF_TERM err_msg(ErlNifEnv *env, const std::string &msg) {
  return enif_make_tuple2(env, enif_make_atom(env, "error"),
                          enif_make_string(env, msg.c_str(), ERL_NIF_LATIN1));
}

bool parse_dev(ErlNifEnv *env, ERL_NIF_TERM term, std::string *out) {
  const char *s = atom_c_str(env, term);
  if (!s)
    return false;
  *out = s;
  return *out == "cpu" || *out == "gpu";
}

#define CATCH                                                                  \
  catch (const std::exception & e) {                                         \
    return err_msg(env, e.what());                                           \
  }                                                                          \
  catch (...) { return err_msg(env, "unknown error"); }

mlx_dtype to_mlxc_dtype(const char *s) {
  if (!strcmp(s, "float32"))
    return MLX_FLOAT32;
  if (!strcmp(s, "float16"))
    return MLX_FLOAT16;
  if (!strcmp(s, "bfloat16"))
    return MLX_BFLOAT16;
  if (!strcmp(s, "int32"))
    return MLX_INT32;
  if (!strcmp(s, "int64"))
    return MLX_INT64;
  return MLX_BOOL; // invalid, callers validate
}

int device_stream(const std::string &d, mlx_device *dev, mlx_stream *stream) {
  *dev = mlx_device_new_type(d == "gpu" ? MLX_GPU : MLX_CPU, 0);
  *stream = mlx_stream_new();
  return mlx_get_default_stream(stream, *dev);
}

const void *data_ptr_of(mlx_array a) {
  switch (mlx_array_dtype(a)) {
    case MLX_FLOAT32:
      return (const void *)mlx_array_data_float32(a);
    case MLX_FLOAT16:
      return (const void *)mlx_array_data_float16(a);
    case MLX_BFLOAT16:
      return (const void *)mlx_array_data_bfloat16(a);
    case MLX_INT32:
      return (const void *)mlx_array_data_int32(a);
    case MLX_INT64:
      return (const void *)mlx_array_data_int64(a);
    default:
      return nullptr;
  }
}

ERL_NIF_TERM eval_out(ErlNifEnv *env, mlx_array a) {
  if (mlx_array_eval(a) != 0)
    return err_msg(env, "mlx_array_eval failed");
  const void *p = data_ptr_of(a);
  if (!p)
    return err_msg(env, "unsupported output dtype");
  return ok_bin(env, p, mlx_array_nbytes(a));
}

mlx_array new_arr(const Bin &b, const int *shape, int dim, const char *dt) {
  mlx_dtype d = to_mlxc_dtype(dt);
  if (d == MLX_BOOL)
    throw std::runtime_error("unsupported dtype");
  size_t n = 1;
  for (int i = 0; i < dim; i++)
    n *= (size_t)shape[i];
  size_t need = n * mlx_dtype_size(d);
  if (b.size < need)
    throw std::runtime_error("binary smaller than shape");
  return mlx_array_new_data(b.data, shape, dim, d);
}

// ───────────────────────────────── ops ──────────────────────────────────────

ERL_NIF_TERM nif_device_check(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  bool gpu_avail = false;
  mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
  mlx_device_is_available(&gpu_avail, gpu);
  mlx_device_free(gpu);
  mlx_string ver = mlx_string_new();
  mlx_version(&ver);
  const char *cstr = mlx_string_data(ver);
  ERL_NIF_TERM ret = enif_make_tuple3(
      env, enif_make_atom(env, "ok"), enif_make_atom(env, gpu_avail ? "gpu" : "cpu"),
      enif_make_string(env, cstr ? cstr : "?", ERL_NIF_LATIN1));
  mlx_string_free(ver);
  return ret;
}

ERL_NIF_TERM nif_copy(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin x;
    std::string dev;
    int n;
    if (!get_bin(env, argv[0], &x) || !parse_dev(env, argv[1], &dev) ||
        !get_int(env, argv[2], &n))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[1] = {n};
    mlx_array a = new_arr(x, shp, 1, atom_c_str(env, argv[3]));
    mlx_device_free(mxdev);
    ERL_NIF_TERM ret = eval_out(env, a);
    mlx_array_free(a);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_matmul(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a, b;
    std::string dev;
    int m, k, n;
    if (!get_bin(env, argv[0], &a) || !get_bin(env, argv[1], &b) ||
        !parse_dev(env, argv[3], &dev) || !get_int(env, argv[4], &m) ||
        !get_int(env, argv[5], &k) || !get_int(env, argv[6], &n))
      return enif_make_badarg(env);
    const char *dt = atom_c_str(env, argv[2]);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp_a[2] = {m, k}, shp_b[2] = {k, n};
    mlx_array xa = new_arr(a, shp_a, 2, dt);
    mlx_array yb = new_arr(b, shp_b, 2, dt);
    mlx_array res = mlx_array_empty;
    int rc = mlx_matmul(&res, xa, yb, stream);
    mlx_array_free(xa);
    mlx_array_free(yb);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_matmul failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_bmm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a, b;
    std::string dev;
    int bb, m, k, n;
    if (!get_bin(env, argv[0], &a) || !get_bin(env, argv[1], &b) ||
        !parse_dev(env, argv[3], &dev) || !get_int(env, argv[4], &bb) ||
        !get_int(env, argv[5], &m) || !get_int(env, argv[6], &k) ||
        !get_int(env, argv[7], &n))
      return enif_make_badarg(env);
    const char *dt = atom_c_str(env, argv[2]);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp_a[3] = {bb, m, k}, shp_b[3] = {bb, k, n};
    mlx_array xa = new_arr(a, shp_a, 3, dt);
    mlx_array yb = new_arr(b, shp_b, 3, dt);
    mlx_array res = mlx_array_empty;
    int rc = mlx_matmul(&res, xa, yb, stream);
    mlx_array_free(xa);
    mlx_array_free(yb);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_matmul (bmm) failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

#define BINARY_OP(NAME, MLXC)                                                \
  ERL_NIF_TERM nif_##NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) { \
    try {                                                                    \
      Bin a, b;                                                            \
      std::string dev;                                                     \
      int m, n;                                                            \
      if (!get_bin(env, argv[0], &a) || !get_bin(env, argv[1], &b) ||    \
          !parse_dev(env, argv[3], &dev) || !get_int(env, argv[4], &m) || \
          !get_int(env, argv[5], &n))                                      \
        return enif_make_badarg(env);                                      \
      const char *dt = atom_c_str(env, argv[2]);                           \
      mlx_device mxdev;                                                    \
      mlx_stream stream;                                                   \
      if (device_stream(dev, &mxdev, &stream)) {                           \
        mlx_device_free(mxdev);                                            \
        return err_msg(env, "device init failed");                         \
      }                                                                    \
      int shp_a[2] = {m, n}, shp_b[2] = {m, n};                          \
      mlx_array xa = new_arr(a, shp_a, 2, dt);                           \
      mlx_array yb = new_arr(b, shp_b, 2, dt);                           \
      mlx_array res = mlx_array_empty;                                     \
      int rc = MLXC(&res, xa, yb, stream);                               \
      mlx_array_free(xa);                                                  \
      mlx_array_free(yb);                                                  \
      mlx_stream_free(stream);                                             \
      mlx_device_free(mxdev);                                              \
      if (rc != 0)                                                         \
        return err_msg(env, #MLXC " failed");                            \
      ERL_NIF_TERM ret = eval_out(env, res);                             \
      mlx_array_free(res);                                                 \
      return ret;                                                          \
    }                                                                      \
    CATCH                                                                    \
  }

BINARY_OP(add, mlx_add)
BINARY_OP(subtract, mlx_subtract)
BINARY_OP(multiply, mlx_multiply)
BINARY_OP(divide, mlx_divide)

ERL_NIF_TERM nif_exp(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a;
    std::string dev;
    int m, n;
    if (!get_bin(env, argv[0], &a) || !parse_dev(env, argv[2], &dev) ||
        !get_int(env, argv[3], &m) || !get_int(env, argv[4], &n))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[2] = {m, n};
    mlx_array xa = new_arr(a, shp, 2, atom_c_str(env, argv[1]));
    mlx_array res = mlx_array_empty;
    int rc = mlx_exp(&res, xa, stream);
    mlx_array_free(xa);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_exp failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_softmax_axis(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a;
    std::string dev;
    int m, n, axis;
    if (!get_bin(env, argv[0], &a) || !parse_dev(env, argv[2], &dev) ||
        !get_int(env, argv[3], &m) || !get_int(env, argv[4], &n) ||
        !get_int(env, argv[5], &axis))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[2] = {m, n};
    mlx_array xa = new_arr(a, shp, 2, atom_c_str(env, argv[1]));
    mlx_array res = mlx_array_empty;
    int rc = mlx_softmax_axis(&res, xa, axis, false, stream);
    mlx_array_free(xa);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_softmax_axis failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_sum(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a;
    std::string dev;
    int m, n;
    if (!get_bin(env, argv[0], &a) || !parse_dev(env, argv[2], &dev) ||
        !get_int(env, argv[3], &m) || !get_int(env, argv[4], &n))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[2] = {m, n};
    mlx_array xa = new_arr(a, shp, 2, atom_c_str(env, argv[1]));
    mlx_array res = mlx_array_empty;
    int axes[2] = {0, 1};
    int rc = mlx_sum_axes(&res, xa, axes, 2, false, stream);
    mlx_array_free(xa);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_sum_axes failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_transpose(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a;
    std::string dev;
    int m, n;
    if (!get_bin(env, argv[0], &a) || !parse_dev(env, argv[2], &dev) ||
        !get_int(env, argv[3], &m) || !get_int(env, argv[4], &n))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[2] = {m, n};
    mlx_array xa = new_arr(a, shp, 2, atom_c_str(env, argv[1]));
    mlx_array res = mlx_array_empty;
    int rc = mlx_transpose(&res, xa, stream);
    mlx_array_free(xa);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_transpose failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_astype(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a;
    std::string dev;
    int n;
    if (!get_bin(env, argv[0], &a) || !parse_dev(env, argv[3], &dev) ||
        !get_int(env, argv[4], &n))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[1] = {n};
    mlx_array xa = new_arr(a, shp, 1, atom_c_str(env, argv[1]));
    mlx_dtype to = to_mlxc_dtype(atom_c_str(env, argv[2]));
    if (to == MLX_BOOL) {
      mlx_array_free(xa);
      return err_msg(env, "unsupported dtype");
    }
    mlx_array res = mlx_array_empty;
    int rc = mlx_astype(&res, xa, to, stream);
    mlx_array_free(xa);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_astype failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_reshape(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a;
    std::string dev;
    std::vector<int> shape;
    if (!get_bin(env, argv[0], &a) || !parse_dev(env, argv[2], &dev) ||
        !get_shape(env, argv[3], &shape))
      return enif_make_badarg(env);
    size_t n = 1;
    for (int s : shape)
      n *= (size_t)s;
    size_t elem = mlx_dtype_size(to_mlxc_dtype(atom_c_str(env, argv[1])));
    if (elem == 0 || a.size < n * elem)
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    mlx_array xa =
        mlx_array_new_data(a.data, shape.data(), (int)shape.size(),
                           to_mlxc_dtype(atom_c_str(env, argv[1])));
    // reshape with same element count is a metadata-only view; evaluate + copy
    mlx_array res = mlx_array_empty;
    int rc = mlx_reshape(&res, xa, shape.data(), (int)shape.size(), stream);
    mlx_array_free(xa);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_reshape failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_svd(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin x;
    std::string dev;
    int n;
    if (!get_bin(env, argv[0], &x) || !parse_dev(env, argv[1], &dev) ||
        !get_int(env, argv[2], &n))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[2] = {n, n};
    mlx_array xa = new_arr(x, shp, 2, "float32");
    mlx_vector_array res = mlx_vector_array_new();
    int rc = mlx_linalg_svd(&res, xa, false, stream);
    mlx_array_free(xa);
    mlx_device_free(mxdev);
    if (rc != 0) {
      mlx_vector_array_free(res);
      return err_msg(env, "mlx_linalg_svd failed");
    }
    mlx_array s = mlx_array_empty;
    rc = mlx_vector_array_get(&s, res, 0);
    mlx_vector_array_free(res);
    if (rc != 0)
      return err_msg(env, "svd result extract failed");
    ERL_NIF_TERM ret = eval_out(env, s);
    mlx_array_free(s);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_attn(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin q, k, v;
    std::string dev;
    int b, h, s, d;
    double scale;
    if (!get_bin(env, argv[0], &q) || !get_bin(env, argv[1], &k) ||
        !get_bin(env, argv[2], &v) || !get_double(env, argv[3], &scale) ||
        !parse_dev(env, argv[4], &dev) || !get_int(env, argv[5], &b) ||
        !get_int(env, argv[6], &h) || !get_int(env, argv[7], &s) ||
        !get_int(env, argv[8], &d))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[4] = {b, h, s, d};
    mlx_array q_ = mlx_array_new_data(q.data, shp, 4, MLX_FLOAT16);
    mlx_array k_ = mlx_array_new_data(k.data, shp, 4, MLX_FLOAT16);
    mlx_array v_ = mlx_array_new_data(v.data, shp, 4, MLX_FLOAT16);
    mlx_array res = mlx_array_empty;
    int rc = mlx_fast_scaled_dot_product_attention(&res, q_, k_, v_, (float)scale, "",
                                                   mlx_array_empty, mlx_array_empty, false,
                                                   stream);
    mlx_array_free(q_);
    mlx_array_free(k_);
    mlx_array_free(v_);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    if (rc != 0)
      return err_msg(env, "mlx_fast_scaled_dot_product_attention failed");
    ERL_NIF_TERM ret = eval_out(env, res);
    mlx_array_free(res);
    return ret;
  }
  CATCH
}

ERL_NIF_TERM nif_tiny_loop(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a, b;
    std::string dev;
    long iters;
    int n;
    if (!get_bin(env, argv[0], &a) || !get_bin(env, argv[1], &b) ||
        !parse_dev(env, argv[2], &dev) || !get_int64(env, argv[3], &iters) ||
        !get_int(env, argv[4], &n))
      return enif_make_badarg(env);
    mlx_device mxdev;
    mlx_stream stream;
    if (device_stream(dev, &mxdev, &stream)) {
      mlx_device_free(mxdev);
      return err_msg(env, "device init failed");
    }
    int shp[2] = {n, n};
    mlx_array xa = mlx_array_new_data(a.data, shp, 2, MLX_FLOAT32);
    mlx_array yb = mlx_array_new_data(b.data, shp, 2, MLX_FLOAT32);
    mlx_array res = mlx_array_empty;
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; i++) {
      mlx_matmul(&res, xa, yb, stream);
      mlx_array_eval(res);
    }
    auto t1 = std::chrono::steady_clock::now();
    long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    ErlNifBinary ob;
    enif_alloc_binary(mlx_array_nbytes(res), &ob);
    memcpy(ob.data, mlx_array_data_float32(res), mlx_array_nbytes(res));
    mlx_array_free(xa);
    mlx_array_free(yb);
    mlx_array_free(res);
    mlx_stream_free(stream);
    mlx_device_free(mxdev);
    return enif_make_tuple3(env, enif_make_atom(env, "ok"),
                            enif_make_int64(env, ns), enif_make_binary(env, &ob));
  }
  CATCH
}

ErlNifFunc nif_funcs[] = {
    {"device_check", 0, nif_device_check, 0},
    {"copy", 4, nif_copy, 0},
    {"matmul", 7, nif_matmul, 0},
    {"bmm", 8, nif_bmm, 0},
    {"add", 6, nif_add, 0},
    {"subtract", 6, nif_subtract, 0},
    {"multiply", 6, nif_multiply, 0},
    {"divide", 6, nif_divide, 0},
    {"exp", 5, nif_exp, 0},
    {"softmax_axis", 6, nif_softmax_axis, 0},
    {"sum", 5, nif_sum, 0},
    {"transpose", 5, nif_transpose, 0},
    {"astype", 5, nif_astype, 0},
    {"reshape", 4, nif_reshape, 0},
    {"svd", 3, nif_svd, 0},
    {"attn", 9, nif_attn, 0},
    {"tiny_loop", 5, nif_tiny_loop, 0},
};

} // namespace

ERL_NIF_INIT(Elixir.EMLX.C.NIF, nif_funcs, nullptr, nullptr, nullptr, nullptr)
