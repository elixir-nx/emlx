// EMLX direct-C++ reference lane: identical op surface to emlx_c_nif.cpp
// but implemented with direct mlx::core C++ calls (the style emlx's c_src
// uses today). Used by tests/bench to differential-check the mlx-c lane.
//
// Copyright (c) 2026. MIT license (same as EMLX).

#include <erl_nif.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace cpp = mlx::core;

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

cpp::Device to_dev(const std::string &d) {
  return d == "gpu" ? cpp::Device(cpp::Device::DeviceType::gpu, 0)
                    : cpp::Device(cpp::Device::DeviceType::cpu, 0);
}

cpp::Dtype to_dtype(const char *s) {
  if (!strcmp(s, "float32"))
    return cpp::float32;
  if (!strcmp(s, "float16"))
    return cpp::float16;
  if (!strcmp(s, "bfloat16"))
    return cpp::bfloat16;
  if (!strcmp(s, "int32"))
    return cpp::int32;
  if (!strcmp(s, "int64"))
    return cpp::int64;
  throw std::runtime_error("unsupported dtype");
}

size_t dtype_size(cpp::Dtype d) { return d.size(); }

cpp::array new_arr(const Bin &b, std::vector<int> shape, const char *dt) {
  cpp::Dtype d = to_dtype(dt);
  size_t n = 1;
  for (int s : shape)
    n *= (size_t)s;
  size_t nbytes = n * dtype_size(d);
  if (b.size < nbytes)
    throw std::runtime_error("binary smaller than shape");
  cpp::allocator::Buffer buf = cpp::allocator::malloc(nbytes);
  std::memcpy(buf.raw_ptr(), b.data, nbytes);
  return cpp::array(buf, cpp::Shape(shape.begin(), shape.end()), d,
                    [](cpp::allocator::Buffer x) { cpp::allocator::free(x); });
}

ERL_NIF_TERM eval_out(ErlNifEnv *env, const cpp::array &a) {
  cpp::eval({a});
  const void *p = nullptr;
  if (a.dtype() == cpp::float32)
    p = a.data<float>();
  else if (a.dtype() == cpp::float16)
    p = a.data<cpp::float16_t>();
  else if (a.dtype() == cpp::bfloat16)
    p = a.data<cpp::bfloat16_t>();
  else if (a.dtype() == cpp::int32)
    p = a.data<int32_t>();
  else if (a.dtype() == cpp::int64)
    p = a.data<int64_t>();
  else
    return err_msg(env, "unsupported output dtype");
  return ok_bin(env, p, a.nbytes());
}

// ───────────────────────────────── ops ──────────────────────────────────────

ERL_NIF_TERM nif_device_check(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    bool gpu_avail = cpp::is_available(cpp::Device(cpp::Device::DeviceType::gpu, 0));
    (void)cpp::default_device();
    return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                            enif_make_atom(env, gpu_avail ? "gpu" : "cpu"));
  }
  CATCH
}

ERL_NIF_TERM nif_copy(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin x;
    std::string dev;
    int n;
    if (!get_bin(env, argv[0], &x) || !parse_dev(env, argv[1], &dev) ||
        !get_int(env, argv[2], &n))
      return enif_make_badarg(env);
    cpp::Device d = to_dev(dev);
    cpp::array a = new_arr(x, {n}, atom_c_str(env, argv[3]));
    return eval_out(env, a);
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
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, {m, k}, dt);
    cpp::array yb = new_arr(b, {k, n}, dt);
    return eval_out(env, cpp::matmul(xa, yb, d));
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
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, {bb, m, k}, dt);
    cpp::array yb = new_arr(b, {bb, k, n}, dt);
    return eval_out(env, cpp::matmul(xa, yb, d));
  }
  CATCH
}

#define BINARY_OP(NAME, CPPF)                                                \
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
      cpp::Device d = to_dev(dev);                                         \
      cpp::array xa = new_arr(a, {m, n}, dt);                            \
      cpp::array yb = new_arr(b, {m, n}, dt);                            \
      return eval_out(env, CPPF(xa, yb, d));                             \
    }                                                                      \
    CATCH                                                                    \
  }

BINARY_OP(add, cpp::add)
BINARY_OP(subtract, cpp::subtract)
BINARY_OP(multiply, cpp::multiply)
BINARY_OP(divide, cpp::divide)

ERL_NIF_TERM nif_exp(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    Bin a;
    std::string dev;
    int m, n;
    if (!get_bin(env, argv[0], &a) || !parse_dev(env, argv[2], &dev) ||
        !get_int(env, argv[3], &m) || !get_int(env, argv[4], &n))
      return enif_make_badarg(env);
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, {m, n}, atom_c_str(env, argv[1]));
    return eval_out(env, cpp::exp(xa, d));
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
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, {m, n}, atom_c_str(env, argv[1]));
    return eval_out(env, cpp::softmax(xa, std::vector<int>{axis}, false, d));
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
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, {m, n}, atom_c_str(env, argv[1]));
    return eval_out(env, cpp::sum(xa, std::vector<int>{0, 1}, false, d));
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
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, {m, n}, atom_c_str(env, argv[1]));
    return eval_out(env, cpp::transpose(xa, d));
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
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, {n}, atom_c_str(env, argv[1]));
    return eval_out(env, cpp::astype(xa, to_dtype(atom_c_str(env, argv[2])), d));
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
    const char *dt = atom_c_str(env, argv[1]);
    size_t n = 1;
    for (int s : shape)
      n *= (size_t)s;
    if (a.size < n * dtype_size(to_dtype(dt)))
      return enif_make_badarg(env);
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, shape, dt);
    return eval_out(env, xa); // reshape of row-major host buffer is identity bytes
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
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(x, {n, n}, "float32");
    std::vector<cpp::array> res = cpp::linalg::svd(xa, false, d);
    return eval_out(env, res[0]);
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
    cpp::Device dev_ = to_dev(dev);
    cpp::array q_ = new_arr(q, {b, h, s, d}, "float16");
    cpp::array k_ = new_arr(k, {b, h, s, d}, "float16");
    cpp::array v_ = new_arr(v, {b, h, s, d}, "float16");
    return eval_out(env, cpp::fast::scaled_dot_product_attention(
                               q_, k_, v_, (float)scale, "", std::nullopt, std::nullopt,
                               dev_));
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
    cpp::Device d = to_dev(dev);
    cpp::array xa = new_arr(a, {n, n}, "float32");
    cpp::array yb = new_arr(b, {n, n}, "float32");
    cpp::array c = cpp::matmul(xa, yb, d);
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; i++) {
      c = cpp::matmul(xa, yb, d);
      cpp::eval({c});
    }
    auto t1 = std::chrono::steady_clock::now();
    long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    ErlNifBinary ob;
    enif_alloc_binary(c.nbytes(), &ob);
    memcpy(ob.data, c.data<float>(), c.nbytes());
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

ERL_NIF_INIT(Elixir.EMLX.C.RefNIF, nif_funcs, nullptr, nullptr, nullptr, nullptr)
