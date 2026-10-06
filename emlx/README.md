# EMLX

[![Package](https://img.shields.io/badge/-Package-important)](https://hex.pm/packages/emlx) [![Documentation](https://img.shields.io/badge/-Documentation-blueviolet)](https://hexdocs.pm/emlx)

EMLX is the Nx Backend for the [MLX](https://github.com/ml-explore/mlx) library.

Because of MLX's nature, EMLX with GPU backend is only supported on macOS.

MLX with CPU backend is available on most mainstream platforms, however, the CPU backend may not be as optimized as the GPU backend,
especially for non-macOS OSes, as they're not prioritized for development. Right now, EMLX supports x86_64 and arm64 architectures
on both macOS and Linux.

The M-Series Macs have an unified memory architecture, which allows for more passing data between the CPU and GPU to be effectively a no-op.

Besides the backend, this library also provides a `Nx.Defn.Compiler` implementation that JIT-compiles Nx functions with smart use of MLX command queues.

- **Worker-thread dispatch** — MLX ops run on dedicated threads instead of BEAM dirty schedulers, eliminating scheduler starvation under load.
- **Per-process Metal command queues (`EMLX.CommandQueue`)** — each BEAM process can get its own GPU command queue for true process-level GPU isolation.
- **GPU pointer interop** — `Nx.Backend.from_pointer/5` and `to_pointer/2` for zero-copy Metal buffer sharing with other languages, such as with Python via Pythonx.

Metal does not support 64-bit floats, so neither MLX nor EMLX do either.

## Usage

To use EMLX, you can add it as a dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:emlx, github: "elixir-nx/emlx", sparse: "emlx", branch: "main"}
  ]
end
```

Then, you just need to set `EMLX.Backend` as the default backend for your Nx functions:

```elixir
Nx.default_backend(EMLX.Backend)

# Setting the device to the CPU (default)
Nx.default_backend({EMLX.Backend, device: :cpu})

# Setting the device to the GPU
Nx.default_backend({EMLX.Backend, device: :gpu})

# or use the application config using one of the alternatives above as the value:

config :nx, :default_backend, EMLX.Backend
config :nx, :default_backend, {EMLX.Backend, device: :cpu}
config :nx, :default_backend, {EMLX.Backend, device: :gpu}
```

If you want to use the JIT compiler, you can set the default compiler as shown below.

```elixir
Nx.Defn.default_options(compiler: EMLX)

# Alternatively, we can set this in the application environment

config :nx, :default_defn_options, compiler: EMLX
```

### Compile-time debug flags

Development-only assertion flags (`:enable_bounds_check`, `:detect_non_finites`,
`:compiler_debug`) are read at compile time via `Application.compile_env/3`.
See the [EMLX moduledoc](https://hexdocs.pm/emlx/EMLX.html#module-compile-time-debug-flags)
and `config/dev.exs` for how to enable them and what each flag does.

### MLX binaries

EMLX relies on the [MLX](https://github.com/ml-explore/mlx) library to function, and currently EMLX will download precompiled builds from [mlx-build](https://github.com/cocoa-xu/mlx-build).

#### Using precompiled binaries

While the default configuration should be suitable for most cases, there is however a number of environment variables that you may want to use in order to customize the variant of MLX binary.

The binaries are always downloaded to match the current configuration, so you should set the environment variables in .bash_profile or a similar configuration file so you don't need to export it in every shell session.

##### `LIBMLX_VERSION`

The version of the MLX binary to download. By default EMLX will always use the latest version possible.

##### `LIBMLX_MACOS_COMPAT`

Defaults to `false`.

On Apple Silicon macOS, precompiled libmlx archives are keyed by deployment target ([mlx-build](https://github.com/cocoa-xu/mlx-build#macos)):

| deployment target | runs on     | AOT NAX kernels |
|-------------------|-------------|-----------------|
| `26.2` (default)  | macOS 26.2+ | yes             |
| `14.0`            | macOS 14+   | no              |

The reduced featureset is only the missing **ahead-of-time NAX kernels** — MLX's fast GEMM/attention paths on Apple's `MetalPerformancePrimitives` (Metal 4). The `14.0` archive still provides normal Metal GPU ops. On macOS 26.2+, `LIBMLX_ENABLE_JIT=true` can still JIT-compile NAX at runtime even with the `14.0` archive.

Set `LIBMLX_MACOS_COMPAT=true` to download the `14.0` archive. Use this on macOS 15 (and any host older than 26.2); the default `26.2` build will refuse to compile there instead of crashing the VM at runtime.

##### `LIBMLX_DEPLOYMENT_TARGET`

Optional explicit override of the macOS deployment target segment. Accepted values: `26.2`, `14.0`. When set, this wins over `LIBMLX_MACOS_COMPAT`.

##### `LIBMLX_ENABLE_JIT`

Defaults to `false`.

Using JIT compilation for Metal kernels when set to `true`.

##### `LIBMLX_ENABLE_DEBUG`

Defaults to `false`.

Enhance metal debug workflow by enabling debug information in the Metal shaders when set to `true`.

##### `LIBMLX_CACHE`

The directory to store the downloaded and built archives in. Defaults to the standard cache location for the given operating system.

#### Compiling from source

If you want to compile MLX from source, you can do so by setting the `LIBMLX_BUILD` environment variable to `true`.

Environment variables listed in the previous section will still apply.

#### Testing on macOS 15

On a macOS 15 machine locally:

```bash
export LIBMLX_MACOS_COMPAT=true
cd emlx && mix deps.get && mix test
```

Without that flag, compilation fails with a message naming `LIBMLX_MACOS_COMPAT` instead of crashing the VM at runtime.

### The `mlx-c` lane (experimental, opt-in)

[`mlx-c`](https://github.com/ml-explore/mlx-c) is Apple's official C API
for MLX. This branch adds an opt-in dispatch layer built on it, so Elixir
can reach MLX through the officially maintained binding instead of direct
`mlx::core` C++ calls.

Build the two lane libraries (fetches the pinned `mlx-c` release and builds
its CMake-pinned `libmlx` alongside EMLX's own):

```bash
EMLX_MLXC=true mix compile
```

This produces two self-contained libraries in `priv/`, each statically
linking its own `libmlx` so they run side-by-side with `libemlx.so`:

* `libemlx_c.so` (`EMLX.C.NIF`) — ops implemented with `mlx_c_*` calls
* `libemlx_ref.so` (`EMLX.C.RefNIF`) — the identical op surface via direct
  C++, used as the differential reference

Select the layer at runtime:

```elixir
config :emlx, :mlx_api, :c    # or :cpp (default)
```

`EMLX.C.matmul(a_bin, b_bin, :float32, :gpu, m, k, n)` then dispatches
through the configured lane (binary in, evaluated binary out).

Differential tests (both lanes must produce bit-identical outputs):

```bash
EMLX_MLXC=true mix test test/emlx/c_test.exs
```

Benchmark (interleaved lane sampling; prints min latencies and the ratio):

```bash
REPS=25 TINY_ITERS=50000 mix run bench/emlx_c_bench.exs
```

On an M3 Ultra, across matmul/bmm/add/softmax/transpose/astype/sdpa/svd on
GPU and CPU, the mlx-c lane matched the direct-C++ lane within measurement
noise (ratios 0.8–1.1, results bit-exact, including a 50k-iteration tiny
matmul dispatch loop).

#### Caveats

* The pinned `mlx-c` (v0.7.0) builds against `libmlx` 0.32.2, while the
  default EMLX path currently ships prebuilt 0.32.0 — the lanes are built
  and versioned independently by design, but bumping EMLX's `@mlx_version`
  to the mlx-c pin would keep a single MLX version across the project.
* `mlx-c` out-params are struct handles: they must be initialized with the
  matching `mlx_*_new()` constructors before use (uninitialized stack
  garbage faults hard inside `mlx_get_default_stream`-style calls).
* The lane currently exposes a representative op set with binary I/O;
  resource-tensor plumbing, async dispatch (`EMLX.CommandQueue`), and the
  remaining ops (`emlx_fast` custom kernels, compiler/plugins) stay on the
  C++ path for now.
