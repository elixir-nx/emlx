# EMLX

[![Package](https://img.shields.io/badge/-Package-important)](https://hex.pm/packages/emlx) [![Documentation](https://img.shields.io/badge/-Documentation-blueviolet)](https://hexdocs.pm/emlx)

EMLX is the Nx Backend for the [MLX](https://github.com/ml-explore/mlx) library.

Because of MLX's nature, EMLX with GPU backend is only supported on macOS.

MLX with CPU backend is available on most mainstream platforms, however, the CPU backend may not be as optimized as the GPU backend,
especially for non-macOS OSes, as they're not prioritized for development. Right now, EMLX supports x86_64 and arm64 architectures
on both macOS and Linux.

The M-Series Macs have an unified memory architecture, which allows for more passing data between the CPU and GPU to be effectively a no-op.

Besides the backend, this library also provides an Nx.Defn.Compiler implementation that allows for JIT compilation of Nx functions to MLX kernels.
Using this compiler is not much different from just using the default `Nx.Defn.Evaluator`, but because it sets an explicit `EMLX.eval` call,
it allows for better caching of Nx-defined functions by MLX itself.

Metal does not support 64-bit floats, so neither MLX nor EMLX do either.

## Usage

To use EMLX, you can add it as a dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:emlx, github: "elixir-nx/emlx", branch: "main"}
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
```

If you want to use the JIT compiler, you can set the default compiler as shown below.

Currently, the underlying implementation is not totally safe and could lead to a deadlocked dirty NIF, so this must be used with care.
Defaulting to Nx.Defn.Evaluator is the safest option for now.

```elixir
Nx.Defn.default_options(compiler: EMLX)
```

### MLX binaries

EMLX relies on the [MLX](https://github.com/ml-explore/mlx) library to function, and currently EMLX will download precompiled builds from [mlx-build](https://github.com/cocoa-xu/mlx-build).

#### Using precompiled binaries

While the default configuration should be suitable for most cases, there is however a number of environment variables that you may want to use in order to customize the variant of MLX binary.

The binaries are always downloaded to match the current configuration, so you should set the environment variables in .bash_profile or a similar configuration file so you don't need to export it in every shell session.

##### `LIBMLX_VERSION`

The version of the MLX binary to download. By default EMLX will always use the latest version possible.

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
