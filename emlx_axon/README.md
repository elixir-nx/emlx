# EMLXAxon

`emlx_axon` provides [Axon](https://github.com/elixir-nx/axon) model rewrites that swap supported nodes for `EMLX.Fast` Metal shader implementations, accelerating inference on Apple Silicon.

It builds on top of [`emlx`](https://github.com/elixir-nx/emlx) and is intended to be used alongside [Bumblebee](https://github.com/elixir-nx/bumblebee) for running LLM serving workloads on MLX.

## Usage

Add `emlx_axon` as a dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:emlx_axon, github: "elixir-nx/emlx", sparse: "emlx_axon", branch: "main"},
    {:emlx, github: "elixir-nx/emlx", branch: "main"}
  ]
end
```

## Model download

The examples and tests that run inference require local model checkpoints downloaded from HuggingFace.

Install the HuggingFace CLI if you don't have it:

```bash
pipx install huggingface_hub[cli]
```

Download the model checkpoints:

```bash
# 0.6B — small, fast to iterate (~400 MB)
huggingface-cli download lmstudio-community/Qwen3-0.6B-MLX-4bit \
  --local-dir ~/models/Qwen3-0.6B-MLX-4bit

# 8B — headline size (~5 GB)
huggingface-cli download lmstudio-community/Qwen3-8B-MLX-4bit \
  --local-dir ~/models/Qwen3-8B-MLX-4bit
```

Export the path before running tests or benchmarks:

```bash
export EMLX_QWEN3_MODEL_PATH=~/models/Qwen3-0.6B-MLX-4bit
```

### Pinning a model revision

For golden-token determinism, pin the model revision in HuggingFace by passing
`--revision <commit_sha>` to `huggingface-cli download`.

### CI note

Tests that require a local checkpoint are excluded from the default `mix test` run
and from CI — do not add a CI job that downloads the checkpoint, as the 8B model
is ~5 GB and the tests require local Apple Silicon hardware.
