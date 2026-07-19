# EMLXAxon

`emlx_axon` provides [Axon](https://github.com/elixir-nx/axon) model rewrites
and native model plugins that accelerate inference on Apple Silicon.

It builds on top of [`emlx`](https://github.com/elixir-nx/emlx) and is intended to be used alongside [Bumblebee](https://github.com/elixir-nx/bumblebee) for running LLM serving workloads on MLX.

## Usage

Add `emlx_axon` as a dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:emlx_axon, github: "elixir-nx/emlx", sparse: "emlx_axon", branch: "main"},
    {:emlx, github: "elixir-nx/emlx", branch: "main", override: true}
  ]
end
```

## Development and release sequencing

The monorepo CI sets `EMLX_AXON_LOCAL_EMLX=true` so EMLXAxon is compiled and
tested against the sibling EMLX source tree. This is required while the generic
plugin ABI is newer than the latest published EMLX package.

Before publishing the matching EMLXAxon release, EMLX must first release the
plugin ABI used here. The EMLXAxon dependency lower bound can then be updated
to that released version. The package includes its plugin C++ source and
Makefile; EMLX packages the shared plugin ABI header.

The EMLX plugin registry keeps accepted shared objects loaded for the VM
lifetime. Stopping and starting `:emlx_axon` is supported and loads the same
Qwen3 and Llama plugins idempotently; replacing an accepted plugin under the
same name requires restarting the BEAM VM.

## Native dense Llama

Standard Hugging Face Llama checkpoints can be loaded from safetensors and run
through the native Llama plugin:

```elixir
{:ok, tokenizer} =
  Bumblebee.load_tokenizer({:hf, "meta-llama/Llama-3.2-1B"})

{:ok, state} =
  EMLXAxon.Llama.DenseLoader.from_safetensors_dir(
    "~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/<revision>",
    type: :f16
  )

result =
  EMLXAxon.TextGeneration.run(
    tokenizer,
    state,
    "The capital of France is",
    max_new_tokens: 32,
    sampler: :greedy
  )

IO.inspect(result)
```

`EMLXAxon.TextGeneration.run/4`, `stream/5`, and `serving/3` dispatch from the
loaded state, so the same APIs support native Qwen3 and dense Llama models.

## Model download

The examples and tests that run inference require local model checkpoints downloaded from HuggingFace.

Install the HuggingFace CLI if you don't have it:

```bash
pipx install huggingface_hub[cli]
```

Download the model checkpoints:

```bash
# Dense Llama 3.2 1B. This repository is gated and requires access approval.
huggingface-cli download meta-llama/Llama-3.2-1B \
  --local-dir ~/models/Llama-3.2-1B

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

Run the dense Llama comparison from `emlx_axon`:

```bash
EMLX_DENSE_LLAMA_MODEL=~/models/Llama-3.2-1B \
EMLX_DENSE_LLAMA_DEVICE=gpu \
EMLX_DENSE_LLAMA_TYPE=f16 \
EMLX_DENSE_LLAMA_STRICT_LENGTH=true \
mix run bench/validate_llama_dense.exs
```

The report compares stock Bumblebee, the EMLXAxon rewrite path, and the native
Llama plugin with equal token counts. It includes load time, warmed one-token
latency, request duration, throughput, token counts, and finish reasons.

### Pinning a model revision

For golden-token determinism, pin the model revision in HuggingFace by passing
`--revision <commit_sha>` to `huggingface-cli download`.

### CI note

Tests that require a local checkpoint are excluded from the default `mix test` run
and from CI — do not add a CI job that downloads the checkpoint, as the 8B model
is ~5 GB and the tests require local Apple Silicon hardware.
