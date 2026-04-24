# emlx validation

Validation tests and end-to-end benchmarks for [`emlx`](../).

## Running the standard test suite

```bash
cd validation
mix deps.get
mix test
```

---

## Quantized inference (opt-in)

The Qwen3 quantized inference tests and benchmarks require a local model
checkpoint and are excluded from the default test run. They are part of
stage A3 of the emlx#108 investigation.

### One-time setup

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

Export the path:

```bash
export EMLX_QWEN3_MODEL_PATH=~/models/Qwen3-0.6B-MLX-4bit
```

### Run the conformance test

```bash
cd validation
mix test --only quantized_inference
```

### Run the throughput benchmark

```bash
cd validation
mix run bench/qwen3_e2e_bench.exs
```

### Run the headline 8B benchmark

```bash
cd validation
EMLX_QWEN3_MODEL_PATH=~/models/Qwen3-8B-MLX-4bit \
  mix run bench/qwen3_e2e_bench.exs
```

### Expected results (A0 reference, M4 Max 64 GB, macOS 26.3)

Captured from `bobby_posts` (reference Elixir implementation):

| model  | sampler   | prefill ms | median tok ms | e2e tok/s |
|--------|-----------|-----------|---------------|-----------|
| 0.6B   | greedy    | 67.4      | 14.23         | 69.7      |
| 0.6B   | top_p_cpu | 35.6      | 56.19         | 17.3      |
| 8B     | greedy    | 118.1     | 31.16         | 30.6      |
| 8B     | top_p_cpu | 117.9     | 71.70         | 13.6      |

### Pinning a model revision

For golden-token determinism, pin the model revision in HuggingFace by passing
`--revision <commit_sha>` to `huggingface-cli download`. Record the SHA in this
file alongside the golden token sequence in `test/qwen3_quantized_test.exs`.

### CI note

The `:quantized_inference` tag is excluded from the default `mix test` run
and from CI. Do not add a CI job that downloads the checkpoint — the 8B model
is ~5 GB and the tests require local Apple Silicon hardware.
