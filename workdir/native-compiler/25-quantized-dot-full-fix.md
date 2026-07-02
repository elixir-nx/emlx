# Stage 25 — full fix: quantized `Nx.dot` under `compiler: EMLX`

Status: not started. Follow-on to
[`24-quantized-dot-compiler-gap`](24-quantized-dot-compiler-gap.md)'s interim
raise; implements that stage's "Deferred: the full fix" section. Numbered 25
(inserted into the burndown ahead of the then-Stage-25
`25-fine-nif-refactor`, which shifted down to
[`26-fine-nif-refactor`](26-fine-nif-refactor.md); Stages 26–29 shifted to
27–30 accordingly).

## Why this stage exists

Stage 24 root-caused and gave an interim, actionable raise for a real gap:
a quantized `Nx.dot` operand is invisible to the native compiler, because
quantization dispatch (`EMLX.Backend.dot/7` → `EMLX.quantized_matmul`) is
resolved by inspecting a bound tensor's runtime `quantization_config`
metadata inside an eager per-op callback — metadata that does not exist yet
when `EMLX.Native.Expr` traces and compiles the graph once, ahead of any real
tensor binding. The result: `compiler: EMLX` cannot run a stock
Bumblebee-generated Axon graph (`defn_options: [compiler: EMLX]`, no
`EMLXAxon.rewrite/2`) against MLX-4bit-quantized weights — Stage 24 shipped a
clear `ArgumentError` instead of the previous opaque NIF crash, but the
underlying configuration still does not work.

This was re-confirmed live via `emlx_axon/bench/validate_qwen3.exs`'s `bb
base` warmup path against `Qwen3-0.6B-MLX-4bit`:

```
** (ArgumentError) compiler: EMLX does not support a quantized input tensor
(shape {1024, 3072}, type {:bf, 16}). Quantized-weight matmul dispatch
(EMLX.Backend.dot/7 -> EMLX.quantized_matmul) only happens in the eager
per-op backend path; the native single-NIF-replay compiler traces and
compiles the graph once, before any real tensor (and its quantization
metadata) is bound, so it cannot see or lower this dispatch (see
workdir/native-compiler/24-quantized-dot-compiler-gap.md). Use compiler:
Nx.Defn.Evaluator, or dequantize the tensor first with EMLX.dequantize/1, or
use a hand-written eager pipeline (e.g. EMLXAxon.Qwen3.Generate) instead.
    (emlx 0.3.1) lib/emlx.ex:2213: EMLX.reject_quantized_native_input!/1
    (emlx 0.3.1) lib/emlx.ex:2199: anonymous fn/2 in EMLX.materialise_input_refs/2
    (elixir 1.20.1) lib/enum.ex:1725: Enum."-map/2-lists^map/1-1-"/2
    (emlx 0.3.1) lib/emlx.ex:2240: anonymous fn/6 in EMLX.build_native_eval_fn/4
    (nx 0.12.1) lib/nx/defn/compiler.ex:161: Nx.Defn.Compiler.__jit__/4
```

Stage 24 explicitly deferred the full fix pending a scoping decision: is
"stock Bumblebee graph + quantized weights + `compiler: EMLX`" (`bb base`)
worth supporting, given the hand-written `native` path
(`EMLXAxon.TextGeneration`/`Qwen3.Generate`) already covers real deployment
and performs far better? This stage's premise is **yes** — the goal is to
get `bb base` actually running end-to-end (through `validate_qwen3.exs`'s
warmup and bench loop) against the quantized Qwen3 checkpoint, not just to
raise a clearer error.

## Procedure

Advisor-recommended direction from Stage 24, to implement here: **call-time
program specialization.**

1. **Quantization-signature detection.** `EMLX.build_native_eval_fn`'s
   closure already sees real bound tensors (via `materialise_input_refs`,
   `emlx/lib/emlx.ex`) before the NIF call. At that point, inspect each bound
   parameter's `quantization_config` and derive a "quantization signature"
   (e.g. a bitset/map of parameter positions → `{bits, group_size,
   transpose}`).
2. **Specialized program cache.** On first call with a non-empty
   quantization signature, compile a *second* program (cached alongside the
   original, keyed by `{original compile key, quantization signature}`)
   whose `:dot` lowering for the quantized positions emits a new
   `quantized_matmul` IR opcode instead of plain `:dot`, mirroring
   `EMLX.Backend.quantized_dot/4` — `transpose`/`group_size`/`bits` ride the
   int64 iattr channel, `scales`/`biases` are threaded through as additional
   hidden inputs appended at call-construction time (not part of the
   originally-traced `Expr`).
3. **New C++ opcode.** Add a `quantized_matmul` opcode to
   `emlx_compiler.cpp` wrapping `mlx::core::quantized_matmul` (already
   NIF-exposed via `EMLX.quantized_matmul/7` and used eagerly by
   `EMLX.Backend.quantized_dot/4` today) — no new MLX-level capability
   needed, just a lowering/dispatch path mirroring the existing eager one.
4. **Wire it into `materialise_input_refs`/`build_native_eval_fn`.** Replace
   (or gate) Stage 24's `reject_quantized_native_input!/1` pre-flight raise
   with: detect the signature, compile/fetch the specialized program, and
   dispatch to it instead of raising — falling back to the Stage 24 raise
   only for configurations this stage explicitly does not cover (e.g.
   `EMLXAxon.rewrite/2`'s `native_kv_attn_callback`, which stays
   unsupported per Stage 24's "related but distinct finding": real host
   blocking + mutable ETS side-channel state, permanently unlowerable,
   out of scope here).
5. **Validate against `validate_qwen3.exs`.** The concrete acceptance
   target: `bb base`'s warmup + bench loop runs to completion against
   `Qwen3-0.6B-MLX-4bit` and produces coherent output, without dequantizing
   first and without switching `compiler:` away from `EMLX`.
6. **Regression tests.** Extend/replace Stage 24's
   `emlx/test/emlx/native/expr_test.exs` `:stage24` tests: keep a case
   proving the previously-unsupported configurations that remain
   out-of-scope still raise clearly, and add new tests proving a quantized
   `Nx.dot` (single- and multi-quantized-operand `defn`s) under `compiler:
   EMLX` now matches `Nx.Defn.Evaluator` / eager `EMLX.Backend.dot/7` output
   within tolerance.

## Acceptance

- `emlx_axon/bench/validate_qwen3.exs`'s `bb base` path runs end-to-end
  (warmup + bench) against `Qwen3-0.6B-MLX-4bit` under `compiler: EMLX`,
  producing coherent generated text — the terminal failure quoted above no
  longer reproduces.
- A quantized `Nx.dot` (and, if reachable through the same specialization
  path, other quantized-weight ops the Bumblebee graph exercises) lowers and
  replays correctly under `compiler: EMLX`, equivalence-tested against
  `Nx.Defn.Evaluator`/eager `EMLX.Backend` within documented tolerance.
- `EMLXAxon.rewrite/2` + quantized weights (`bb+rewrite`) remains explicitly
  out of scope (per Stage 24) and continues to raise clearly, not silently
  mis-specialize.
- Full `emlx`/`emlx_axon` suites green; no perf regression on the existing
  `bench/` baseline for non-quantized `compiler: EMLX` programs.

## Results

_(fill in when tackled)_
