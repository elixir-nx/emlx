# Stage 21 — observability: telemetry + debug assertion flags

Status: not started. Emily M18/M22 parity (see Stage 20).

## Why this stage exists

EMLX has zero `:telemetry` instrumentation and no compile-time
debug-assertion flags. Both are cheap, self-contained, and valuable
independent of the fallback-removal work (Stages 16–19) — they're about
making EMLX's *existing* behavior observable, not changing what it lowers.

## Procedure

1. Add `{:telemetry, "~> 1.0"}` to `emlx/mix.exs`.
2. New `EMLX.Telemetry` module, `:telemetry.span/3`-wrapped events at the
   evaluation boundary (not at every graph-construction NIF — those are <10μs
   and do no work; the evaluation boundary is where MLX actually runs
   kernels, mirroring Emily M18's own scope call):
   - `[:emlx, :eval, *]` around the existing `EMLX.eval/1`.
   - `[:emlx, :to_binary, *]` around `EMLX.Backend.to_binary/2` (the
     `Nx.to_binary/1` path) with `:shape`/`:dtype`/`:byte_size` metadata.
   - A discrete `[:emlx, :memory, :stats]` event wrapping EMLX's existing
     memory-stats NIF(s) (active/peak/cache bytes).
3. Two compile-time opt-in debug flags, mirroring Emily M22:
   - `:debug_bounds_check` — raise on out-of-range/negative indices in
     `gather`/`take`/`take_along_axis`/`indexed_add`/`indexed_put`.
   - `:debug_detect_nan_inf` — scan results of `dot`/`conv` and the fused
     `EMLX.Fast` kernels (rms_norm/layer_norm/sdpa) for NaN/Inf.
   Both default `false`; guarded branches must be dead-code-eliminated when
   off (verify via `Application.compile_env/3`, not a runtime `Application.
   get_env/3` check, so the cost is genuinely zero when disabled).
4. Tests: attach a test `:telemetry` handler and assert span start/stop fire
   with correct metadata; compile a small test target with each debug flag
   on and assert it raises on a crafted violation, and with it off (default)
   assert no raise/no overhead regression.
5. Document every event in `EMLX.Telemetry`'s moduledoc (mirrors Emily's own
   `Emily.Telemetry` moduledoc structure — an explicit enumerated list, not a
   scattered set of call sites).

## Acceptance

- `EMLX.Telemetry` ships with `[:emlx, :eval, *]`, `[:emlx, :to_binary, *]`,
  `[:emlx, :memory, :stats]`, all documented and tested.
- `:debug_bounds_check` / `:debug_detect_nan_inf` are off by default with
  zero runtime cost when off, and correctly raise on violation when enabled.
