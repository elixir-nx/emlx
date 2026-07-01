# Stage 21 — observability: telemetry + debug assertion flags

Status: not started. Emily M18/M22 parity (see Stage 20).

## Why this stage exists

EMLX has zero `:telemetry` instrumentation. Both telemetry and
debug-assertion flags are cheap, self-contained, and valuable independent of
the fallback-removal work (Stages 16–19) — they're about making EMLX's
*existing* behavior observable, not changing what it lowers.

> **Correction from Stage 20's audit.** This doc originally assumed EMLX had
> *no* compile-time debug-assertion flags, mirroring Emily's M22 from
> scratch. That's wrong: `backend.ex:6-94` already ships
> `@enable_bounds_check` and `@detect_non_finites` — same
> `Application.compile_env/3`-gated, default-`false`, dead-code-eliminated
> design as Emily's M22. `@enable_bounds_check` already covers 100% of
> Emily's M22 target op list (`gather`/`indexed_add`/`indexed_put`/`take`/
> `take_along_axis`). `@detect_non_finites` covers `dot` only — **not**
> `conv` or the `EMLX.Fast` kernels. Procedure step 3 below is corrected to
> reflect "extend existing coverage," not "build from scratch."

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
3. Two compile-time opt-in debug flags, mirroring Emily M22 —
   **`:enable_bounds_check` already fully covers its target op list, no new
   work needed there; `:detect_non_finites` needs extending**:
   - `:enable_bounds_check` — already raises on out-of-range/negative indices
     in `gather`/`take`/`take_along_axis`/`indexed_add`/`indexed_put`
     (`backend.ex:1934,1978,2111,2121`). No action.
   - `:detect_non_finites` — already scans `dot` (`backend.ex:1313`) for
     NaN/Inf. **Extend** to `conv` and the fused `EMLX.Fast` kernels
     (rms_norm/layer_norm/sdpa) — these are the only two gaps.
   Both already default `false` and are dead-code-eliminated when off (verify
   via `Application.compile_env/3`, not a runtime `Application.get_env/3`
   check, so the cost is genuinely zero when disabled) — confirm the
   extension preserves that property.
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
- `:enable_bounds_check` (already complete, verified not regressed) and
  `:detect_non_finites` (extended to `conv` + `EMLX.Fast` kernels) are off by
  default with zero runtime cost when off, and correctly raise on violation
  when enabled.
