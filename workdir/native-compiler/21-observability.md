# Stage 21 — observability: telemetry + debug assertion flags

Status: done. Emily M18/M22 parity (see Stage 20).

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

## Results

**Advisor sign-off (before starting).** Confirmed the M22-correction claim
(bounds-check needs no re-audit, just a green no-op regression test) and
flagged the `EMLX.Fast`-is-a-separate-module problem as the real design
decision: extract a shared `EMLX.Debug` module with a public macro (one
`compile_env` read, one source of truth) rather than duplicating the flag
attribute into `fast.ex` — the `debug_flags_test.exs` redeclaration pattern
is a test-only precedent, not a production one. Also flagged span
boundaries (span the full `await_worker` round-trip for `eval`, not just
NIF dispatch; span the sync-forcing blob copy for `to_binary`) and scope
(`:detect_non_finites` extension strictly to the named kernels —
rms_norm/layer_norm/sdpa — not RoPE/SwiGLU, which are out of this stage's
list).

**What shipped:**

1. **`EMLX.Debug`** (`emlx/lib/emlx/debug.ex`, new) — extracted
   `:detect_non_finites` + `assert_no_nan_inf!/2` out of `EMLX.Backend`'s
   former private macro into a shared public macro, `require`d/`import`ed by
   both `EMLX.Backend` and `EMLX.Fast`. `:enable_bounds_check` and its two
   macros stayed in `EMLX.Backend` untouched (single-module use, no sharing
   problem, and the advisor's "don't re-audit" guidance applied).
2. **`:detect_non_finites` extended** to `EMLX.Backend.conv/4` (bound the
   final ref before `to_nx/2`, mirroring `dot/7`'s existing pattern) and to
   `EMLX.Fast`'s `rms_norm_callback/2`, `layer_norm_callback/2`,
   `layer_norm_no_bias_callback/2`, and all four `sdpa_*_callback/2`
   variants (including `sdpa_causal_key_masked_callback/2`, not explicitly
   named in the stage doc's "sdpa" bullet but the same op family). RoPE and
   SwiGLU deliberately excluded (advisor: not on Stage 21's named list — a
   follow-up if ever wanted, not scope creep here).
3. **`EMLX.Telemetry`** (`emlx/lib/emlx/telemetry.ex`, new) — `[:emlx, :eval,
   *]` spans the full `EMLX.eval/1` round-trip (through
   `resolve_worker`/`await_worker`, not just NIF dispatch, per advisor);
   `[:emlx, :to_binary, *]` spans `EMLX.Backend.to_binary/2`'s sync-forcing
   blob copy with `:shape`/`:dtype`/`:byte_size` metadata (byte size of the
   binary actually returned, i.e. measured after `maybe_modify_binary/3`,
   not the raw MLX blob); `[:emlx, :memory, :stats]` is a discrete event via
   `EMLX.Telemetry.memory_stats/0` wrapping the existing
   `EMLX.memory_info/0`. `{:telemetry, "~> 1.0"}` added as a direct dep to
   `emlx/mix.exs` (was already resolved transitively via `nx`, so no new
   library entered the build). Moduledoc mirrors `~/coding/emily`'s
   `Emily.Telemetry` structure (explicit enumerated event list + "Attaching
   a handler" example) for the applicable event subset only — Emily's
   fallback/block-dispatch/compiler-fallback events don't apply to EMLX's
   zero-fallback design (Stage 19) and were intentionally not mirrored.
4. **Tests:**
   - `EMLX.TelemetryTest` (new) — mirrors `Emily.TelemetryTest`'s structure
     for the applicable events: eval start/stop, to_binary stop metadata,
     memory_stats measurements + event.
   - `debug_flags_test.exs` extended with two new "off, zero-cost" checks —
     for `conv/4` and for all seven touched `EMLX.Fast` callbacks — that
     disassemble the compiled BEAM and assert `EMLX.is_nan/1`/
     `EMLX.is_infinity/1` calls are absent. **Correction made while writing
     these**: the pre-existing `dot/7` test's technique (asserting no local
     call named `:assert_no_nan_inf!`) is actually vacuous — macros never
     compile to a call named after themselves, on or off, so that assertion
     was passing trivially regardless of flag state (confirmed by manual
     disassembly, see below). Left the pre-existing `dot`/`gather` tests
     unmodified (out of this stage's scope per the advisor, and
     `:enable_bounds_check`'s equivalent macro has the same characteristic)
     but documented the distinction in the test file's moduledoc and used
     the more meaningful check — presence/absence of the macro's *expanded*
     calls — for every new test added here.
   - `debug_flags_functional_test.exs` (new) — the acceptance criteria's
     "correctly raise on violation when enabled" half, which had **no
     existing test infrastructure at all** (neither for
     `:detect_non_finites` nor `:enable_bounds_check` — confirmed via
     `rg`/`grep` across `emlx/test` and `.github/workflows/emlx.yml`).
     `Application.compile_env/3` bakes both flags in at compile time, so
     neither can be toggled mid-test-run; added an opt-in path instead:
     `config/config.exs` reads `EMLX_DEBUG_FLAGS=1` (test env only) to flip
     **both** `:detect_non_finites` and `:enable_bounds_check` on,
     `test/test_helper.exs` excludes the `:debug_flags_functional` tag
     unless that var is set (so a normal `mix test` — flags off, matching
     production — doesn't fail on tests that only pass when compiled with
     them on). Covers `take` (bounds-check) plus `dot`/`conv`/`EMLX.Fast.rms_norm`
     (non-finites). Verified manually both ways: `EMLX_DEBUG_FLAGS=1 mix
     test --force --include debug_flags_functional` → 4 passed; plain `mix
     test` → those 4 excluded, the "off" opcode tests (including the two
     new ones) pass. Full suite: 2572 passed, 5 excluded (1
     `:large_memory`, 4 `:debug_flags_functional`) both before and after.
     **Round-1 reviewer caught a real gap here** (see below) — the first
     pass only wired the opt-in toggle and a functional test for
     `:detect_non_finites`, leaving `:enable_bounds_check`'s "raises when
     on" half of the acceptance criteria completely unverified. Fixed by
     extending the same toggle/test to cover both flags (the `take`
     out-of-bounds-index test above).
   - Setup-guard implementation note: the functional test file's
     `setup_all` diagnostic (flunk with a helpful message if run without
     `EMLX_DEBUG_FLAGS=1`) reads the flags via `Application.get_env/3`, not
     `compile_env/3` — using the compile-time-constant attributes there
     triggered an Elixir 1.18 type-checker warning ("conditional expression
     … will always evaluate to … false") that fails CI, since
     `.github/workflows/emlx.yml` runs `mix test --warnings-as-errors`.
     Confirmed via a clean-worktree (`git stash -u`) comparison that this
     warning was newly introduced by this stage's code (not pre-existing) —
     `--warnings-as-errors` already fails on the pristine tree too, but only
     from an unrelated, pre-existing `Nx.reflect/2` deprecation warning in a
     vendored Nx doctest; left untouched, out of scope.
   - Manual disassembly cross-check (not committed, exploratory): toggled
     `:detect_non_finites` on via a temporary `config.exs` edit + `mix
     compile --force`, confirmed `EMLX.is_nan/1`/`EMLX.is_infinity/1` calls
     appear in `standard_dot/10`, `conv/4`, `rms_norm_callback/2`, and
     `sdpa_callback/2`'s bytecode when on and are absent when off — the
     concrete evidence behind the advisor's flagged risk ("confirm dead-code
     elimination holds in `EMLX.Fast` too") and behind the "single source of
     truth" `EMLX.Debug` choice actually working across the module
     boundary.
5. `config/dev.exs`'s comment updated (it had predicted this exact
   extension: "When `EMLX.Fast` is implemented (task 05), rms_norm,
   layer_norm, and scaled_dot_product_attention will also be checked" — now
   true, comment reworded from future tense to present).

**Reviewer sign-off (round 1, blocker found and fixed).** A fresh reviewer
subagent (fed only the Acceptance criteria + concrete outputs, no
reasoning) found one real blocker: the acceptance criteria require
`:enable_bounds_check` to "correctly raise on violation when enabled," but
the first pass only built functional (raise-on-violation) coverage for
`:detect_non_finites`, leaving `:enable_bounds_check` completely
unverified by any test — a gap the stage doc itself had already flagged
existed pre-stage but then only partially closed. **Fixed**: extended the
`EMLX_DEBUG_FLAGS=1` opt-in toggle to flip both flags, added a `take`
out-of-bounds-index functional test, and fixed an Elixir 1.18
type-checker warning the fix's `setup_all` guard introduced (would have
failed CI's `--warnings-as-errors`) by reading the guard's flags via
`Application.get_env/3` instead of the compile-time-constant
`compile_env/3` attributes. Re-verified: 4/4 functional tests pass with
`EMLX_DEBUG_FLAGS=1`; full suite (2572 passed, 5 excluded) and
`mix test --warnings-as-errors` (no *new* warnings — the sole remaining
failure is the pre-existing, unrelated `Nx.reflect/2` deprecation, present
on the pristine tree too) both clean with flags off.

**Reviewer sign-off (round 2, clean).** A fresh reviewer subagent (no
`resume`, same clean-context discipline, fed only the acceptance criteria +
the round-1 fix's outcome artifacts) independently re-ran the functional
suite with `EMLX_DEBUG_FLAGS=1` (4/4 pass, including the new `take`
bounds-check test), the full suite with flags off (2572 passed, 5
excluded, exact match), `mix test --warnings-as-errors` (aborts only on
the pre-existing unrelated `Nx.reflect/2` deprecation, confirming the
`Application.get_env/3` fix left no new warning), and `mix format
--check-formatted`. Verdict: **pass**, no blockers remaining.
