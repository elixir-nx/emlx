# Stage 34 — Investigation: `native` (`EMLXAxon.TextGeneration`) throughput regression

Status: not started. Named by the user directly off two `validate_qwen3.exs`
snapshots collected during [`32b-emlx-metadata-custom-grad`](32b-emlx-metadata-custom-grad.md)'s
work, not (yet) root-caused. `bb+rewrite` is now faster than it has ever
been, but `native` — historically the fastest path by a wide margin — looks
regressed, and now sits *behind* `bb+rewrite`, which is backwards from every
prior benchmark in this plan.

## Symptom

Two `validate_qwen3.exs` runs, same benchmark script/model
(Qwen3-0.6B-MLX-4bit, 60 max_new_tokens), different points in time:

| snapshot | `bb base` | `bb+rewrite` | `native` | `bb+rewrite`/`bb base` | `native`/`bb base` | `native`/`bb+rewrite` |
| -------- | --------- | ------------ | -------- | ---------------------- | ------------------- | ---------------------- |
| older (`terminals/1.txt:1014-1024`) | 7.3 tok/s | 29.7 tok/s | **81.7 tok/s** | 4.07× | 11.19× | **2.75×** |
| newer (`terminals/37.txt:627-638`)  | 66.2 tok/s | 93.6 tok/s | **65.4 tok/s** | 1.41× | 0.99× | **0.7×** |

Two things changed, not one:

1. **`bb base` got ~9× faster** (7.3 → 66.2 tok/s) and **`bb+rewrite` got
   ~3× faster** (29.7 → 93.6 tok/s) — expected, this plan's whole point:
   Stage 25 (quantized-dot compiler fix), Stage 31/32/32b (runtime-call
   split points + dispatch caching) all specifically targeted `bb base`/
   `bb+rewrite`'s compiled (`compiler: EMLX`) path.
2. **`native` got ~20% *slower*** (81.7 → 65.4 tok/s) — unexpected and
   backwards. `native` (`EMLXAxon.TextGeneration`, `emlx_axon/lib/emlx_axon/qwen3/{model,attention,layers}.ex`)
   is a hand-written forward pass using `Nx.Defn.Evaluator` (**not**
   `compiler: EMLX`) per its own moduledoc — see
   `emlx_axon/lib/emlx_axon/qwen3/model.ex`'s "Defn / JIT strategy" section.
   None of Stage 31/32/32b's `compiler: EMLX`-specific work (split points,
   the `dispatch_key` ETS cache, `Nx.Defn.Graph.split`) should even run for
   this path, yet it's the one that regressed. Old absolute number (81.7)
   closely matches Stage 11's original Results table (`native`: 62.6–71.4
   tok/s) and Stage 30's canary baseline — i.e. the "older" snapshot may
   itself predate several stages, not just the most recent ones; the actual
   regression window is wider than "since Stage 32b" and needs bisecting,
   not assumed.

## Already ruled out

- **Not `git blame`-able to `emlx_axon`'s own model code.** `git log --
  emlx_axon/lib/emlx_axon/qwen3/{model,attention,layers}.ex` shows no
  commits since `ad17016` (Stage-19-era "Support dense Qwen3 generation in
  EMLXAxon") — the native forward pass itself hasn't changed across this
  entire regression window. If it regressed, the cause is in something it
  *calls into* (`EMLX.Fast`, `EMLX.Backend`, `EMLX.Quantization`, the NIF
  layer) or in benchmark-harness/environment noise, not in
  `EMLXAxon.Qwen3.*` itself.
- **Not `Nx.Defn.Evaluator`-side overhead from Stage 32b's `:__EMLX__`/
  `custom_grad` metadata wrapping.** Verified by reading
  `deps/nx/nx/lib/nx/defn/evaluator.ex`'s `compute_cache/3`: its `:metadata`
  clause (`compute_cache(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr,
  _meta]}}, state, cache)`) recurses straight into the wrapped `expr` and
  **adds no cache entry for the metadata node itself** — this runs once per
  `precompile` (i.e. once per unique input shape, cached by
  `Nx.Defn.Compiler.__jit__`), not once per token. So the two-layer
  `custom_grad(emlx_metadata(runtime_call(...)))` wrapping `EMLX.Fast.*`
  now builds is fully elided by `Nx.Defn.Evaluator` at trace-cache-build
  time, with **zero incremental per-call evaluation cost** — this specific,
  most-obvious-looking suspect (since it's exactly what Stage 32b touched)
  is not it. Don't re-litigate this without new evidence.
- **Not the `:detect_non_finites`/`:enable_bounds_check` debug flags** —
  confirmed `false` by default (`emlx/lib/emlx/debug.ex`'s
  `assert_no_nan_inf!/2` compiles to a bare `nil`, no `EMLX.eval` call, no
  atom reference) and only ever flipped on inside `config_env() == :test`
  behind an opt-in `EMLX_DEBUG_FLAGS=1` env var (`emlx/config/config.exs`)
  — `validate_qwen3.exs` runs under `mix run`, not `mix test`, so these are
  compiled out.

## Likely-culprit candidates (unconfirmed — bisect, don't assume)

Every commit that touched `EMLX.Fast`/`EMLX.Backend`/`EMLX.Quantization`/
the NIF layer since whenever the "older" snapshot was actually taken is a
candidate, since `native` calls all three modules directly and eagerly
(`EMLX.Fast.rms_norm`/`EMLX.Fast.swiglu` in `qwen3/model.ex`, quantized
`Nx.dot` per that module's own moduledoc). In rough chronological order,
starting from the plan's own stage list:

- Stage 22 (SDPA attention sinks + microscaled quantization) — new
  `OPTIONAL_TENSOR_PARAM` NIF macro, new opcodes; check for any added
  per-call overhead on the *non*-sinks/non-microscaled path that `native`
  actually exercises (Qwen3-0.6B doesn't use attention sinks or microscaled
  quant).
- Stage 25 (quantized-dot full fix) — call-time program specialization;
  `native`'s moduledoc says it does its *own* quantized `Nx.dot` handling
  bypassing `compiler: EMLX` specifically because of a documented
  incompatibility — confirm Stage 25's changes didn't touch the eager
  quantization dispatch path `native` actually calls.
- Stage 26 (fine NIF refactor) — converted all 15 `emlx_fast.cpp` NIFs
  (i.e. exactly the ops `native` calls most: `rms_norm`, `swiglu`, `sdpa*`,
  `rope*`) to the `fine` library's marshalling. Stage 26's own Results claim
  "no measurable perf regression (micro-benchmarked `git stash`
  before/after)" but that was a micro-benchmark of the NIF layer in
  isolation, not a full `validate_qwen3.exs` run — worth re-checking against
  the real benchmark now that a regression is suspected in exactly the code
  path this stage rewrote.
- Stage 32b itself — even though the `custom_grad`/`:__EMLX__` metadata
  *evaluation* cost is ruled out above, Stage 32b also changed what
  `EMLX.Fast.*`'s **eager** branch does when called directly (i.e. from
  `Nx.Defn.Evaluator`, not `compiler: EMLX`): confirm the eager
  `*_callback/2` functions `native` ultimately hits are byte-for-byte
  unchanged from before Stage 32b (they should be — Stage 32b only touched
  the `traced?` branch's *construction*, not the eager branch — but this
  needs an explicit diff check, not an assumption, since it's the most
  recent change to the file `native`'s hot path depends on).
- **Benchmark-harness/environment noise** — the two snapshots were not
  collected back-to-back under controlled conditions (different session,
  possibly different machine thermal state, different `mix` build
  artifacts). `bb+rewrite`'s newer numbers show real variance too
  (`min/max=90.1/93.2` old vs the terminal-37 snapshot's own run-to-run
  spread) — rule this out explicitly with several repeated, back-to-back
  `native`-only runs before trusting a single 81.7-vs-65.4 comparison as
  the true delta.

## Procedure

1. **Control for noise first.** Run `emlx_axon/bench/validate_qwen3.exs`
   with `native` only (comment out `bb base`/`bb+rewrite` or add a
   fast-path flag) 5+ times back-to-back, same machine, same session, to
   get a real current baseline with a confidence interval — don't compare
   against the old terminal snapshot's single run directly.
2. **Bisect.** Identify the actual commit range between whatever state
   produced 81.7 tok/s and `HEAD`. If that commit isn't identifiable from
   history/terminal timestamps, treat Stage 19/`ad17016` (last commit
   touching `EMLXAxon.Qwen3.*` itself) through `HEAD` as the outer bound and
   bisect within it, re-running the controlled `native`-only benchmark
   (step 1) at each candidate commit — mirrors Stage 11's bisection
   procedure exactly (same tool, same rationale: `native`'s own code hasn't
   changed, so the regression is a dependency, and `git bisect` finds it
   faster than guessing from a diff).
3. **Once localized to a commit**, diff it specifically for anything that
   changes the **eager** (non-`compiler: EMLX`) call path of whatever op(s)
   `native` calls per decode step: `EMLX.Fast.rms_norm/3`,
   `EMLX.Fast.swiglu/2`, `EMLX.Fast.scaled_dot_product_attention*`,
   `EMLX.Fast.rope*`, the quantized `Nx.dot`/`EMLX.Quantization` path, and
   any shared `EMLX.Backend` primitive all of the above route through.
4. **Fix + guard.** Land the fix at the identified seam. Add a permanent,
   CI-sized regression *benchmark* assertion (not just a numeric
   `assert_all_close`, which can't catch a throughput regression) — e.g. a
   documented acceptable floor for `native` tok/s on this model/token-count
   combination, checked manually before merging future stages that touch
   `EMLX.Fast`/`EMLX.Backend`/`EMLX.Quantization`'s eager paths, since none
   of Stage 11/22/25/26/32b's own acceptance criteria required re-checking
   `native` specifically (each focused on the compiled/`bb+rewrite` path
   instead) — that blind spot is exactly how this regression shipped
   unnoticed across several stages.

## Acceptance

- `native` throughput restored to at least its pre-regression level (~80+
  tok/s on this model/token-count combination, matching the historical
  Stage 11/30 baseline), confirmed via several controlled back-to-back
  runs, not a single sample.
- `native` is once again at least as fast as `bb+rewrite` (it should never
  be slower — it's the hand-written, no-Axon-graph-overhead path; regressing
  below `bb+rewrite` is itself a signal something is structurally wrong,
  independent of the absolute tok/s number).
- Root cause documented here: which commit/stage, which function, why.
- A repeatable benchmark-floor check added so future stages touching
  `EMLX.Fast`/`EMLX.Backend`/`EMLX.Quantization` re-verify `native`
  explicitly, not just `bb base`/`bb+rewrite`.
