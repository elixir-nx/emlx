# Stage 34 — Investigation: `native` (`EMLXAxon.TextGeneration`) throughput regression

Status: **mitigated, not closed** — see Update below. Named by the user
directly off two `validate_qwen3.exs` snapshots collected during
[`32b-emlx-metadata-custom-grad`](32b-emlx-metadata-custom-grad.md)'s work.
`bb+rewrite` is now faster than it has ever been, but `native` —
historically the fastest path by a wide margin — looked regressed, and sat
*behind* `bb+rewrite`, which is backwards from every prior benchmark in this
plan. The immediate symptom (a below-`bb+rewrite` `native` number) is
resolved for benchmarking purposes; the underlying question this stage now
exists to answer is why `compiler: EMLX` is apparently slower than
`Nx.Defn.Evaluator` for this exact hand-written workload, given `bb+rewrite`
proves `compiler: EMLX` itself is not slow on this same model.

## Update — mitigated via the benchmark script, root cause still open

User report: setting `compiler: Nx.Defn.Evaluator` explicitly in
`emlx_axon/bench/validate_qwen3.exs`'s `EMLXAxon.TextGeneration.from_mlx4bit/3`
call recovers `native` to **90+ tok/s** (better than the original 81.7
tok/s baseline), while `bb+rewrite` holds at **88+ tok/s** under
`compiler: EMLX` — i.e. the two paths are now roughly at parity again, and
`native` is no longer the slower one. The script has already been updated
this way (`git diff HEAD~3 -- emlx_axon/bench/validate_qwen3.exs` shows
exactly one line added: `compiler: Nx.Defn.Evaluator,` in the `from_mlx4bit`
call, landed in `c514502`).

**Caveat found while writing this up, worth flagging before anyone trusts
the causal story implied by that diff**: `compiler:` is not currently
consumed anywhere in `EMLXAxon.TextGeneration`'s call chain. Traced
`from_mlx4bit/3` → `serving/3` → `Generate.generate/3` →
`EMLXAxon.Qwen3.{Layers,Attention}`'s `defnp` kernels — grepped every file
in that chain for `compiler`/`:compiler` and found zero reads of that key
out of `opts` (only `Keyword.get(opts, :max_len | :sampler | :temperature |
:top_p | :profile_timing | :host_sync | ...)`-style fetches for *other*
keys). `serving/3` doesn't raise on unrecognized opts, so
`compiler: Nx.Defn.Evaluator` is silently accepted and **currently does
nothing** — the `defnp` kernels were already running under
`Nx.Defn.Evaluator` before and after this line (Nx's own hardcoded default
compiler; this repo sets no `:nx, default_defn_options` app config to
override it). So the *literal mechanism* in the diff doesn't explain the
recovered throughput. Plausible explanations, not yet distinguished:

1. **Benchmark-run variance** (thermal state, background load, warm MLX
   kernel-compile cache from a prior run in the same `mix run` session) —
   the swing size (65→90+) is within the kind of noise already seen between
   the two original snapshots (81.7→65.4) with no code change implicated at
   all.
2. **The option should be wired but isn't** — perhaps the intent was to let
   `native` opt into `compiler: EMLX` for direct comparison against
   `Evaluator`, and the option's current no-op status is itself a small bug
   worth fixing (see Procedure) so this kind of A/B test is actually
   possible from the benchmark script instead of requiring a code edit.
3. **A different, uncaptured change** — re-run `git diff` across the full
   commit (`c514502`), not just this one file, in case a sibling change in
   the same commit (e.g. to `text_generation.ex`/`generate.ex`) affects
   `native`'s real behavior and was conflated with the benchmark-script
   edit when the user summarized the fix.

**This changes the framing of the rest of this stage.** The original
"restore `native` to its old absolute number" bar is provisionally met
(90+ tok/s, better than the 81.7 baseline). The more interesting and
still-open question, now that `bb+rewrite` independently proves
`compiler: EMLX` hits 88+ tok/s on this exact model: **why would running
this same Qwen3 forward pass under `compiler: EMLX` (once that option is
actually wired to do something) be any slower than `Nx.Defn.Evaluator`, if
it's slower at all?** That's the real remaining scope — see the retitled
Procedure/Acceptance below.

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
   get a real current baseline with a confidence interval — confirm the
   90+ tok/s figure holds up and isn't itself a one-off sample, before
   trusting it as "mitigated."
2. **Wire `compiler:` for real.** `EMLXAxon.TextGeneration.serving/3`
   (`emlx_axon/lib/emlx_axon/text_generation.ex`) and
   `EMLXAxon.Qwen3.Generate.generate/3` currently drop a `:compiler` opt on
   the floor — thread it through to whatever `Nx.Defn.Compiler.__jit__`/
   `Nx.Defn.jit` call sites back the `defnp` kernels in `Layers`/
   `Attention`, defaulting to today's implicit `Nx.Defn.Evaluator` when
   unset, so the benchmark script's existing (currently inert)
   `compiler: Nx.Defn.Evaluator,` line becomes a real, working switch and
   `compiler: EMLX` becomes an actual option to A/B against for this exact
   workload — not just for `bb`/`bb+rewrite`.
3. **A/B the two compilers on `native` directly**, once wired, holding
   everything else fixed (step 1's controlled-run methodology). If
   `compiler: EMLX` is genuinely slower than `Evaluator` for this hand-
   written graph shape, profile *why* — e.g. compare instruction/dispatch
   counts, check whether `Nx.Defn.Graph.split`/the `dispatch_key` ETS cache
   (Stage 31/32/32b machinery, `compiler: EMLX`-only) is firing on any
   `runtime_call` inside this specific graph shape the way it doesn't for
   `bb+rewrite`'s (since `bb+rewrite`'s 88+ tok/s already proves
   `compiler: EMLX` isn't inherently slow on this model — the two graphs
   aren't necessarily structurally identical, e.g. `EMLXAxon.rewrite/2`'s
   output vs the hand-written `Layers`/`Attention` defnp bodies may differ
   in exactly the ways that matter here, such as which ops end up as
   unrecognized `runtime_call`s needing a graph split).
4. **If `compiler: EMLX` can be brought to parity with (or ahead of)
   `Evaluator`** for `native`, that's the real fix — `compiler: EMLX`
   should, in principle, never lose to op-by-op interpretation once its
   compile/dispatch-cache warms up, so a persistent gap here is itself an
   optimization opportunity worth chasing independent of this stage's
   original regression-hunting framing. If it's confirmed structurally
   equivalent in effort and still slower, document why and consider
   whether `Evaluator` should just be the documented, permanent choice for
   this specific hand-written path (i.e. accept the finding rather than
   force `compiler: EMLX` where it doesn't help).
5. **Fix + guard.** Land whichever of steps 3/4 applies. Add a permanent,
   CI-sized regression *benchmark* assertion (not just a numeric
   `assert_all_close`, which can't catch a throughput regression) — e.g. a
   documented acceptable floor for `native` tok/s on this model/token-count
   combination, checked manually before merging future stages that touch
   `EMLX.Fast`/`EMLX.Backend`/`EMLX.Quantization`'s eager paths or
   `emlx.ex`'s `compiler: EMLX` dispatch machinery, since none of Stage
   11/22/25/26/32b's own acceptance criteria required re-checking `native`
   specifically (each focused on the compiled/`bb+rewrite` path instead) —
   that blind spot is exactly how the original regression shipped unnoticed
   across several stages, and is also why the `compiler:` no-op above went
   unnoticed.

## Acceptance

- `native` throughput confirmed at 90+ tok/s (matching/exceeding the
  historical Stage 11/30 baseline) via several controlled back-to-back
  runs, not a single sample — closes the original regression symptom.
- `:compiler` is a real, working option on `EMLXAxon.TextGeneration.serving/3`
  (and `from_mlx4bit/3`), not a silently-ignored keyword.
- A direct `native`-under-`compiler: EMLX` vs `native`-under-`Evaluator`
  comparison exists and is documented here, with either (a) `compiler:
  EMLX` brought to parity/ahead via a concrete fix, or (b) a documented,
  understood reason it's legitimately not worth using for this specific
  hand-written graph shape.
- `native` is once again at least as fast as `bb+rewrite` regardless of
  which compiler it ends up using by default.
- A repeatable benchmark-floor check added so future stages touching
  `EMLX.Fast`/`EMLX.Backend`/`EMLX.Quantization`/`emlx.ex`'s compiler
  dispatch re-verify `native` explicitly, not just `bb base`/`bb+rewrite`.
