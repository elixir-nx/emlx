# Stage 32b — `:__EMLX__` metadata for `EMLX.Fast.*`, `custom_grad` for its backward pass

Status: **done.** Named by, and supersedes, [`32a-inline-runtime-call`](32a-inline-runtime-call.md)
per user directive after that stage's generalized "any `runtime_call` becomes
an in-graph opcode" approach hit an unresolved race condition (see 32a's
Results). This stage narrows the charter to exactly the production case that
needed fast, non-split, in-graph execution — `EMLX.Fast.*`'s fused kernels —
and drops the general-purpose `:host_callback` opcode entirely. Every
*other* `runtime_call` (e.g. `EMLXAxon.native_kv_attn_callback/2`) goes back
to Stage 31's `Nx.Defn.Graph.split` behavior.

## Why this shape

`Nx.Defn.metadata/2` already gives any `Nx.Defn.Expr` an opaque wrapper node
carrying an arbitrary map — this is exactly what `custom_grad`/`stop_grad`/
hooks already use, and it needs no upstream Nx changes. `EMLX.Fast`'s
`deftransform`s attach a `:__EMLX__` key to that map (`%{op: opcode,
operands: [...], attrs: [...]}` — the native opcode name, its operand
tensors, and int-encoded attrs) naming the fused instruction directly.
`EMLX.Native.Expr`'s `:metadata` `expand_node` clause recognizes this key
and lowers straight to that native op, ignoring the wrapped `_inner`
entirely — no graph split, no host round-trip, no new C++ machinery. Every
*other* `:metadata` node (custom_grad, stop_grad, hooks, ...) falls through
to a second, generic `:metadata` clause that treats `_inner` as a pure
value pass-through (aliases its already-lowered ref) — the mechanism Nx
itself already relies on for those constructs.

The wrapped `_inner` is an ordinary `Nx.runtime_call/4` invoking the same
eager NIF-backed callback used by the eager path (e.g. `rms_norm_callback/2`)
— never evaluated by EMLX's own compiler (discarded in favor of the
`:__EMLX__` payload), but free to build (a single lightweight node, not a
full composite sub-expression) and exists so:

1. The operand tensors are ordinary reachable dependencies for
   `EMLX.Defn.Tree.post_order/1` to visit (a new `visit_scope_deps` clause
   in `emlx/lib/emlx/defn/tree.ex` skips `_inner`, visits `operands` only).
2. Any *other* `Nx.Defn.Compiler` (`Nx.Defn.Evaluator`, or `Nx.Defn.Grad`
   absent a `custom_grad` override) still gets an exact fallback: the real
   NIF against concrete tensors, not a slower plain-`Nx` approximation.

## Two false starts on the way here, both fixed

1. **First attempt wrapped a full plain-`Nx` composite reference formula as
   `_inner`** (differentiable by construction, no `custom_grad` needed) —
   correct, but tracing that formula on every `EMLX.Fast.*` call inside a
   host-driven decode loop (`run_while_loop`) re-built large sub-expression
   graphs every step, tanking `bb+rewrite` from ~63 tok/s to ~6 tok/s (10×
   regression). **User directive: revert to `Nx.runtime_call` for `_inner`**
   (cheap to build, one node) and defer differentiability to `custom_grad`
   instead of a differentiable-by-construction `_inner`.
2. **Reverting to `Nx.runtime_call` exposed a real `Nx.Defn.Graph.split`
   bug**: `Nx.Defn.Graph.split/2`'s generic traversal (unlike EMLX's own
   `Tree.post_order/1`) *does* walk into a `:metadata` node's `_inner`, so
   it treated the reference-formula's embedded `runtime_call` as a spurious
   split point when a `while` carry crossed an `EMLX.Fast.*` call boundary
   (`while_after_runtime_call` test). **Fix**: `collect_metadata_inner_ids/1`
   pre-scans the expr for every `runtime_call` embedded as `_inner` of a
   `:__EMLX__` node and passes those ids as `hidden_ids` into
   `split_on_split_point/2`, which now returns `:none` for a `runtime_call`
   whose id is hidden even though `split_point?/1` still (correctly) says
   `true` for every `runtime_call` node in isolation.
3. **A second, independent performance regression surfaced after the
   `hidden_ids` fix**: `dispatch_key/3`'s structural-signature computation
   was being re-run from scratch on every `build_eval_fn` call inside
   `run_while_loop` (once per decode step), even for a structurally
   identical `Nx.Defn.Expr`. **Fix**: a process-lifetime ETS memoization
   cache (`@dispatch_key_by_id_table`) keyed by a lightweight
   `expr_id_fingerprint/1` (the output expr's own node ids, not a full
   structural walk) short-circuits `compute_dispatch_key/3` on a cache hit.
   Recovered `bb+rewrite` to ~62 tok/s, on par with `bb base`.

## Gradient support

`Nx.Defn.grad`/`Nx.Defn.Grad.transform` can't differentiate through a bare
`Nx.runtime_call` (not autodiff-aware) or through the opaque `:__EMLX__`
metadata (no gradient rule registered for an arbitrary map key). Each
`EMLX.Fast.*` op's traced-path result is therefore wrapped one layer further
in `Nx.Defn.Kernel.custom_grad/3` (`with_reference_grad/3` in `fast.ex`):

```elixir
fused = emlx_metadata(Nx.runtime_call(...), :fast_rms_norm, [x, weight], [...])
with_reference_grad(fused, [x, weight], fn x, weight -> rms_norm_reference(x, weight, eps) end)
```

`custom_grad/3` is itself implemented as `Nx.Defn.Expr.metadata(expr, %{custom_grad:
{inputs, fun}, ...})` — the *same* underlying `:metadata` op, with a
*different* map key than `:__EMLX__`. Stacking them (`custom_grad`'s wrapper
outermost, `:__EMLX__`'s wrapper as its `_inner`) composes cleanly with no
new lowering code: `Nx.Defn.Grad`'s own `:metadata`/`custom_grad` clause
short-circuits the backward pass at the outer node (never looks past it into
`_inner`), while EMLX's forward-value lowering treats the outer node via its
generic pass-through clause, which recurses into `_inner` normally and hits
the `:__EMLX__`-specific clause there, unaffected by the extra wrapper.

`with_reference_grad/3`'s `fun` (the `custom_grad` callback, called with the
upstream cotangent `g`) reuses `Nx.Defn.Grad.transform/3` directly (not the
outer `Nx.Defn.grad/2`, which wraps a nested `jit_apply` — invalid to call
from inside an already-tracing `deftransform`) on the standard VJP-via-scalar-grad
trick: for `y = reference_fn(inputs)` and cotangent `g` (same shape as `y`),
the VJP w.r.t. `inputs` is `grad(inputs, sum(g * y))`. This differentiates
each op's existing plain-`Nx` `*_reference/N` formula (kept from the first
false start, previously dead code) instead of hand-deriving a backward
formula per op. All eleven `EMLX.Fast.*` traced call sites (`rms_norm`,
`layer_norm` ×2, `rope`/`rope_with_positions`/`rope_with_freqs`, `swiglu`,
and all `scaled_dot_product_attention*` sinks/masked/causal/key-masked
combinations) are wired this way.

## Results

- `mix test`: 2671/2671 passing (no regressions from the metadata/custom_grad
  layering, the C++ comment cleanup, or the bench-script deletions).
- Numerically verified `Nx.Defn.grad` through `EMLX.Fast.rms_norm` and
  `EMLX.Fast.swiglu` against hand-written pure-`Nx` equivalents (exact
  match, `all_close` within `1.0e-4`); `EMLX.Fast.rope`'s gradient checked
  for finiteness and correctness in the zero-offset (identity-rotation)
  case.
- `bench/validate_qwen3.exs`: `bb+rewrite` ~92 tok/s vs `bb base` ~65 tok/s
  (1.4×) — the extra `custom_grad` metadata wrapper adds no measurable
  forward-pass overhead (it's a zero-instruction pass-through at lowering
  time).
- Bare (unrecognized) `Nx.runtime_call` correctly still forces a
  `Nx.Defn.Graph.split` per Stage 31 — re-verified via the existing
  `:stage31`/`:stage32` test tags (8/8 passing) and a structural audit of
  `split_point?/1`/`contains_split_point?/1`/`bare_runtime_call?/1`/
  `build_runtime_call_base_eval_fn/2`/`build_split_chain_eval_fn/2`/
  `split_on_split_point/2`/`collect_metadata_inner_ids/1` in `lib/emlx.ex`.
- Dead `:host_callback` production code from Stage 32a fully absent from
  `lib/`/`c_src/`'s implementation (confirmed via full-codebase grep for
  `host_callback`/`HostCallback`/`host_round_trip`/`dispatch_host_callback`
  and friends). Remaining artifacts cleaned up: the three broken bench
  scripts that called since-removed NIFs (`bench/host_callback_opcode.exs`,
  `bench/host_callback_multi_caller.exs`, `bench/spike32a_host_callback.exs`)
  deleted; stale comments in `c_src/emlx_nif.cpp` (`eval_many`'s doc
  comment) and `c_src/emlx_async.hpp` (`g_current_caller_pid_ptr`'s doc
  comment) updated to stop referencing the removed `host_callback`
  primitive (both are otherwise-harmless, still-compiled leftovers —
  `eval_many` is a generic unused-but-callable multi-ref eval NIF;
  `g_current_caller_pid_ptr` is set by `async_dispatch` on every dispatched
  call but currently has no reader — left in place as low-risk, not worth a
  native rebuild+re-verification cycle to excise for a doc-only cleanup
  pass); a misleading test comment in
  `test/emlx/native/expr_test.exs` (claimed `split_point?/1` only flags
  *unrecognized* `runtime_call`s — it flags all of them; recognition
  happens one layer up, in `EMLX.Defn.Tree.post_order/1`'s `:__EMLX__`
  skip) corrected.
