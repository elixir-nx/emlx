# Stage 18 — `token` / `attach_token` native lowering (contingent)

Status: done.

## Why this stage might exist

`token`/`attach_token` (surfaced via `Nx.Defn.Kernel.hook/2,3`) are the last
node types with *no* lowering path at all — they raise unconditionally today,
by design (§ EXPR_NODES.md: "imply host side effects → not lowerable to a
pure replay"). They exist to run a host-side Elixir callback mid-graph
(debug/logging hooks), which cannot execute inside a single compiled MLX
program — there is no mechanism for C++ to call back into the BEAM
mid-replay, and the project's worker model deliberately never blocks a NIF on
a BEAM operation.

The `while` precedent (Stage 08) is the relevant prior art: rather than
"lower everything in one NIF call, or fall back," structurally split the
expression around the thing that can't be lowered, drive the boundary from
Elixir, and recompile each side natively. The question this stage answers:
does the same `Nx.Defn.Graph.split`-style approach generalize from `while` to
`token`/`attach_token` boundaries — native segment, host-side hook fire,
native segment — never touching `Nx.Defn.Evaluator`?

## Procedure

1. Confirm whether `Nx.Defn.Graph.split` (or an equivalent primitive) can
   split on non-`while` node types today. If not, spike whether
   `attach_token` can be treated as a synthetic split point the same way
   `while` is.
2. Check `Nx.Defn.Kernel.hook/2,3`'s exact runtime semantics (does the hook's
   return value feed back into the graph, or is it fire-and-forget observing
   a value?) — this determines whether the split needs to thread a value
   back in or can simply observe-and-continue.
3. **If splittable**: implement the split + host-side hook dispatch
   (recompiling each side via this compiler, mirroring Stage 08's
   `build_while_base_eval_fn` pattern), and equivalence-test a `defn` with a
   mid-graph hook (native vs eager), confirming the hook fires with the
   correct value and the surrounding graph still replays as one or two NIF
   calls (not per-node).
4. **If NOT splittable** (e.g. `Graph.split`'s machinery is `while`-specific
   in a way that doesn't generalize, or hooks can appear inside sub-scopes in
   ways that make splitting unsound): stop, and document the no-go with the
   same rigor as Stages 12/14 — a concrete blocking finding or measurement,
   not a hunch. Hand back an explicit decision, don't default silently to
   either option:
   - (a) accept `token`/`attach_token` as the one permanent, structurally
     necessary hard-raise (a genuine host side effect, not a compiler gap),
     revising Stage 19's "zero fallback, no exceptions" scope to name this
     one construct explicitly; or
   - (b) a narrowly-scoped fallback that routes *only* the sub-graph rooted
     at a hook through the Evaluator (not the whole `defn`) — a middle
     ground the current `try_native_compile` doesn't offer today, and a
     materially different (smaller, bounded) risk than today's whole-defn
     fallback.

## Acceptance

Either: `token`/`attach_token` lower natively with equivalence tests and
`EXPR_NODES.md`'s line flips to `[x]`; or: a documented, measurement-backed
no-go with an explicit (a)/(b) recommendation handed back for a decision
before Stage 19 proceeds.

**Met**, with a narrowed scope: hooks lower natively and `EXPR_NODES.md`'s
line flips to `[x]`, except for a hook reachable only from inside a `cond`
branch, which raises deliberately (see Results) — a correctness-driven carve-
out, not a coverage gap.

## Results

**Step 1 answer: no, `Nx.Defn.Graph.split` is not the right tool here — hooks
are not control flow.** Traced `Nx.Defn.Kernel.hook/3`'s desugaring
(`nx/lib/nx/defn/kernel.ex`, `expr.ex`, `token.ex`) and `Nx.Defn.Evaluator`'s
`:token`/`:attach_token` clauses: `attach_token(token, expr)`'s runtime value
*is* `expr` unchanged (the token's own eval result is discarded), and a
hook's callback return value is never read back into the graph — fire-and-
forget, not a runtime-value-dependent branch like `while`. Layer A
(`EMLX.Defn.Tree.post_order/1`) already treats hook exprs as ordinary
same-scope nodes (confirmed via `Nx.Defn.Tree.apply_args/4`'s `:token`
clause, which recurses into each hook's `expr` unconditionally) — unlike
`while`/`fun`/`block`, `token`/`attach_token` are not scope-boundary nodes.
Empirically confirmed `Nx.Defn.Graph.split` itself *can* split on
`:attach_token` today (its splitting engine is generic, not `while`-specific)
— but doing so doesn't remove the need to lower `:token`/`:attach_token` in
Layer B, and buys nothing for a construct with no control-flow dependency.

**Design implemented instead (advisor-approved, see chat): extra-output
augmentation, zero `Graph.split`, zero host round-trip.** `:attach_token`
lowers as a zero-instruction passthrough (its ref aliases its wrapped expr's
already-lowered ref). `:token` contributes zero instructions but records
each hook's `{name, callback, template, refs}` into a new `hooks` field on
`EMLX.Native.Expr.t/0`; `to_wire/1` appends the hook refs after the real
outputs, so the *same* single `eval_program` NIF call returns both. Wait —
that's still one NIF call per `defn` invocation (better than `while`'s N
calls per loop, since no runtime branching is involved). `EMLX.__compile__`
slices the returned refs back apart and fires each callback host-side, once,
right after the call returns. A hook with neither a trace-time callback nor
a runtime override is skipped (no instruction, no output), mirroring
`Nx.Defn.Evaluator`'s skip-if-unhandled rule — verified with an
unreferenced-value hook and a name-only (no-fn) hook, both agreeing with
`Evaluator` (neither fires).

**Cond-branch hooks hard-raise (advisor-flagged correctness issue, not a
coverage gap).** EMLX's `cond` lowers by unconditionally evaluating every
branch and `:select`-ing the result (Stage 08) — a hook nested in one branch
would fire on *every* call regardless of which branch is actually taken, a
genuine behavior divergence from `Nx.Defn.Evaluator` (which only fires the
selected branch's hook; `Nx.Defn.Tree`'s own `scope_ids/1` docs note "cond"s
need special handling for exactly this reason). Detection reuses
`Nx.Defn.Tree.scope_ids/1` directly (an existing, already-tested upstream
primitive) rather than a hand-rolled tree walk: empirically confirmed it
returns `false` for both the `:token` and `:attach_token` node ids of a
cond-branch-local hook, and `true` for a top-level hook. `lower/2` raises a
clear, permanent `ArgumentError` (a message that does **not** match `"does
not yet lower op"`, so `try_native_compile`'s Evaluator-fallback rescue does
not silently swallow it as "coverage gap, coming soon").

**Reconciled with, and narrowed, one advisor recommendation.** The advisor's
first-pass recommendation hard-raised for a hook inside *either* `cond` or
`while`. Primary-source evidence contradicted the `while` half: a `while`
body is always recompiled by re-entering this same compiler as its own
fresh top-level scope (Stage 08's existing `Nx.Defn.jit`-based recursion), so
a hook inside a `while` body is "top-level" for that inner compile and fires
once per host loop iteration — empirically verified to match `Evaluator`
exactly (`[iter: 3, iter: 5, iter: 6]` on both sides for a 3-iteration
countdown loop). Only the `cond` case raises.

**Found and fixed a real `Nx.Defn.Graph` bug along the way (same pattern as
Stages 11/17 — a bug only surfaces by executing, not by reading).** A hook
*before and after* a non-bare `while` (the while has surrounding work on
both sides) routes through `Nx.Defn.Graph.split`'s multi-stage chain
(`EMLX.build_while_chain_eval_fn`, Stage 08) even though the hook itself
needs no splitting. This crashed the NIF with `"vector in NIF.eval_program/2"`
(a C++ `std::length_error`, i.e. an input-count mismatch) — root-caused to
`Nx.Defn.Graph`'s `do_rewrite_subtree/3` (the per-stage parameter-remapping
pass) having no clause for `:token`: its generic fallback tries to recurse
into `args = [%Nx.Defn.Token{}]` via the list-traversal clause, which
silently leaves a struct that's neither `%Nx.Tensor{}` nor a list
*untouched* — so a hook payload depending on a stage-boundary-hoisted value
kept its stale, pre-remap parameter position (e.g. position 2 instead of the
correct 0), while the same value used *outside* the hook got correctly
renumbered. The compiled stage then declared a wire arity that the actual
call-time argument count didn't satisfy. Fixed by adding a `:token` clause
to `do_rewrite_subtree/3` that recurses into each hook's `expr` (mirroring
`Nx.Defn.Tree.apply_args/4`'s existing `:token` clause and the adjacent
`:runtime_call` clause's comment pattern). Applied in both `~/coding/nx/nx`
(the fork this project's fixes flow through upstream, per Stage 11
precedent — left uncommitted, pending review/push per the "never autocommit"
rule) and `emlx/deps/nx` (this repo's pinned `github:` checkout, so this
session's test suite exercises the fix now). Full suite: 2555 → 2562 passed
(825 doctests, 1737 tests), 0 failures, 0 regressions from the `nx`-side
change.

**Descoped, not a gap:** a runtime `hooks:` jit-option override
(`Nx.Defn.jit(fun, hooks: %{name => fn})`, which lets a *caller* supply a
callback for a name-only hook at call time) is not threaded through the
native path — `EMLX.__compile__/4` already drops `rest_opts` entirely on the
native-compile branch today (pre-existing, not new), so this is new plumbing
outside this stage's charter, not a correctness bug. A name-only hook with
no override is a silent no-op today, matching `Evaluator`'s behavior for the
same (unhandled) case.

**Reviewer caught a real false-positive/false-negative pair in the cond-branch
guard, fixed before closing the stage.** A first reviewer pass (fed only the
diffs + test output, no reasoning) reproduced a false positive: a hook nested
directly in a custom-fun `reduce` body (no `cond` anywhere) raised the
cond-branch message. Root cause: `Nx.Defn.Tree.scope_ids/1` walks in
`:scope` mode, which *by design* never descends into a `:fun`/`:while` body
(that's the scope boundary) — so `lower/2`'s one-shot `top_scope_ids` (built
from `scope_ids(output)` on the pristine top-level tree) simply never sees
ids inside a `reduce`/`window_reduce` custom-fun body or a Stage-17
statically-unrolled nested `while` (both lowered *inline*, within the same
`lower/2` call, via `lower_fun_body/3` / `lower_tuple_body/3` — unlike a bare
`while` body, which re-enters `lower/2` fresh as its own top-level scope).
Both are "always executes in full" shapes like `while`, not conditionally-
executed like `cond`, so this was a real bug, not a documentation gap. Fixed
by extending `top_scope_ids` with a fresh `scope_ids` pass over each such
body right before lowering it inline (`merge_scope_ids/2`) — which still
correctly excludes a `cond` nested a level *deeper* inside that body, so a
genuinely cond-branch-local hook there still raises (regression-tested).
Investigating that false positive surfaced a second, independent bug via
direct execution (not visible from reading the diff): `lower_fun_body/3` and
`lower_tuple_body/3` each reconstruct their returned `state` from an explicit
field allowlist (`instructions`/`captures`/`constants`/`inputs`) that
predates this stage and simply never included `hooks` — so *any* hook fired
from inside a `reduce`/`window_reduce`/unrolled-`while` body was silently
dropped (zero crash, wrong answer: the reduce's numeric result was correct,
the hook just never fired). Fixed by adding `hooks: inner_state.hooks` to
both return sites.

**Verification.** 9 tests (`expr_test.exs`, `@tag :stage18`): top-level hook
fires once with the correct value; an unreferenced-value hook and a
name-only hook both agree with `Evaluator` (no-op); a tuple-payload hook
fires with the matching container shape; a cond-branch hook raises with the
documented message; a while-body hook matches `Evaluator` iteration-by-
iteration; a hook straddling a non-bare `while` (the `Graph.split`-chain
regression case) matches `Evaluator` end-to-end; a hook inside a custom-fun
`reduce` body fires once per fold step matching `Evaluator` (the
reviewer-caught regression, oracled against `Nx.BinaryBackend` per the
existing `check_reduce_equiv/3` convention — eager EMLX has no `reduce`); a
hook inside a `cond` nested inside a `reduce` body still raises. Full suite:
2564 passed (825 doctests, 1739 tests), 0 failures, 0 regressions.

**Reviewer sign-off (second pass, clean).** A fresh reviewer subagent (no
access to this reasoning, only the diff + test output) independently
re-ran both the `stage18`-tagged tests and the full suite, confirmed the
`hooks: inner_state.hooks` and `merge_scope_ids/2` fixes close the gap by
reading `Nx.Defn.Tree.scope_ids/1`/`apply_args/4` directly, and grepped the
file for the same "field-allowlist forgot a field" bug class elsewhere.
One non-blocking observation: `expand_block_via_default` (the generic
`:block` default-decomposition path) doesn't defensively call
`merge_scope_ids`, unlike the two reducer/while-unroll helpers — currently
unreachable (Nx's `:block` default bodies are library-authored
decompositions, e.g. `top_k`/`cumulative_*`/`Nx.Random.*`, never user `defn`
code, so a user `hook()` call can't structurally land inside one today), but
flagged as an asymmetry worth a defensive `merge_scope_ids` call if that
assumption ever changes. Not required for this stage's acceptance.
