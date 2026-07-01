# Stage 16 — `EXPR_NODES.md` accuracy audit (`fun` / `optional` / `from_binary`)

Status: done.

## Why this stage exists

`EXPR_NODES.md` marks `fun`, `optional`, and the "`from_binary` / constant
materialization" line as `[ ]` (not lowered), which reads as three more raise
paths that can fire the `Nx.Defn.Evaluator` fallback. Planning-time
investigation for this round found all three are already non-issues in the
current vendored Nx fork (`emlx/deps/nx`) — the doc is stale, not the code:

- **`fun`** (`(params, expr, mfa)`): the only two call sites that ever produce
  a `:fun` node are `apply_fun/4` inside `Nx.Defn.Expr.reduce/5` and
  `window_reduce/…` (`nx/lib/nx/defn/expr.ex:996,1017`). Both already extract
  `fun.data.args` and re-lower the body directly (Stages 12–13's static
  unroll). `EMLX.Native.Expr`'s `expand_node` clause for `op: :fun`
  (`expr.ex:1768`) is a documented no-op precisely because a standalone
  `:fun` node is unreachable — it is always consumed as a `reduce`/
  `window_reduce` operand before the generic dispatch ever sees it.
- **`optional`**: no `:optional` op tag exists anywhere in `emlx/deps/nx` —
  dead node type from an older Nx version this fork no longer has.
- **`from_binary`**: `Nx.Defn.Expr.from_binary/3` immediately materializes via
  `Nx.BinaryBackend.from_binary/3` and re-wraps the result as a `constant`/
  `tensor` expr (`nx/lib/nx/defn/expr.ex:825-826`) — there is no distinct
  `:from_binary` Expr op reaching the lowerer; it's already covered by the
  existing `constant`/`tensor` handling (`[x]`).

## Procedure

1. Confirm each finding above still holds against the exact vendored Nx
   commit (re-grep `emlx/deps/nx` — this doc's grep results are a snapshot).
2. Add a small regression test that traces a `defn` using a custom-fun
   `Nx.Defn.Expr.reduce`/`window_reduce` and asserts the standalone `:fun`
   clause path is exercised as a no-op (not a raise), pinning the invariant
   the doc claim rests on.
3. Flip the `EXPR_NODES.md` lines for `fun` (section A) and `optional`/
   `from_binary` (sections A/I) to `[x]`, with inline rationale worded
   precisely as "unreachable / already-subsumed", not "lowered" — so future
   readers don't reintroduce the belief that these are real gaps requiring
   new lowering code.
4. Add a one-line "re-audit on Nx version bump" note next to `optional`/
   `from_binary`, since their status is a property of the vendored Nx fork's
   node set, not of anything EMLX owns.

## Acceptance

- `EXPR_NODES.md`'s only remaining `[ ]`/`[~]` boxes are `attach_token`/
  `token` and the `block` while-in-`default_expr` boundary (Stages 17–18).
- No `expr.ex` code changes are expected. If step 1's re-grep surfaces a real
  reachable path (e.g. a future Nx bump reintroduces `optional`, or produces
  `:fun` outside a reduce/window_reduce operand), pivot this stage to a real
  fix instead of a doc flip — do not flip the checkbox in that case.

## Results

Re-grep against the exact vendored Nx fork confirmed every line-number
citation in this doc's original finding, with no third producer of `:fun`:
`apply_fun/4` (`nx/lib/nx/defn/expr.ex:1376`) is the sole caller of the
private `fun/4` node constructor, and its only two call sites remain
`reduce/5` (`expr.ex:996`) and `window_reduce/6` (`expr.ex:1017`); a literal
`grep -r ":optional"` across `emlx/deps/nx` does surface unrelated hits
(`Nx.Backend.behaviour_info(:optional_callbacks)`, `ex_doc` dependency
source), but none is an `Expr` op tag — `op: :optional` has zero matches, so
the "dead node type" claim holds; `from_binary/3`
(`expr.ex:825-826`) still resolves through `to_expr` into a `constant`/
`tensor` node. No `expr.ex` code changes were needed — confirmed as a pure
doc-audit stage, per Acceptance.

Added two regression tests (`emlx/test/emlx/native/expr_test.exs`, describe
"Stage 16 — :fun node unreachability (doc audit)", tag `:stage16`) that
lower a custom-fun `Nx.reduce` and a custom-fun `Nx.window_reduce` and assert
the resulting `EMLX.Native.Expr` program's instruction list never contains a
`:fun` op — pinning that the standalone-`:fun` `expand_node` no-op clause
(`expr.ex:1768`) is exercised on a real program, not just theoretically
unreachable. Full suite green (253 passed, 0 failures).

Flipped `EXPR_NODES.md`: `fun` (section A) and `optional`/`from_binary`
(sections A/I) → `[x]`, worded "unreachable / already-subsumed" (not
"lowered"); added a "re-audit on Nx version bump" note next to `optional`/
`from_binary` since their status is a property of the vendored Nx fork's
node set. Section A's only remaining `[ ]`/`[~]` boxes are now exactly
`attach_token`/`token` (Stage 18) and `block`'s while-in-`default_expr`
boundary (Stage 17), matching Acceptance.
