# Stage 18 — `token` / `attach_token` native lowering (contingent)

Status: not started — spike; may end in a no-go like Stages 12/14.

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
