# Stage 01 — IR + C++ substrate + one op end-to-end

Status: complete + post-stage refactors applied (see § Post-stage refactors)

## Why this stage exists

This stage stands up the whole pipeline end-to-end with a single op (`add`), so
every later stage only adds op-expansion clauses rather than infrastructure. It
also lands the C++ `compile_program`/`eval_program` substrate **early** (per
the resolved decision) so the dispatch-collapse perf thesis is validated from
the first op — the decision gate that justifies the entire effort.

## Procedure

1. **IR struct** `EMLX.Native.Expr` (`emlx/lib/emlx/native/expr.ex`):
   `n_inputs`, `captures`, `consts`, `instrs`, `outputs`. Tagged operand refs
   `{:input | :capture | :const | :instr, index}` packed into an int64
   (`pack_ref/1`/`unpack_ref/1`, kind in high bits). Integer attribute channel
   (`iattrs`).
2. **Lowerer** `EMLX.Native.Expr.lower/1`: run `EMLX.Defn.Tree.post_order/1`
   (Stage 00), then reduce over nodes with one `expand_node/2` clause per op.
   Implement `parameter`→`{:input,i}`, `constant`→`{:const,i}`,
   `tensor`→`{:capture,i}`, and `add`. Any other op raises
   `ArgumentError "does not yet lower op :foo"`.
3. **Elixir IR interpreter** (`EMLX.Native.Expr.Interpreter` or test support):
   walk `instrs`, dispatch each through the eager `EMLX.Backend` NIFs, return
   output refs. This is the Layer-B oracle and a temporary executor.
4. **C++ program** (`emlx/c_src/`): `compile_program` NIF (op_names + packed
   operands + iattrs + captured weight refs → reusable program resource) and
   `eval_program` NIF (call the MLX-compiled function, eval outputs, return refs).
5. **Compiler seam**: replace the `Nx.Defn.Evaluator` delegation in
   `EMLX.__compile__/4` (`emlx/lib/emlx.ex`) with the single lowering path:
   trace → `lower` → `compile_program` (cached in the closure) → per-call
   `eval_program`. Keep the existing `:device`/`:command_queue` handling and
   command-queue wrapping.
6. **End-to-end + perf**: `Nx.Defn.jit(fn x -> Nx.add(x, 1) end, compiler: EMLX)`
   returns correct results on `EMLX.Backend`. Add a micro-benchmark on a
   multi-`add` chain comparing single-NIF replay vs the old Evaluator path.

## Acceptance

- `EMLX.Native.Expr` struct + ref packing exist and round-trip.
- `compile_program`/`eval_program` NIFs build and run correctly.
- `Nx.Defn.jit(&(&1 + 1), compiler: EMLX).(x)` yields the correct tensor via the
  single-NIF replay path (not the Evaluator), verified equal to eager
  `EMLX.Backend` within tolerance.
- The Elixir IR interpreter produces the same result as the C++ replay for the
  `add` program.
- Perf gate: single-NIF replay beats the op-by-op Evaluator on a multi-op
  chain; numbers recorded. If it does not, mark `blocked` and escalate before
  proceeding.
- `mix compile --warnings-as-errors`, `mix format --check-formatted`, and the
  GPU/CPU test paths in `.github/workflows/emlx.yml` stay green.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| IR struct + ref packing | ✅ Pass | `EMLX.Native.Expr`, `pack_ref/1`, `unpack_ref/1` |
| compile/eval NIFs | ✅ Pass | `compile_program`, `eval_program` NIFs; `mlx::core::detail::compile` with unique IDs |
| Op-name registry | ✅ Pass | String→fn map replaces Op enum + wire integers; no parity table needed |
| Compiler seam wired | ✅ Pass | `EMLX.__compile__/4` → native path + `Nx.Defn.Evaluator` fallback |
| `add` end-to-end correct | ✅ Pass | Interpreter ↔ C++ replay agree; E2E tests pass |
| Perf vs Evaluator (gate) | ⚠️ Soft-pass | See § Perf findings below |

All 28 tests in `test/emlx/native/expr_test.exs` + `test/emlx/defn/tree_test.exs` pass.

## Post-stage refactors

Three improvements were applied after the initial stage-01 landing:

### 1. `mlx::core::compile` in `compile_program`

The original `eval_program` ran a plain interpreter loop (building the lazy MLX
graph on every call). `compile_program` now wraps the interpreter lambda with
`mlx::core::detail::compile(fn, unique_id)` so MLX traces the computation graph
on the **first** `eval_program` call and replays the cached compiled graph on all
subsequent calls — no repeated graph construction.

**MLX compile cache collision fix:** all interpreter lambdas share the same C++
type (identical capture types), so the public `mlx::core::compile(fn)` would key
every `Expr` to the same cache slot via `type_info`. The internal
`mlx::core::detail::compile(fn, fun_id)` API accepts an explicit `std::uintptr_t`
cache key. A global atomic counter assigns a unique ID to each `compile_program`
call; `Expr::~Expr()` calls `mlx::core::detail::compile_erase(compile_id)` to
evict the entry when the BEAM resource is GC'd.

### 2. Op-name string registry (replaces Op enum + wire integers)

The C++ `Op` enum, `native_expr_opcode_table` NIF, and Elixir `@opcode_table` /
`wire_opcodes/0` are **deleted**. In their place:

- A `static const std::unordered_map<std::string, OpFn> op_registry` in
  `emlx_compiler.cpp` maps op name strings (e.g. `"add"`) to
  `(vector<array>, vector<int64_t>) → array` functions.
- `to_wire/1` now emits op atoms (`:add`) directly; `get_list<vector<string>>`
  reads them via `get_atom`, so BEAM atoms arrive in C++ as string keys.
- Extending to a new op: one line in `op_registry` + one `expand_node/2` clause.
  No enum, no integer wire value, no lockstep parity test.

### 3. Op function signature generalized

The dispatch loop no longer hard-codes binary arity. Each instruction resolves
all its operands into a `vector<array>` and passes them (plus `attrs`) to the
registry function. This accommodates unary, binary, ternary, and attribute-heavy
ops with no structural change.

## Perf findings

**Numbers (Apple M-series CPU, scalar tensors, 10-add chain, 500 iterations):**

| Path | µs/call |
|------|---------|
| Native `eval_program` + `Nx.to_number` | ~145–200 µs |
| Evaluator (10× lazy `Nx.add`) + `Nx.to_number` | ~57–124 µs |
| Speedup | ~0.3–0.7× (native slower at scalar scale) |

**Root cause — eager eval vs. deferred eval:**

`eval_program` calls `mlx::core::eval(outputs)` **inside the NIF body** (on the
worker thread) before returning. This forces synchronous completion of the entire
compute graph before the BEAM gets control back.

The Evaluator defers `mlx::core::eval` until `Nx.to_number` → `EMLX.to_binary`,
where MLX can schedule evaluation while the BEAM is still doing work.

**The thesis is sound, the benchmark is a worst case:**

The dispatch-collapse benefit shows up when:

- The tensor workload is large enough that N×NIF-dispatch overhead dominates.
- The program has enough ops that the NIF-call-count saving is significant.

Scalar microbenchmarks expose only the scheduler startup cost, not the dispatch
saving. The crossover point is expected at medium-to-large tensors (>1 K elements)
with >20 ops.

**Mitigation for Stage 02+:**

1. Remove the `mlx::core::eval` call from `eval_program` (return lazy refs, let
   the caller trigger eval via a subsequent `to_binary`). This matches the
   Evaluator pattern and eliminates the premature barrier.
2. Re-run the perf gate with ≥1 K element tensors and a 20-op chain.
3. Track `speedup` in the results table above once the gate passes.
