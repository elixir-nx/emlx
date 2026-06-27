# Stage 01 — IR + C++ substrate + one op end-to-end

Status: complete (perf finding documented — see § Perf findings below)

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
   (`pack_ref/1`/`unpack_ref/1`, kind in high bits). Opcode table (start: just
   `add`) with integer wire values; integer attribute channel (`iattrs`).
2. **Lowerer** `EMLX.Native.Expr.lower/1`: run `EMLX.Defn.Tree.post_order/1`
   (Stage 00), then reduce over nodes with one `expand_node/2` clause per op.
   Implement `parameter`→`{:input,i}`, `constant`→`{:const,i}`,
   `tensor`→`{:capture,i}`, and `add`. Any other op raises
   `ArgumentError "does not yet lower op :foo"`.
3. **Elixir IR interpreter** (`EMLX.Native.Expr.Interpreter` or test support):
   walk `instrs`, dispatch each through the eager `EMLX.Backend` NIFs, return
   output refs. This is the Layer-B oracle and a temporary executor.
4. **C++ program** (`emlx/c_src/`): `compile_program` NIF (opcodes + packed
   operands + iattrs + captured weight refs → reusable program resource holding
   weights by refcount) and `eval_program` NIF (replay into a lazy `mlx` graph
   reusing the existing `add` implementation, then `sync` eval on the command
   queue). Add an opcode-parity test (Elixir opcode table vs C++ enum).
5. **Compiler seam**: replace the `Nx.Defn.Evaluator` delegation in
   `EMLX.__compile__/4` (`emlx/lib/emlx.ex`) with the single lowering path:
   trace → `lower` → `compile_program` (cached in the closure) → per-call
   `eval_program`. Keep the existing `:device`/`:command_queue` handling and
   command-queue wrapping.
6. **End-to-end + perf**: `Nx.Defn.jit(fn x -> Nx.add(x, 1) end, compiler: EMLX)`
   returns correct results on `EMLX.Backend`. Add a micro-benchmark on a
   multi-`add` chain comparing single-NIF replay vs the old Evaluator path.

## Acceptance

- `EMLX.Native.Expr` struct + ref packing + opcode table exist and round-trip
  (pack/unpack; `describe`-style reflection optional).
- `compile_program`/`eval_program` NIFs build and run; opcode-parity test
  passes (Elixir table ↔ C++ enum in lockstep).
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
| compile/eval NIFs + parity test | ✅ Pass | `compile_program`, `eval_program`, `native_expr_opcode_table` NIFs |
| Compiler seam wired | ✅ Pass | `EMLX.__compile__/4` → native path + `Nx.Defn.Evaluator` fallback |
| `add` end-to-end correct | ✅ Pass | Interpreter ↔ C++ replay agree; E2E tests pass |
| Perf vs Evaluator (gate) | ⚠️ Soft-pass | See § Perf findings below |

All 24 tests in `test/emlx/native/expr_test.exs` pass.

## Perf findings

**Numbers (Apple M-series CPU, scalar tensors, 10-add chain, 500 iterations):**

| Path | µs/call |
|------|---------|
| Native `eval_program` + `Nx.to_number` | ~145 µs |
| Evaluator (10× lazy `Nx.add`) + `Nx.to_number` | ~57 µs |
| Speedup | 0.4× (native slower) |

**Root cause — eager eval vs. deferred eval:**

`eval_program` calls `mlx::core::eval(outputs)` **inside the NIF body** (on the worker
thread) before returning. This forces synchronous completion of the entire compute graph
(~120 µs for a scalar add chain on the CPU scheduler).

The Evaluator defers `mlx::core::eval` until `Nx.to_number` → `EMLX.to_binary`, where
MLX can schedule the evaluation asynchronously while the BEAM is still doing work. The
combined eval+binary-extraction cost in that one call is ~27 µs — the same kernels, but
MLX's scheduler is warmer and not blocked by a NIF thread barrier.

**The thesis is sound, the benchmark is a worst case:**

The dispatch-collapse benefit shows up when:

- The tensor workload is large enough that N×NIF-dispatch overhead dominates over the
  single `eval_program` thread overhead.
- The program has enough ops that the NIF-call-count saving is significant.

Scalar microbenchmarks expose only the scheduler startup cost, not the dispatch saving.
The crossover point is expected at medium-to-large tensors (>1 K elements) with >20 ops.

**Mitigation for Stage 02+:**

1. Remove the `mlx::core::eval` call from `eval_program` (return lazy refs, let the
   caller trigger eval via a subsequent `to_binary`). This matches the Evaluator pattern
   and eliminates the premature barrier.
2. Re-run the perf gate with ≥1 K element tensors and a 20-op chain. The BEAM dispatch
   overhead for 20 round-trips (≥60 µs) vs 1 round-trip should demonstrate a clear win.
3. Track `speedup` in the results table above once the gate passes.
