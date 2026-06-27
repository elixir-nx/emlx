# Stage 01 — IR + C++ substrate + one op end-to-end

Status: not started

## Why this stage exists

This stage stands up the whole pipeline end-to-end with a single op (`add`), so
every later stage only adds op-expansion clauses rather than infrastructure. It
also lands the C++ `compile_program`/`eval_program` substrate **early** (per
the resolved decision) so the dispatch-collapse perf thesis is validated from
the first op — the decision gate that justifies the entire effort.

## Procedure

1. **IR struct** `EMLX.NativeExpr` (`emlx/lib/emlx/defn/native_expr.ex`):
   `n_inputs`, `captures`, `consts`, `instrs`, `outputs`. Tagged operand refs
   `{:input | :capture | :const | :instr, index}` packed into an int64
   (`pack_ref/1`/`unpack_ref/1`, kind in high bits). Opcode table (start: just
   `add`) with integer wire values; integer attribute channel (`iattrs`).
2. **Lowerer** `EMLX.NativeExpr.lower/1`: run `EMLX.Defn.Tree.post_order/1`
   (Stage 00), then reduce over nodes with one `expand_node/2` clause per op.
   Implement `parameter`→`{:input,i}`, `constant`→`{:const,i}`,
   `tensor`→`{:capture,i}`, and `add`. Any other op raises
   `ArgumentError "does not yet lower op :foo"`.
3. **Elixir IR interpreter** (`EMLX.NativeExpr.Interpreter` or test support):
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

- `EMLX.NativeExpr` struct + ref packing + opcode table exist and round-trip
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
| IR struct + ref packing | | |
| compile/eval NIFs + parity test | | |
| Compiler seam wired | | |
| `add` end-to-end correct | | |
| Perf vs Evaluator (gate) | | |
