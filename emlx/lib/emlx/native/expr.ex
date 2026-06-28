defmodule EMLX.Native.Expr do
  @moduledoc """
  The `EMLX.Native.Expr` IR for EMLX's single-mode `defn` compiler.

  Each node in the graph is identified by an Erlang ref (`make_ref()`), making
  the program easy to inspect and manipulate without worrying about index arithmetic.
  Opcodes are atoms (`:add`, `:mul`, …) for readability.

  The integer wire format required by the C++ `compile_program` NIF is produced
  by `to_wire/1`, which runs once per cache miss and never on the hot path.

  ## Structure

  - `inputs`   — one ref per `defn` parameter, in position order.
  - `captures` — `[{ref, %Nx.Tensor{}}]` for tensors closed over at compile time.
  - `constants`     — `[{ref, number, Nx.Type.t()}]` for compile-time scalar literals.
  - `instructions`  — `[{result_ref, opcode_atom, [operand_ref]}]` in dependency order.
  - `outputs`  — list of refs identifying the return values.

  ## Opcode table

  `wire_opcodes/0` returns the `[{atom, integer}]` parity table. The integer
  values must stay in sync with the `NativeExprOpcode` C++ enum in
  `emlx_nif.cpp`. The opcode parity test in `EMLX.Native.ExprTest` verifies
  the two tables are identical at runtime.
  """

  import Bitwise

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T

  # Kind tag bits used when packing refs for the NIF wire format.
  # Only referenced in to_wire/1; not part of the public struct.
  @kind_input   0
  @kind_capture 1
  @kind_const   2
  @kind_instr   3
  @kind_shift   60

  @enforce_keys [:inputs, :captures, :constants, :instructions, :outputs]
  defstruct [:inputs, :captures, :constants, :instructions, :outputs]

  @type node_ref :: reference()
  @type t :: %__MODULE__{
          inputs: [node_ref()],
          captures: [{node_ref(), Nx.Tensor.t()}],
          constants: [{node_ref(), number(), Nx.Type.t()}],
          instructions: [{node_ref(), atom(), [node_ref()]}],
          outputs: [node_ref()]
        }

  # ── lowering ──────────────────────────────────────────────────────────────

  @doc """
  Lowers a traced `Nx.Defn.Expr` output to an `EMLX.Native.Expr` program.

  `output` is any `Nx.Container.t()` — the result of `fun.(vars)`.

  Raises `ArgumentError` with message `"does not yet lower op :foo"` for any
  op not yet implemented. The compiler seam in `EMLX.__compile__/4` catches
  this message and falls back to `Nx.Defn.Evaluator`.
  """
  @spec lower(Nx.Container.t()) :: t()
  def lower(output) do
    ordered = EMLX.Defn.Tree.post_order(output)

    # inputs is a map of pos → ref during lowering; sorted to a list at the end.
    state = %{
      inputs: %{},
      captures: [],
      constants: [],
      instructions: [],
      node_to_ref: %{}
    }

    state = Enum.reduce(ordered, state, &expand_node/2)

    inputs_list =
      state.inputs
      |> Enum.sort_by(fn {pos, _ref} -> pos end)
      |> Enum.map(fn {_pos, ref} -> ref end)

    flat_outputs = Composite.flatten_list([output])
    output_refs = Enum.map(flat_outputs, &Map.fetch!(state.node_to_ref, &1.data.id))

    %__MODULE__{
      inputs: inputs_list,
      captures: Enum.reverse(state.captures),
      constants: Enum.reverse(state.constants),
      instructions: Enum.reverse(state.instructions),
      outputs: output_refs
    }
  end

  # ── node expansion ────────────────────────────────────────────────────────

  defp expand_node(%T{data: %Nx.Defn.Expr{id: id, op: :parameter, args: [pos]}}, state) do
    ref = make_ref()

    %{
      state
      | inputs: Map.put(state.inputs, pos, ref),
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{id: id, op: :constant, args: [number]}} = node, state) do
    ref = make_ref()

    %{
      state
      | constants: [{ref, number, node.type} | state.constants],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :tensor, args: [backend_tensor]}},
         state
       ) do
    ref = make_ref()

    %{
      state
      | captures: [{ref, backend_tensor} | state.captures],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :metadata, args: [inner, _meta]}},
         state
       ) do
    inner_ref = Map.fetch!(state.node_to_ref, inner.data.id)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, inner_ref)}
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :add, args: [left, right]}},
         state
       ) do
    ref = make_ref()
    left_ref = Map.fetch!(state.node_to_ref, left.data.id)
    right_ref = Map.fetch!(state.node_to_ref, right.data.id)

    %{
      state
      | instructions: [{ref, :add, [left_ref, right_ref]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: op}}, _state) do
    raise ArgumentError, "does not yet lower op #{inspect(op)}"
  end

  # ── wire serialisation ────────────────────────────────────────────────────

  @doc """
  Translates an `EMLX.Native.Expr` to the wire format expected by
  `EMLX.NIF.compile_program/9`.

  Returns an 8-tuple:
  `{num_inputs, capture_nif_refs, constant_values, constant_types,
    op_names, operands, iattrs, output_packed_refs}`

  `op_names` is a list of strings (e.g. `"add"`) that map directly to entries
  in the C++ `op_registry`; no integer opcode table is required.

  This runs once per compilation cache miss; it has no effect on hot-path
  performance.
  """
  @spec to_wire(t()) ::
          {non_neg_integer(), list(), list(), list(), list(), list(), list(), list()}
  def to_wire(%__MODULE__{} = prog) do
    # Build ref → packed_int map for all non-instruction nodes.
    input_map =
      prog.inputs
      |> Enum.with_index()
      |> Map.new(fn {ref, i} -> {ref, (@kind_input <<< @kind_shift) ||| i} end)

    capture_map =
      prog.captures
      |> Enum.with_index()
      |> Map.new(fn {{ref, _t}, i} -> {ref, (@kind_capture <<< @kind_shift) ||| i} end)

    constant_map =
      prog.constants
      |> Enum.with_index()
      |> Map.new(fn {{ref, _v, _t}, i} -> {ref, (@kind_const <<< @kind_shift) ||| i} end)

    ref_to_packed = Map.merge(input_map, Map.merge(capture_map, constant_map))

    # Walk instructions in order, building the wire arrays and extending the map.
    {op_names, operands, iattrs, ref_to_packed} =
      prog.instructions
      |> Enum.with_index()
      |> Enum.reduce({[], [], [], ref_to_packed}, fn {{id, op, operand_refs}, idx},
                                                     {ops, ors, ias, rmap} ->
        wire_operands = Enum.map(operand_refs, &Map.fetch!(rmap, &1))
        rmap2 = Map.put(rmap, id, (@kind_instr <<< @kind_shift) ||| idx)
        {[op | ops], [wire_operands | ors], [[] | ias], rmap2}
      end)

    wire_outputs = Enum.map(prog.outputs, &Map.fetch!(ref_to_packed, &1))

    capture_nif_refs =
      Enum.map(prog.captures, fn {_ref, %Nx.Tensor{data: %EMLX.Backend{ref: {_, nif_ref}}}} ->
        nif_ref
      end)

    constant_values = Enum.map(prog.constants, fn {_, v, _} -> v * 1.0 end)
    constant_types = Enum.map(prog.constants, fn {_, _, t} -> EMLX.Native.to_mlx_type(t) end)

    {length(prog.inputs), capture_nif_refs, constant_values, constant_types,
     Enum.reverse(op_names), Enum.reverse(operands), Enum.reverse(iattrs), wire_outputs}
  end

end

defmodule EMLX.Native.Expr.Interpreter do
  @moduledoc """
  Pure-Elixir reference implementation of the `EMLX.Native.Expr` evaluator.

  Running the same program through this interpreter and through the C++
  `eval_program` NIF must produce numerically equivalent results. When they
  disagree, the bug is in the C++ path — the interpreter is the ground truth.

  Maintains a `%{node_ref => %Nx.Tensor{}}` environment and dispatches each
  instruction by its atom opcode through the eager `EMLX.Backend` NIFs. Adding
  a new op here before its C++ counterpart lets you verify the lowering logic
  in isolation.
  """

  alias EMLX.Native.Expr

  @doc """
  Evaluates `program` against `inputs`.

  `inputs` is a list of `%Nx.Tensor{}` (with `EMLX.Backend`), one per declared
  input in position order. Returns a list of output tensors matching
  `program.outputs`.
  """
  @spec eval(IR.t(), [Nx.Tensor.t()]) :: [Nx.Tensor.t()]
  def eval(%Expr{} = prog, inputs) when is_list(inputs) do
    env =
      %{}
      |> Map.merge(Map.new(Enum.zip(prog.inputs, inputs)))
      |> Map.merge(Map.new(prog.captures, fn {ref, t} -> {ref, t} end))
      |> Map.merge(
        Map.new(prog.constants, fn {ref, v, type} ->
          {ref, Nx.tensor(v, type: type, backend: EMLX.Backend)}
        end)
      )

    env =
      Enum.reduce(prog.instructions, env, fn {id, op, operand_refs}, env ->
        args = Enum.map(operand_refs, &Map.fetch!(env, &1))
        Map.put(env, id, dispatch(op, args))
      end)

    Enum.map(prog.outputs, &Map.fetch!(env, &1))
  end

  # Dispatch by atom opcode. Use Nx.add/2 so EMLX.Backend handles broadcasting,
  # type promotion, and worker routing automatically.
  defp dispatch(:add, [a, b]), do: Nx.add(a, b)

  defp dispatch(op, _args),
    do: raise(ArgumentError, "Native.Expr.Interpreter: unknown op #{inspect(op)}")
end
