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
  - `instructions`  — `[{result_ref, opcode_atom, [operand_ref], [integer_attr]}]`
                       in dependency order. The integer-attr list is opcode-specific;
                       for `:astype` it carries a single dtype integer (see `@mlx_type_to_int`).
  - `outputs`  — list of refs identifying the return values.

  ## astype encoding

  `:astype` instructions carry the target MLX dtype as `attrs[0]`. The mapping is
  the `@mlx_type_to_int` module attribute (see below), which must stay in sync with
  the `int_to_dtype` helper in `emlx_compiler.cpp`.
  """

  import Bitwise

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T

  # Kind tag bits used when packing refs for the NIF wire format.
  # Only referenced in to_wire/1; not part of the public struct.
  @kind_input 0
  @kind_capture 1
  @kind_const 2
  @kind_instr 3
  @kind_shift 60

  # Stable integer encoding for MLX dtype atoms, used in :astype iattrs.
  # Must stay in sync with int_to_dtype() in emlx_compiler.cpp.
  @mlx_type_to_int %{
    bool: 0,
    uint8: 1,
    uint16: 2,
    uint32: 3,
    uint64: 4,
    int8: 5,
    int16: 6,
    int32: 7,
    int64: 8,
    float16: 9,
    bfloat16: 10,
    float32: 11,
    complex64: 12
  }

  @enforce_keys [:inputs, :captures, :constants, :instructions, :outputs]
  defstruct [:inputs, :captures, :constants, :instructions, :outputs]

  @type node_ref :: reference()
  @type t :: %__MODULE__{
          inputs: [node_ref()],
          captures: [{node_ref(), Nx.Tensor.t()}],
          constants: [{node_ref(), number(), Nx.Type.t()}],
          instructions: [{node_ref(), atom(), [node_ref()], [integer()]}],
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

  # ── unary elementwise ops ─────────────────────────────────────────────────

  # Direct unary ops — no coercion; MLX infers result dtype from input.
  @unary_direct_ops [
    :abs,
    :ceil,
    :floor,
    :negate,
    :round,
    :sign,
    :real,
    :imag,
    :is_nan,
    :is_infinity,
    :bitwise_not,
    :conjugate,
    :logical_not,
    :sigmoid,
    :asin,
    :asinh,
    :acos,
    :acosh,
    :atan,
    :atanh,
    :cos,
    :cosh,
    :erf,
    :erf_inv,
    :exp,
    :expm1,
    :log,
    :log1p,
    :rsqrt,
    :sin,
    :sinh,
    :sqrt,
    :tan,
    :tanh,
    :cbrt,
    :erfc
  ]

  for op <- @unary_direct_ops do
    defp expand_node(
           %T{data: %Nx.Defn.Expr{id: id, op: unquote(op), args: [operand]}},
           state
         ) do
      ref = make_ref()
      operand_ref = Map.fetch!(state.node_to_ref, operand.data.id)

      %{
        state
        | instructions: [{ref, unquote(op), [operand_ref], []} | state.instructions],
          node_to_ref: Map.put(state.node_to_ref, id, ref)
      }
    end
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: :count_leading_zeros}}, _state) do
    raise ArgumentError, "count_leading_zeros is not supported by EMLX"
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: :population_count}}, _state) do
    raise ArgumentError, "population_count is not supported by EMLX"
  end

  # block: dispatch on struct — Stage 02 handles Nx.Block.LogicalNot.
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.LogicalNot{}, [operand], _default, _fun]
           }
         },
         state
       ) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, operand.data.id)

    %{
      state
      | instructions: [{ref, :logical_not, [operand_ref], []} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # ── binary elementwise ops ────────────────────────────────────────────────

  # Arithmetic group: add, subtract, multiply, pow, left_shift
  # Binary coercion: maybe_upcast(l, r), op, astype(result, out.type)
  @binary_arithmetic_ops [:add, :subtract, :multiply, :pow, :left_shift]

  for op <- @binary_arithmetic_ops do
    defp expand_node(
           %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: unquote(op), args: [left, right]}},
           state
         ) do
      expand_binary_node(id, unquote(op), out_type, left, right, state)
    end
  end

  # Binary group 2: divide, quotient, atan2, right_shift, bitwise_and/or/xor,
  # compare (equal/not_equal/…), logical (and/or/xor)
  @binary_generic_ops [
    :divide,
    :quotient,
    :atan2,
    :right_shift,
    :bitwise_and,
    :bitwise_or,
    :bitwise_xor,
    :equal,
    :not_equal,
    :greater,
    :less,
    :greater_equal,
    :less_equal,
    :logical_and,
    :logical_or,
    :logical_xor
  ]

  for op <- @binary_generic_ops do
    defp expand_node(
           %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: unquote(op), args: [left, right]}},
           state
         ) do
      expand_binary_node(id, unquote(op), out_type, left, right, state)
    end
  end

  # min → minimum, max → maximum (mapped in C++ registry)
  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :min, args: [left, right]}},
         state
       ) do
    expand_binary_node(id, :min, out_type, left, right, state)
  end

  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :max, args: [left, right]}},
         state
       ) do
    expand_binary_node(id, :max, out_type, left, right, state)
  end

  # remainder: composite sign-fix is handled entirely in the C++ registry.
  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :remainder, args: [left, right]}},
         state
       ) do
    expand_binary_node(id, :remainder, out_type, left, right, state)
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: op}}, _state) do
    raise ArgumentError, "does not yet lower op #{inspect(op)}"
  end

  # ── binary lowering helpers ────────────────────────────────────────────────

  # Implements EMLX.Backend's maybe_upcast + op + astype(out.type) pattern:
  #   1. Cast both inputs to merge_type if their types differ.
  #   2. Emit the op instruction.
  #   3. Cast result to out_type (no-op when merge_type == out_type, e.g. arithmetic;
  #      needed for compare ops where result is MLX bool_ but out_type is {:u,8}).
  defp expand_binary_node(id, op, out_type, left, right, state) do
    merge_type = Nx.Type.merge(left.type, right.type)
    left_ref0 = Map.fetch!(state.node_to_ref, left.data.id)
    right_ref0 = Map.fetch!(state.node_to_ref, right.data.id)

    {left_ref, state} = emit_cast_if_needed(left_ref0, left.type, merge_type, state)
    {right_ref, state} = emit_cast_if_needed(right_ref0, right.type, merge_type, state)

    op_ref = make_ref()

    state = %{
      state
      | instructions: [{op_ref, op, [left_ref, right_ref], []} | state.instructions]
    }

    {result_ref, state} = emit_cast_if_needed(op_ref, merge_type, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # Emit an :astype instruction; returns {new_ref, updated_state}.
  defp emit_cast_to(ref, nx_type, state) do
    cast_ref = make_ref()
    mlx_type = EMLX.Native.to_mlx_type(nx_type)
    type_int = Map.fetch!(@mlx_type_to_int, mlx_type)
    instr = {cast_ref, :astype, [ref], [type_int]}
    {cast_ref, %{state | instructions: [instr | state.instructions]}}
  end

  # Emit :astype only when the MLX type representation of from_type differs from to_type.
  defp emit_cast_if_needed(ref, from_type, to_type, state) do
    if EMLX.Native.to_mlx_type(from_type) == EMLX.Native.to_mlx_type(to_type) do
      {ref, state}
    else
      emit_cast_to(ref, to_type, state)
    end
  end

  # ── int ↔ Nx.Type conversion (used by Interpreter) ────────────────────────

  @doc """
  Converts an MLX dtype integer (from `@mlx_type_to_int`) back to an `Nx.Type.t()`.
  Used by `EMLX.Native.Expr.Interpreter` to dispatch `:astype` instructions.
  """
  @spec int_to_nx_type(integer()) :: Nx.Type.t()
  def int_to_nx_type(0), do: {:u, 8}
  def int_to_nx_type(1), do: {:u, 8}
  def int_to_nx_type(2), do: {:u, 16}
  def int_to_nx_type(3), do: {:u, 32}
  def int_to_nx_type(4), do: {:u, 64}
  def int_to_nx_type(5), do: {:s, 8}
  def int_to_nx_type(6), do: {:s, 16}
  def int_to_nx_type(7), do: {:s, 32}
  def int_to_nx_type(8), do: {:s, 64}
  def int_to_nx_type(9), do: {:f, 16}
  def int_to_nx_type(10), do: {:bf, 16}
  def int_to_nx_type(11), do: {:f, 32}
  def int_to_nx_type(12), do: {:c, 64}

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
      |> Map.new(fn {ref, i} -> {ref, @kind_input <<< @kind_shift ||| i} end)

    capture_map =
      prog.captures
      |> Enum.with_index()
      |> Map.new(fn {{ref, _t}, i} -> {ref, @kind_capture <<< @kind_shift ||| i} end)

    constant_map =
      prog.constants
      |> Enum.with_index()
      |> Map.new(fn {{ref, _v, _t}, i} -> {ref, @kind_const <<< @kind_shift ||| i} end)

    ref_to_packed = Map.merge(input_map, Map.merge(capture_map, constant_map))

    # Walk instructions in order, building the wire arrays and extending the map.
    {op_names, operands, iattrs, ref_to_packed} =
      prog.instructions
      |> Enum.with_index()
      |> Enum.reduce({[], [], [], ref_to_packed}, fn {{id, op, operand_refs, attrs}, idx},
                                                     {ops, ors, ias, rmap} ->
        wire_operands = Enum.map(operand_refs, &Map.fetch!(rmap, &1))
        rmap2 = Map.put(rmap, id, @kind_instr <<< @kind_shift ||| idx)
        {[op | ops], [wire_operands | ors], [attrs | ias], rmap2}
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
  @spec eval(Expr.t(), [Nx.Tensor.t()]) :: [Nx.Tensor.t()]
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
      Enum.reduce(prog.instructions, env, fn {id, op, operand_refs, attrs}, env ->
        args = Enum.map(operand_refs, &Map.fetch!(env, &1))
        Map.put(env, id, dispatch(op, args, attrs))
      end)

    Enum.map(prog.outputs, &Map.fetch!(env, &1))
  end

  # ── dispatch ──────────────────────────────────────────────────────────────
  # Each clause mirrors the C++ op_registry entry in emlx_compiler.cpp.
  # Uses Nx public API so EMLX.Backend handles broadcasting, type promotion,
  # and worker routing automatically.

  # cast
  defp dispatch(:astype, [tensor], [type_int]) do
    Nx.as_type(tensor, Expr.int_to_nx_type(type_int))
  end

  # unary math
  defp dispatch(:abs, [x], []), do: Nx.abs(x)
  defp dispatch(:ceil, [x], []), do: Nx.ceil(x)
  defp dispatch(:floor, [x], []), do: Nx.floor(x)
  defp dispatch(:negate, [x], []), do: Nx.negate(x)
  defp dispatch(:round, [x], []), do: Nx.round(x)
  defp dispatch(:sign, [x], []), do: Nx.sign(x)
  defp dispatch(:real, [x], []), do: Nx.real(x)
  defp dispatch(:imag, [x], []), do: Nx.imag(x)
  defp dispatch(:is_nan, [x], []), do: Nx.is_nan(x)
  defp dispatch(:is_infinity, [x], []), do: Nx.is_infinity(x)
  defp dispatch(:bitwise_not, [x], []), do: Nx.bitwise_not(x)
  defp dispatch(:conjugate, [x], []), do: Nx.conjugate(x)
  defp dispatch(:logical_not, [x], []), do: Nx.logical_not(x)
  defp dispatch(:cbrt, [x], []), do: Nx.cbrt(x)
  defp dispatch(:erfc, [x], []), do: Nx.erfc(x)

  # unary trig / math funs
  defp dispatch(:sigmoid, [x], []), do: Nx.sigmoid(x)
  defp dispatch(:asin, [x], []), do: Nx.asin(x)
  defp dispatch(:asinh, [x], []), do: Nx.asinh(x)
  defp dispatch(:acos, [x], []), do: Nx.acos(x)
  defp dispatch(:acosh, [x], []), do: Nx.acosh(x)
  defp dispatch(:atan, [x], []), do: Nx.atan(x)
  defp dispatch(:atanh, [x], []), do: Nx.atanh(x)
  defp dispatch(:cos, [x], []), do: Nx.cos(x)
  defp dispatch(:cosh, [x], []), do: Nx.cosh(x)
  defp dispatch(:erf, [x], []), do: Nx.erf(x)
  defp dispatch(:erf_inv, [x], []), do: Nx.erf_inv(x)
  defp dispatch(:exp, [x], []), do: Nx.exp(x)
  defp dispatch(:expm1, [x], []), do: Nx.expm1(x)
  defp dispatch(:log, [x], []), do: Nx.log(x)
  defp dispatch(:log1p, [x], []), do: Nx.log1p(x)
  defp dispatch(:rsqrt, [x], []), do: Nx.rsqrt(x)
  defp dispatch(:sin, [x], []), do: Nx.sin(x)
  defp dispatch(:sinh, [x], []), do: Nx.sinh(x)
  defp dispatch(:sqrt, [x], []), do: Nx.sqrt(x)
  defp dispatch(:tan, [x], []), do: Nx.tan(x)
  defp dispatch(:tanh, [x], []), do: Nx.tanh(x)

  # binary arithmetic
  defp dispatch(:add, [a, b], []), do: Nx.add(a, b)
  defp dispatch(:subtract, [a, b], []), do: Nx.subtract(a, b)
  defp dispatch(:multiply, [a, b], []), do: Nx.multiply(a, b)
  defp dispatch(:divide, [a, b], []), do: Nx.divide(a, b)
  defp dispatch(:pow, [a, b], []), do: Nx.pow(a, b)
  defp dispatch(:remainder, [a, b], []), do: Nx.remainder(a, b)
  defp dispatch(:atan2, [a, b], []), do: Nx.atan2(a, b)
  defp dispatch(:min, [a, b], []), do: Nx.min(a, b)
  defp dispatch(:max, [a, b], []), do: Nx.max(a, b)
  defp dispatch(:quotient, [a, b], []), do: Nx.quotient(a, b)

  # binary bitwise + shifts
  defp dispatch(:bitwise_and, [a, b], []), do: Nx.bitwise_and(a, b)
  defp dispatch(:bitwise_or, [a, b], []), do: Nx.bitwise_or(a, b)
  defp dispatch(:bitwise_xor, [a, b], []), do: Nx.bitwise_xor(a, b)
  defp dispatch(:left_shift, [a, b], []), do: Nx.left_shift(a, b)
  defp dispatch(:right_shift, [a, b], []), do: Nx.right_shift(a, b)

  # binary compare
  defp dispatch(:equal, [a, b], []), do: Nx.equal(a, b)
  defp dispatch(:not_equal, [a, b], []), do: Nx.not_equal(a, b)
  defp dispatch(:greater, [a, b], []), do: Nx.greater(a, b)
  defp dispatch(:less, [a, b], []), do: Nx.less(a, b)
  defp dispatch(:greater_equal, [a, b], []), do: Nx.greater_equal(a, b)
  defp dispatch(:less_equal, [a, b], []), do: Nx.less_equal(a, b)

  # binary logical
  defp dispatch(:logical_and, [a, b], []), do: Nx.logical_and(a, b)
  defp dispatch(:logical_or, [a, b], []), do: Nx.logical_or(a, b)
  defp dispatch(:logical_xor, [a, b], []), do: Nx.logical_xor(a, b)

  defp dispatch(op, _args, _attrs),
    do: raise(ArgumentError, "Native.Expr.Interpreter: unknown op #{inspect(op)}")
end
