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

  ## iattrs encoding per opcode

  The integer attribute list (`attrs`) is opcode-specific. The encoding is the
  source of truth shared with `emlx_compiler.cpp`; keep them in sync.

  | Opcode       | `attrs` layout                                          |
  |-------------|----------------------------------------------------------|
  | `:astype`   | `[dtype_int]` — target MLX dtype (see `@mlx_type_to_int`) |
  | `:bitcast`  | `[dtype_int]` — target MLX dtype                        |
  | `:reshape`  | `[d0, d1, …]` — new shape dims (flat)                   |
  | `:squeeze`  | `[a0, a1, …]` — axes to remove (non-negative)           |
  | `:transpose`| `[p0, p1, …]` — axis permutation (non-negative)         |
  | `:broadcast`| `[n, d0..dn-1, m, a0..am-1]` — `n` shape dims then `m` axes (length-delimited) |
  | `:pad`      | `[n_dims, lo0, hi0, int0, lo1, hi1, int1, …]` — n_dims triples per dim |
  | `:reverse`  | `[a0, a1, …]` — axes to flip (non-negative)             |
  | `:concatenate` | `[axis]` — concat axis; all input tensors in `operands` |
  | `:stack`    | `[axis]` — stack axis; all input tensors in `operands`  |
  | `:sum`, `:product`, `:all`, `:any`, `:reduce_max`, `:reduce_min` | `[keep_axes_int, a0, a1, …]` — 0/1 keep-axes flag then explicit axis list |
  | `:argmax`, `:argmin` | `[axis, keep_axis_int]` — axis index (−1 = global/no-axis) then 0/1 keep flag |
  | `:dot`      | `[n_ca, ca…, n_cb, cb…, n_ba, ba…, n_bb, bb…]` — four length-delimited axis lists: contract-left, contract-right, batch-left, batch-right |
  | `:conv_general` | `[n_dims, s0..sn-1, pl0, ph0, pl1, ph1, …, kd0..kdn-1, id0..idn-1, fgs]` — spatial dims count, strides, padding lo/hi pairs, kernel dilation, input dilation, feature group count |
  | `:select`       | (no attrs) — operands are `[pred, on_true, on_false]`               |
  | `:clip`         | (no attrs) — operands are `[tensor, min, max]`                       |
  | `:slice`        | `[n_dims, dyn_mask, d0..dn-1, l0..ln-1, str0..strn-1, sv0..svn-1]` — rank, dynamic bitmask, input shape, lengths, strides, static starts (0 for dynamic). Dynamic tensor starts are operands after the tensor. |
  | `:put_slice`    | `[n_dims, dyn_mask, d0..dn-1, l0..ln-1, sv0..svn-1]` — rank, dynamic bitmask, input shape, slice shape, static starts (0 for dynamic). Operands are `[input, slice, dyn_starts…]`. |
  | `:gather`       | `[n_gather_axes, a0…, n_tensor_dims, ss0…, n_out_dims, od0…]` — axes, slice_sizes, output shape. Operands: `[tensor, indices]`. |
  | `:take`         | `[axis]` — operands are `[tensor, indices]`                           |
  | `:take_along_axis` | `[axis]` — operands are `[tensor, indices]`                      |
  | `:indexed_add`  | `[n_axes, a0…, n_updates_shape, us0…]` — axes and reshaped-updates dims. Operands: `[target, indices, updates]`. |
  | `:indexed_put`  | same as `:indexed_add`                                               |

  Non-negative axes: the lowerer normalises negative axis values before encoding
  so C++ handlers can use them directly as 0-based indices.

  `:pad` raises for `interior > 0` or negative `lo`/`hi` (not yet lowered).
  `:reduce` (custom-fun reduce) raises — deferred to Stage 08 (requires child programs).
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

  # ── shape / movement ops ──────────────────────────────────────────────────────

  # reshape: iattrs = new shape dims (flat list); shape from the output tensor.
  defp expand_node(
         %T{shape: out_shape, data: %Nx.Defn.Expr{id: id, op: :reshape, args: [tensor]}},
         state
       ) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    shape_attrs = Tuple.to_list(out_shape)

    %{
      state
      | instructions: [{ref, :reshape, [operand_ref], shape_attrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # squeeze: iattrs = axes to remove (non-negative).
  defp expand_node(%T{data: %Nx.Defn.Expr{id: id, op: :squeeze, args: [tensor, axes]}}, state) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    norm_axes = normalize_axes(axes, tuple_size(tensor.shape))

    %{
      state
      | instructions: [{ref, :squeeze, [operand_ref], norm_axes} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # transpose: iattrs = axis permutation (non-negative).
  defp expand_node(%T{data: %Nx.Defn.Expr{id: id, op: :transpose, args: [tensor, axes]}}, state) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    norm_axes = normalize_axes(axes, tuple_size(tensor.shape))

    %{
      state
      | instructions: [{ref, :transpose, [operand_ref], norm_axes} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # as_type: reuse existing :astype opcode; always emit the cast.
  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :as_type, args: [tensor]}},
         state
       ) do
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    {result_ref, state} = emit_cast_to(operand_ref, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # bitcast: iattrs = [target_dtype_int]. Target type from the output tensor.
  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :bitcast, args: [tensor]}},
         state
       ) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    mlx_type = EMLX.Native.to_mlx_type(out_type)
    type_int = Map.fetch!(@mlx_type_to_int, mlx_type)

    %{
      state
      | instructions: [{ref, :bitcast, [operand_ref], [type_int]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # broadcast: iattrs = [n_shape, d0…, n_axes, a0…] (both shape and axes, length-delimited).
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :broadcast, args: [tensor, shape, axes]}},
         state
       ) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    shape_list = Tuple.to_list(shape)
    n_shape = length(shape_list)
    n_axes = length(axes)
    iattrs = [n_shape | shape_list] ++ [n_axes | axes]

    %{
      state
      | instructions: [{ref, :broadcast, [operand_ref], iattrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # pad: raises for interior > 0 or negative lo/hi (not yet lowered).
  # iattrs = [n_dims, lo0, hi0, int0, lo1, hi1, int1, …].
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :pad, args: [tensor, pad_value, config]}},
         state
       ) do
    if Enum.any?(config, fn {lo, hi, interior} -> lo < 0 or hi < 0 or interior > 0 end) do
      raise ArgumentError,
            "does not yet lower op :pad with interior padding or negative lo/hi values"
    end

    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    pad_value_ref = Map.fetch!(state.node_to_ref, pad_value.data.id)
    n_dims = length(config)
    iattrs = [n_dims | Enum.flat_map(config, fn {lo, hi, interior} -> [lo, hi, interior] end)]

    %{
      state
      | instructions: [{ref, :pad, [operand_ref, pad_value_ref], iattrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # reverse: iattrs = axes to flip (non-negative).
  defp expand_node(%T{data: %Nx.Defn.Expr{id: id, op: :reverse, args: [tensor, axes]}}, state) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    norm_axes = normalize_axes(axes, tuple_size(tensor.shape))

    %{
      state
      | instructions: [{ref, :reverse, [operand_ref], norm_axes} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # concatenate / stack: variadic — args is [list_of_tensors, axis].
  # iattrs = [axis], all tensor refs go into operands.
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :concatenate, args: [tensors, axis]}},
         state
       ) do
    ref = make_ref()
    operand_refs = Enum.map(tensors, &Map.fetch!(state.node_to_ref, &1.data.id))
    norm_axis = if axis < 0, do: tuple_size(hd(tensors).shape) + axis, else: axis

    %{
      state
      | instructions: [{ref, :concatenate, operand_refs, [norm_axis]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :stack, args: [tensors, axis]}},
         state
       ) do
    ref = make_ref()
    operand_refs = Enum.map(tensors, &Map.fetch!(state.node_to_ref, &1.data.id))
    # stack output rank = input rank + 1; normalise axis against output rank
    out_rank = tuple_size(hd(tensors).shape) + 1
    norm_axis = if axis < 0, do: out_rank + axis, else: axis

    %{
      state
      | instructions: [{ref, :stack, operand_refs, [norm_axis]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # ── reductions ──────────────────────────────────────────────────────────────

  # sum, product: emit reduction then cast to out_type.
  # all, any: MLX returns bool_; always cast to out_type (u8).
  @reduction_cast_ops [:sum, :product, :all, :any]

  for op <- @reduction_cast_ops do
    defp expand_node(
           %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: unquote(op), args: [tensor, opts]}},
           state
         ) do
      ref = make_ref()
      operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
      axes = opts[:axes] || Nx.axes(tensor)
      keep_axes = if opts[:keep_axes], do: 1, else: 0
      iattrs = [keep_axes | normalize_axes(axes, tuple_size(tensor.shape))]

      state = %{
        state
        | instructions: [{ref, unquote(op), [operand_ref], iattrs} | state.instructions]
      }

      {result_ref, state} = emit_cast_to(ref, out_type, state)
      %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
    end
  end

  # reduce_max, reduce_min: MLX preserves input dtype, no cast needed.
  @reduction_nocast_ops [:reduce_max, :reduce_min]

  for op <- @reduction_nocast_ops do
    defp expand_node(
           %T{data: %Nx.Defn.Expr{id: id, op: unquote(op), args: [tensor, opts]}},
           state
         ) do
      ref = make_ref()
      operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
      axes = opts[:axes] || Nx.axes(tensor)
      keep_axes = if opts[:keep_axes], do: 1, else: 0
      iattrs = [keep_axes | normalize_axes(axes, tuple_size(tensor.shape))]

      %{
        state
        | instructions: [{ref, unquote(op), [operand_ref], iattrs} | state.instructions],
          node_to_ref: Map.put(state.node_to_ref, id, ref)
      }
    end
  end

  # argmax / argmin: axis = nil → -1 (global), otherwise normalised non-negative.
  # MLX returns uint32; always cast to out_type.
  @argreduce_ops [:argmax, :argmin]

  for op <- @argreduce_ops do
    defp expand_node(
           %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: unquote(op), args: [tensor, opts]}},
           state
         ) do
      ref = make_ref()
      operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
      axis = opts[:axis]
      keep_axis = if opts[:keep_axis], do: 1, else: 0

      norm_axis =
        cond do
          is_nil(axis) -> -1
          axis < 0 -> tuple_size(tensor.shape) + axis
          true -> axis
        end

      state = %{
        state
        | instructions: [
            {ref, unquote(op), [operand_ref], [norm_axis, keep_axis]} | state.instructions
          ]
      }

      {result_ref, state} = emit_cast_to(ref, out_type, state)
      %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
    end
  end

  # custom-fun reduce: deferred — requires child programs (Stage 08).
  defp expand_node(%T{data: %Nx.Defn.Expr{op: :reduce}}, _state) do
    raise ArgumentError, "does not yet lower op :reduce"
  end

  # ── dot ─────────────────────────────────────────────────────────────────────

  # dot: args = [left, c_left, b_left, right, c_right, b_right]
  # Cast both operands to computation_type in Elixir; emit :dot with 4-axis-list
  # iattrs; cast result to out_type.
  defp expand_node(
         %T{
           type: out_type,
           data: %Nx.Defn.Expr{
             id: id,
             op: :dot,
             args: [left, c_left, b_left, right, c_right, b_right]
           }
         },
         state
       ) do
    left_ref0 = Map.fetch!(state.node_to_ref, left.data.id)
    right_ref0 = Map.fetch!(state.node_to_ref, right.data.id)

    computation_type =
      if Nx.Type.integer?(out_type), do: Nx.Type.to_floating(out_type), else: out_type

    {left_ref, state} = emit_cast_if_needed(left_ref0, left.type, computation_type, state)
    {right_ref, state} = emit_cast_if_needed(right_ref0, right.type, computation_type, state)

    iattrs =
      [length(c_left) | c_left] ++
        [length(c_right) | c_right] ++
        [length(b_left) | b_left] ++
        [length(b_right) | b_right]

    dot_ref = make_ref()

    state = %{
      state
      | instructions: [{dot_ref, :dot, [left_ref, right_ref], iattrs} | state.instructions]
    }

    {result_ref, state} = emit_cast_if_needed(dot_ref, computation_type, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # ── conv ─────────────────────────────────────────────────────────────────────

  # conv: expanded into existing astype/transpose instructions + a single
  # :conv_general op.  This mirrors EMLX.Backend.conv exactly:
  #   1. Cast input + kernel to out_type.
  #   2. Apply input_permutation then channels-last transpose to input.
  #   3. Apply kernel_permutation then channels-last transpose to kernel.
  #   4. Emit :conv_general with strides/padding/dilations/fgs.
  #   5. Apply channels-first then inverse-output-permutation transpose.
  defp expand_node(
         %T{
           type: out_type,
           shape: out_shape,
           data: %Nx.Defn.Expr{id: id, op: :conv, args: [input, kernel, opts]}
         },
         state
       ) do
    batch_group_size = opts[:batch_group_size]

    if batch_group_size != 1 do
      raise ArgumentError, "does not yet lower op :conv with batch_group_size != 1"
    end

    input_permutation = opts[:input_permutation]
    kernel_permutation = opts[:kernel_permutation]
    output_permutation = opts[:output_permutation]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    feature_group_count = opts[:feature_group_size]

    input_ref = Map.fetch!(state.node_to_ref, input.data.id)
    kernel_ref = Map.fetch!(state.node_to_ref, kernel.data.id)

    # 1. Cast to out_type.
    {input_casted, state} = emit_cast_if_needed(input_ref, input.type, out_type, state)
    {kernel_casted, state} = emit_cast_if_needed(kernel_ref, kernel.type, out_type, state)

    # 2. Transpose input: user permutation then channels-last.
    input_rank = tuple_size(input.shape)
    {input_perm1, state} = emit_transpose_instr(input_casted, input_permutation, state)

    {input_processed, state} =
      emit_transpose_instr(
        input_perm1,
        move_channels_last(Enum.to_list(0..(input_rank - 1))),
        state
      )

    # 3. Transpose kernel: user permutation then channels-last.
    kernel_rank = tuple_size(kernel.shape)
    {kernel_perm1, state} = emit_transpose_instr(kernel_casted, kernel_permutation, state)

    {kernel_processed, state} =
      emit_transpose_instr(
        kernel_perm1,
        move_channels_last(Enum.to_list(0..(kernel_rank - 1))),
        state
      )

    # 4. :conv_general — attrs = [n_dims, s…, pl0,ph0,…, kd…, id…, fgs]
    n_dims = input_rank - 2
    {padding_low, padding_high} = Enum.unzip(padding)

    conv_attrs =
      [n_dims | strides] ++
        Enum.flat_map(Enum.zip(padding_low, padding_high), fn {lo, hi} -> [lo, hi] end) ++
        kernel_dilation ++
        input_dilation ++
        [feature_group_count]

    conv_ref = make_ref()

    state = %{
      state
      | instructions: [
          {conv_ref, :conv_general, [input_processed, kernel_processed], conv_attrs}
          | state.instructions
        ]
    }

    # 5. Transpose output: channels-first then inverse of output_permutation.
    out_rank = tuple_size(out_shape)
    [batch | spatial_and_channels] = Enum.to_list(0..(out_rank - 1))
    {channels, spatial} = List.pop_at(spatial_and_channels, -1)
    permute_channels_first = [batch, channels | spatial]

    output_perm_inverse =
      output_permutation
      |> Enum.with_index()
      |> Enum.sort()
      |> Enum.map(&elem(&1, 1))

    {conv_perm1, state} = emit_transpose_instr(conv_ref, permute_channels_first, state)
    {result_ref, state} = emit_transpose_instr(conv_perm1, output_perm_inverse, state)

    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # ── indexing / selection ops ──────────────────────────────────────────────

  # select: cast on_true and on_false to out_type, then emit :select.
  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :select, args: [pred, on_true, on_false]}},
         state
       ) do
    pred_ref = Map.fetch!(state.node_to_ref, pred.data.id)
    true_ref0 = Map.fetch!(state.node_to_ref, on_true.data.id)
    false_ref0 = Map.fetch!(state.node_to_ref, on_false.data.id)
    {true_ref, state} = emit_cast_if_needed(true_ref0, on_true.type, out_type, state)
    {false_ref, state} = emit_cast_if_needed(false_ref0, on_false.type, out_type, state)
    ref = make_ref()

    %{
      state
      | instructions: [{ref, :select, [pred_ref, true_ref, false_ref], []} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # clip: operands = [tensor, min, max].
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :clip, args: [tensor, min_t, max_t]}},
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    min_ref = Map.fetch!(state.node_to_ref, min_t.data.id)
    max_ref = Map.fetch!(state.node_to_ref, max_t.data.id)

    %{
      state
      | instructions: [{ref, :clip, [tensor_ref, min_ref, max_ref], []} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # slice: start_indices can be integers (static) or tensors (dynamic).
  #
  # iattrs = [n_dims, dynamic_mask, d0..dn-1, l0..ln-1, str0..strn-1, sv0..svn-1]
  #   n_dims       = rank of the input tensor
  #   dynamic_mask = n-bit integer, bit i = 1 if start index i is a tensor
  #   d0..dn-1     = input shape dims (for clamping)
  #   l0..ln-1     = slice lengths (always static)
  #   str0..strn-1 = strides (always static)
  #   sv0..svn-1   = static start values (0 for dynamic dims)
  # Operands = [tensor_ref, dyn_ref_0, dyn_ref_1, ...] — dynamic starts in axis order.
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :slice, args: [tensor, start_indices, lengths, strides]}},
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    n_dims = tuple_size(tensor.shape)
    input_shape = Tuple.to_list(tensor.shape)

    # Partition start_indices into dynamic (tensor refs) and static (integers).
    {dynamic_mask, static_vals, dyn_operand_refs, state} =
      Enum.reduce(Enum.with_index(start_indices), {0, [], [], state}, fn
        {idx, _i}, {mask, statics, dyn_refs, st} when is_integer(idx) ->
          {mask, statics ++ [idx], dyn_refs, st}

        {%T{} = idx_tensor, i}, {mask, statics, dyn_refs, st} ->
          dyn_ref = Map.fetch!(st.node_to_ref, idx_tensor.data.id)
          {mask ||| 1 <<< i, statics ++ [0], dyn_refs ++ [dyn_ref], st}
      end)

    iattrs =
      [n_dims, dynamic_mask] ++
        input_shape ++
        lengths ++
        strides ++
        static_vals

    operands = [tensor_ref | dyn_operand_refs]

    %{
      state
      | instructions: [{ref, :slice, operands, iattrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # put_slice: start_indices can be integers (static) or tensors (dynamic).
  #
  # iattrs = [n_dims, dynamic_mask, d0..dn-1, l0..ln-1, sv0..svn-1]
  # Operands = [input_ref, slice_ref, dyn_ref_0, ...] — dynamic starts in axis order.
  defp expand_node(
         %T{
           type: out_type,
           data: %Nx.Defn.Expr{id: id, op: :put_slice, args: [input, start_indices, slice]}
         },
         state
       ) do
    ref = make_ref()
    input_ref0 = Map.fetch!(state.node_to_ref, input.data.id)
    slice_ref0 = Map.fetch!(state.node_to_ref, slice.data.id)

    # Cast both to out_type.
    {input_ref, state} = emit_cast_if_needed(input_ref0, input.type, out_type, state)
    {slice_ref, state} = emit_cast_if_needed(slice_ref0, slice.type, out_type, state)

    n_dims = tuple_size(input.shape)
    input_shape = Tuple.to_list(input.shape)
    lengths = Tuple.to_list(slice.shape)

    {dynamic_mask, static_vals, dyn_operand_refs, state} =
      Enum.reduce(Enum.with_index(start_indices), {0, [], [], state}, fn
        {idx, _i}, {mask, statics, dyn_refs, st} when is_integer(idx) ->
          {mask, statics ++ [idx], dyn_refs, st}

        {%T{} = idx_tensor, i}, {mask, statics, dyn_refs, st} ->
          dyn_ref = Map.fetch!(st.node_to_ref, idx_tensor.data.id)
          {mask ||| 1 <<< i, statics ++ [0], dyn_refs ++ [dyn_ref], st}
      end)

    iattrs = [n_dims, dynamic_mask] ++ input_shape ++ lengths ++ static_vals
    operands = [input_ref, slice_ref | dyn_operand_refs]

    %{
      state
      | instructions: [{ref, :put_slice, operands, iattrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # gather: args = [tensor, indices, opts], opts has axes: [...].
  #
  # Mirrors EMLX.Backend.gather: decomposes the indices tensor along its last axis
  # into per-axis index arrays; calls mlx::core::gather + reshape.
  #
  # iattrs = [n_gather_axes, a0, a1, ..., n_tensor_dims, ss0, ss1, ..., n_out_dims, od0, od1, ...]
  #   n_gather_axes = number of indexed axes
  #   a0..          = axis indices
  #   n_tensor_dims = rank of tensor
  #   ss0..         = slice_sizes (1 for gathered axes, full dim size for others)
  #   n_out_dims    = rank of out tensor
  #   od0..         = output shape dims
  # Operands = [tensor_ref, indices_ref]
  defp expand_node(
         %T{shape: out_shape, data: %Nx.Defn.Expr{id: id, op: :gather, args: [tensor, indices, opts]}},
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    indices_ref = Map.fetch!(state.node_to_ref, indices.data.id)
    axes = opts[:axes]
    n_gather_axes = length(axes)
    n_tensor_dims = tuple_size(tensor.shape)

    slice_sizes =
      Enum.map(Nx.axes(tensor), fn axis ->
        if axis in axes, do: 1, else: elem(tensor.shape, axis)
      end)

    out_shape_list = Tuple.to_list(out_shape)

    iattrs =
      [n_gather_axes | axes] ++
        [n_tensor_dims | slice_sizes] ++
        [length(out_shape_list) | out_shape_list]

    %{
      state
      | instructions: [{ref, :gather, [tensor_ref, indices_ref], iattrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # block: Nx.Block.Take — take(tensor, indices, axis).
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.Take{axis: axis}, [tensor, indices], _default, _fun]
           }
         },
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    indices_ref = Map.fetch!(state.node_to_ref, indices.data.id)
    norm_axis = if axis < 0, do: tuple_size(tensor.shape) + axis, else: axis

    %{
      state
      | instructions: [{ref, :take, [tensor_ref, indices_ref], [norm_axis]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # block: Nx.Block.TakeAlongAxis — take_along_axis(tensor, indices, axis).
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.TakeAlongAxis{axis: axis}, [tensor, indices], _default, _fun]
           }
         },
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    indices_ref = Map.fetch!(state.node_to_ref, indices.data.id)
    norm_axis = if axis < 0, do: tuple_size(tensor.shape) + axis, else: axis

    %{
      state
      | instructions: [
          {ref, :take_along_axis, [tensor_ref, indices_ref], [norm_axis]} | state.instructions
        ],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # indexed_add / indexed_put: scatter_add / scatter.
  # Mirrors EMLX.Backend.indexed_op: decomposes indices along last axis,
  # reshapes updates, then emits :indexed_add/:indexed_put.
  #
  # iattrs = [n_axes, a0, a1, ..., n_updates_shape_dims, us0, us1, ...]
  # Operands = [target_ref, indices_ref, updates_ref]
  defp expand_node(
         %T{
           type: out_type,
           data: %Nx.Defn.Expr{id: id, op: :indexed_add, args: [target, indices, updates, opts]}
         },
         state
       ) do
    expand_indexed_node(id, :indexed_add, out_type, target, indices, updates, opts, state)
  end

  defp expand_node(
         %T{
           type: out_type,
           data: %Nx.Defn.Expr{id: id, op: :indexed_put, args: [target, indices, updates, opts]}
         },
         state
       ) do
    expand_indexed_node(id, :indexed_put, out_type, target, indices, updates, opts, state)
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: op}}, _state) do
    raise ArgumentError, "does not yet lower op #{inspect(op)}"
  end

  # ── indexing helpers ──────────────────────────────────────────────────────

  # Shared lowering for indexed_add / indexed_put.
  # Computes the reshaped updates shape (same as EMLX.Backend.indexed_op), emits
  # astype casts for target and updates, then emits the opcode.
  defp expand_indexed_node(id, op, out_type, target, indices, updates, opts, state) do
    ref = make_ref()
    axes = opts[:axes] || Nx.axes(target)
    num_axes = elem(indices.shape, tuple_size(indices.shape) - 1)

    # Mirror EMLX.Backend.indexed_op: compute reshape of updates.
    insert_index =
      axes
      |> Enum.scan(&(&1 - &2))
      |> Enum.find_index(&(&1 > 1))
      |> then(&(&1 || num_axes))

    [num_updates | updates_inner_shape] = Tuple.to_list(updates.shape)

    updates_shape =
      [num_updates | List.duplicate(1, num_axes)]
      |> List.insert_at(insert_index + 1, updates_inner_shape)
      |> List.flatten()

    target_ref0 = Map.fetch!(state.node_to_ref, target.data.id)
    indices_ref = Map.fetch!(state.node_to_ref, indices.data.id)
    updates_ref0 = Map.fetch!(state.node_to_ref, updates.data.id)

    {target_ref, state} = emit_cast_if_needed(target_ref0, target.type, out_type, state)
    {updates_ref, state} = emit_cast_if_needed(updates_ref0, updates.type, out_type, state)

    iattrs = [length(axes) | axes] ++ [length(updates_shape) | updates_shape]

    %{
      state
      | instructions: [{ref, op, [target_ref, indices_ref, updates_ref], iattrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
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

  # Normalise negative axis values to non-negative, given the input tensor rank.
  defp normalize_axes(axes, rank) do
    Enum.map(axes, fn ax -> if ax < 0, do: rank + ax, else: ax end)
  end

  # Emit a :transpose instruction; returns {result_ref, updated_state}.
  defp emit_transpose_instr(operand_ref, perm, state) do
    ref = make_ref()
    {ref, %{state | instructions: [{ref, :transpose, [operand_ref], perm} | state.instructions]}}
  end

  # Move the second element (channels) to the last position.
  # [0, 1, 2, 3] → [0, 2, 3, 1]  (NCHW → NHWC permutation)
  # Mirrors EMLX.Backend.move_channels_last/1.
  defp move_channels_last([head | [second | rest]]) do
    [head | rest] ++ [second]
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

  import Bitwise

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

  # shape / movement
  defp dispatch(:reshape, [tensor], attrs),
    do: Nx.reshape(tensor, List.to_tuple(attrs))

  defp dispatch(:squeeze, [tensor], attrs),
    do: Nx.squeeze(tensor, axes: attrs)

  defp dispatch(:transpose, [tensor], attrs),
    do: Nx.transpose(tensor, axes: attrs)

  defp dispatch(:bitcast, [tensor], [type_int]),
    do: Nx.bitcast(tensor, Expr.int_to_nx_type(type_int))

  defp dispatch(:broadcast, [tensor], [n_shape | rest]) do
    shape_dims = Enum.take(rest, n_shape)
    [n_axes | axes] = Enum.drop(rest, n_shape)
    axes = Enum.take(axes, n_axes)
    Nx.broadcast(tensor, List.to_tuple(shape_dims), axes: axes)
  end

  defp dispatch(:pad, [tensor, pad_value], [n_dims | rest]) do
    config =
      Enum.map(0..(n_dims - 1), fn i ->
        {Enum.at(rest, i * 3), Enum.at(rest, i * 3 + 1), Enum.at(rest, i * 3 + 2)}
      end)

    Nx.pad(tensor, pad_value, config)
  end

  defp dispatch(:reverse, [tensor], attrs),
    do: Nx.reverse(tensor, axes: attrs)

  defp dispatch(:concatenate, tensors, [axis]),
    do: Nx.concatenate(tensors, axis: axis)

  defp dispatch(:stack, tensors, [axis]),
    do: Nx.stack(tensors, axis: axis)

  # reductions — iattrs = [keep_axes_int, a0, a1, …]
  for op <- [:sum, :product, :all, :any, :reduce_max, :reduce_min] do
    defp dispatch(unquote(op), [tensor], [keep_axes | axes]) do
      apply(Nx, unquote(op), [tensor, [axes: axes, keep_axes: keep_axes == 1]])
    end
  end

  # argmax / argmin — iattrs = [axis, keep_axis_int]; axis = -1 means global
  for op <- [:argmax, :argmin] do
    defp dispatch(unquote(op), [tensor], [axis, keep_axis]) do
      opts = [keep_axis: keep_axis == 1]
      opts = if axis < 0, do: opts, else: Keyword.put(opts, :axis, axis)
      apply(Nx, unquote(op), [tensor, opts])
    end
  end

  # dot — iattrs = [n_ca, ca…, n_cb, cb…, n_ba, ba…, n_bb, bb…]
  defp dispatch(:dot, [left, right], attrs) do
    {n_ca, attrs} = List.pop_at(attrs, 0)
    {ca, attrs} = Enum.split(attrs, n_ca)
    {n_cb, attrs} = List.pop_at(attrs, 0)
    {cb, attrs} = Enum.split(attrs, n_cb)
    {n_ba, attrs} = List.pop_at(attrs, 0)
    {ba, attrs} = Enum.split(attrs, n_ba)
    {_n_bb, attrs} = List.pop_at(attrs, 0)
    bb = attrs
    # inputs already cast to computation_type; Nx.dot does its own type promotion
    Nx.dot(left, ca, ba, right, cb, bb)
  end

  # conv_general — calls EMLX directly since there is no Nx public API for
  # an already-transposed conv.  Inputs are %Nx.Tensor{data: %EMLX.Backend{}}.
  # iattrs = [n_dims, s…, pl0,ph0,…, kd…, id…, fgs]
  defp dispatch(:conv_general, [input, kernel], attrs) do
    [n_dims | rest] = attrs
    strides = Enum.slice(rest, 0, n_dims)
    off = n_dims
    padding_lo = for i <- 0..(n_dims - 1), do: Enum.at(rest, off + i * 2)
    padding_hi = for i <- 0..(n_dims - 1), do: Enum.at(rest, off + i * 2 + 1)
    off = off + n_dims * 2
    kernel_dilation = Enum.slice(rest, off, n_dims)
    off = off + n_dims
    input_dilation = Enum.slice(rest, off, n_dims)
    fgs = Enum.at(rest, off + n_dims)

    in_mx = input.data.ref
    kern_mx = kernel.data.ref

    result_mx =
      EMLX.conv_general(
        in_mx,
        kern_mx,
        strides,
        padding_lo,
        padding_hi,
        kernel_dilation,
        input_dilation,
        fgs
      )

    shape = EMLX.shape(result_mx) |> List.to_tuple()

    EMLX.Backend.to_nx(result_mx, %Nx.Tensor{
      type: input.type,
      shape: shape,
      names: List.duplicate(nil, tuple_size(shape))
    })
  end

  # indexing / selection
  defp dispatch(:select, [pred, on_true, on_false], []),
    do: Nx.select(pred, on_true, on_false)

  defp dispatch(:clip, [tensor, min_t, max_t], []),
    do: Nx.clip(tensor, min_t, max_t)

  # slice: iattrs = [n_dims, dyn_mask, d0..dn-1, l0..ln-1, str0..strn-1, sv0..svn-1]
  # Dynamic starts are extra operands after the tensor.
  defp dispatch(:slice, [tensor | dyn_starts], attrs) do
    [n_dims, _dyn_mask | rest] = attrs
    input_shape = Enum.slice(rest, 0, n_dims)
    lengths = Enum.slice(rest, n_dims, n_dims)
    strides = Enum.slice(rest, 2 * n_dims, n_dims)
    sv = Enum.slice(rest, 3 * n_dims, n_dims)
    dyn_mask = Enum.at(attrs, 1)

    {starts, _} =
      Enum.reduce(Enum.with_index(sv), {[], 0}, fn {sv_i, i}, {acc, dyn_idx} ->
        start_i =
          if (dyn_mask >>> i &&& 1) == 1 do
            Nx.to_number(Enum.at(dyn_starts, dyn_idx))
          else
            sv_i
          end

        clamped = min(max(start_i, 0), Enum.at(input_shape, i) - Enum.at(lengths, i))
        dyn_next = if (dyn_mask >>> i &&& 1) == 1, do: dyn_idx + 1, else: dyn_idx
        {acc ++ [clamped], dyn_next}
      end)

    stops = Enum.zip_with(starts, lengths, &(&1 + &2))
    # Nx.slice takes starts + lengths; we ignore strides for the interpreter
    # (Nx.slice doesn't expose strides in the public API so we call backend slice)
    start_indices = starts
    _ = stops
    _ = strides
    Nx.slice(tensor, start_indices, lengths, strides: strides)
  end

  # put_slice: iattrs = [n_dims, dyn_mask, d0..dn-1, l0..ln-1, sv0..svn-1]
  # Operands: [input, slice, dyn_starts…]
  defp dispatch(:put_slice, [input, slice | dyn_starts], attrs) do
    [n_dims, dyn_mask | rest] = attrs
    input_shape = Enum.slice(rest, 0, n_dims)
    lengths = Enum.slice(rest, n_dims, n_dims)
    sv = Enum.slice(rest, 2 * n_dims, n_dims)

    {starts, _} =
      Enum.reduce(Enum.with_index(sv), {[], 0}, fn {sv_i, i}, {acc, dyn_idx} ->
        start_i =
          if (dyn_mask >>> i &&& 1) == 1 do
            Nx.to_number(Enum.at(dyn_starts, dyn_idx))
          else
            sv_i
          end

        clamped = min(max(start_i, 0), Enum.at(input_shape, i) - Enum.at(lengths, i))
        dyn_next = if (dyn_mask >>> i &&& 1) == 1, do: dyn_idx + 1, else: dyn_idx
        {acc ++ [clamped], dyn_next}
      end)

    Nx.put_slice(input, starts, slice)
  end

  # gather: iattrs = [n_gather_axes, a0…, n_tensor_dims, ss0…, n_out_dims, od0…]
  # Operands: [tensor, indices]
  defp dispatch(:gather, [tensor, indices], attrs) do
    [n_gather_axes | rest] = attrs
    axes = Enum.slice(rest, 0, n_gather_axes)
    [_n_tensor_dims | rest2] = Enum.drop(rest, n_gather_axes)
    [n_out_dims | rest3] = Enum.drop(rest2, n_gather_axes + 1)
    # rest2 starts with n_tensor_dims, then slice_sizes; skip n_tensor_dims count
    _slice_sizes = Enum.slice(rest2, 1, length(rest2) - 1 - 1 - n_out_dims)
    out_shape = Enum.take(rest3, n_out_dims) |> List.to_tuple()
    _ = out_shape
    Nx.gather(tensor, indices, axes: axes)
  end

  defp dispatch(:take, [tensor, indices], [axis]),
    do: Nx.take(tensor, indices, axis: axis)

  defp dispatch(:take_along_axis, [tensor, indices], [axis]),
    do: Nx.take_along_axis(tensor, indices, axis: axis)

  # indexed_add: iattrs = [n_axes, a0…, n_updates_shape, us0…]
  # The interpreter calls the public Nx API with original updates (no pre-reshape).
  defp dispatch(:indexed_add, [target, indices, updates], attrs) do
    [n_axes | rest] = attrs
    axes = Enum.slice(rest, 0, n_axes)
    Nx.indexed_add(target, indices, updates, axes: axes)
  end

  defp dispatch(:indexed_put, [target, indices, updates], attrs) do
    [n_axes | rest] = attrs
    axes = Enum.slice(rest, 0, n_axes)
    _ = rest
    Nx.indexed_put(target, indices, updates, axes: axes)
  end

  defp dispatch(op, _args, _attrs),
    do: raise(ArgumentError, "Native.Expr.Interpreter: unknown op #{inspect(op)}")
end
