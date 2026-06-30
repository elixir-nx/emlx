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
  | `:sort`         | `[axis, asc_int]` — axis (non-negative), 1=asc / 0=desc. NaN-aware (matches EMLX.Backend). |
  | `:argsort`      | `[axis, asc_int]` — same; C++ returns sorted uint32 indices.         |
  | `:window_sum`, `:window_product`, `:window_max`, `:window_min` | `[n_dims, op_int, lo0, hi0, …, s0, …, w0, …, wd0, …]` — op_int: 0=sum 1=product 2=max 3=min; then n_dims lo/hi pairs, strides, window dims, window dilations. Operands: `[tensor]`. |
  | `:window_scatter_max`, `:window_scatter_min` | `[n_dims, lo0, hi0, …, s0, …, w0, …]` — n_dims lo/hi pairs, strides, window dims. Operands: `[tensor_t, source, init_value]`. |
  | `:cumulative_sum`, `:cumulative_product`, `:cumulative_min`, `:cumulative_max` | `[axis, reverse_int]` — axis (non-negative), 0/1 reverse. Always inclusive. |
  | `:fft`, `:ifft` | `[axis, n]` — axis and FFT length.                                  |
  | `:fft2`, `:ifft2` | `[ax0, ax1, n0, n1]` — two axes and two lengths.                |
  | `:iota`   | `[dtype_int, n_dims, axis_int, d0..dn-1]` — dtype, rank, axis (−1=flat), shape dims. No operands. |
  | `:eye`    | `[dtype_int, m, n]` — dtype and the two shape dims. No operands.    |

  Non-negative axes: the lowerer normalises negative axis values before encoding
  so C++ handlers can use them directly as 0-based indices.

  `:pad` raises for `interior > 0` or negative `lo`/`hi` (not yet lowered).
  `:reduce` (custom-fun reduce) raises — deferred to Stage 08 (requires child programs).
  Unrecognized `Nx.Block.*` structs descend into `default_expr` (primitive decomposition).
  `Nx.Random.*` functions decompose via `threefry2x32` into primitive ops (bitwise, add, iota)
  and work automatically once `:iota` is lowered.
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
         %T{
           type: out_type,
           data: %Nx.Defn.Expr{id: id, op: :select, args: [pred, on_true, on_false]}
         },
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
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :slice,
             args: [tensor, start_indices, lengths, strides]
           }
         },
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
         %T{
           shape: out_shape,
           data: %Nx.Defn.Expr{id: id, op: :gather, args: [tensor, indices, opts]}
         },
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

  # ── sort / argsort ────────────────────────────────────────────────────────

  # sort: args = [tensor, opts], opts has :axis and :direction.
  # iattrs = [axis, asc_int]  (1 = ascending, 0 = descending)
  # C++ replicates EMLX.Backend.sort NaN-aware algorithm.
  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :sort, args: [tensor, opts]}},
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    axis = opts[:axis]
    norm_axis = if axis < 0, do: tuple_size(tensor.shape) + axis, else: axis
    asc_int = if opts[:direction] == :asc, do: 1, else: 0

    state = %{
      state
      | instructions: [{ref, :sort, [tensor_ref], [norm_axis, asc_int]} | state.instructions]
    }

    {result_ref, state} = emit_cast_if_needed(ref, tensor.type, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # argsort: args = [tensor, opts], opts has :axis and :direction.
  # iattrs = [axis, asc_int]
  # MLX returns uint32; always cast to out_type.
  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :argsort, args: [tensor, opts]}},
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    axis = opts[:axis]
    norm_axis = if axis < 0, do: tuple_size(tensor.shape) + axis, else: axis
    asc_int = if opts[:direction] == :asc, do: 1, else: 0

    state = %{
      state
      | instructions: [{ref, :argsort, [tensor_ref], [norm_axis, asc_int]} | state.instructions]
    }

    {result_ref, state} = emit_cast_if_needed(ref, {:u, 32}, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # ── window reductions ─────────────────────────────────────────────────────

  # window_sum/max/min/product: args = [tensor, window_dims_tuple, opts].
  # opts has :padding (list of {lo, hi} per dim), :strides, :window_dilations.
  #
  # iattrs = [n_dims, op_int, lo0, hi0, …, s0, …, w0, …, wd0, …]
  #   op_int: 0=sum, 1=product, 2=max, 3=min
  #   lo/hi pairs: padding per dim (2*n_dims values)
  #   s0…: strides per dim
  #   w0…: window dims per dim
  #   wd0…: window dilations per dim
  # Operands: [tensor_ref]
  @window_op_int %{window_sum: 0, window_product: 1, window_max: 2, window_min: 3}

  for op <- [:window_sum, :window_product, :window_max, :window_min] do
    defp expand_node(
           %T{
             type: out_type,
             data: %Nx.Defn.Expr{
               id: id,
               op: unquote(op),
               args: [tensor, window_dims_tuple, opts]
             }
           },
           state
         ) do
      ref = make_ref()
      tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
      n_dims = tuple_size(window_dims_tuple)
      window_dims = Tuple.to_list(window_dims_tuple)
      {low_pads, high_pads} = Enum.unzip(opts[:padding])
      strides = opts[:strides] || List.duplicate(1, n_dims)
      window_dilations = opts[:window_dilations] || List.duplicate(1, n_dims)
      op_int = @window_op_int[unquote(op)]

      iattrs =
        [n_dims, op_int] ++
          Enum.flat_map(0..(n_dims - 1), fn i ->
            [Enum.at(low_pads, i), Enum.at(high_pads, i)]
          end) ++
          strides ++ window_dims ++ window_dilations

      state = %{
        state
        | instructions: [{ref, unquote(op), [tensor_ref], iattrs} | state.instructions]
      }

      {result_ref, state} = emit_cast_if_needed(ref, tensor.type, out_type, state)
      %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
    end
  end

  # window_scatter_max / window_scatter_min:
  # args = [tensor_t, source, init_value, window_dims_tuple, opts].
  # opts has :padding, :strides.
  #
  # iattrs = [n_dims, lo0, hi0, …, s0, …, w0, …]
  # Operands: [tensor_t_ref, source_ref, init_value_ref]
  for op <- [:window_scatter_max, :window_scatter_min] do
    defp expand_node(
           %T{
             data: %Nx.Defn.Expr{
               id: id,
               op: unquote(op),
               args: [tensor_t, source, init_value, window_dims_tuple, opts]
             }
           },
           state
         ) do
      ref = make_ref()
      t_ref = Map.fetch!(state.node_to_ref, tensor_t.data.id)
      src_ref = Map.fetch!(state.node_to_ref, source.data.id)
      init_ref = Map.fetch!(state.node_to_ref, init_value.data.id)

      n_dims = tuple_size(window_dims_tuple)
      window_dims = Tuple.to_list(window_dims_tuple)
      {low_pads, high_pads} = Enum.unzip(opts[:padding])
      strides = opts[:strides] || List.duplicate(1, n_dims)

      iattrs =
        [n_dims] ++
          Enum.flat_map(0..(n_dims - 1), fn i ->
            [Enum.at(low_pads, i), Enum.at(high_pads, i)]
          end) ++
          strides ++ window_dims

      %{
        state
        | instructions: [
            {ref, unquote(op), [t_ref, src_ref, init_ref], iattrs} | state.instructions
          ],
          node_to_ref: Map.put(state.node_to_ref, id, ref)
      }
    end
  end

  # ── Nx.Block.Cumulative* — recognize-struct path ─────────────────────────
  #
  # Cumulative ops surface as Nx.Block.Cumulative{Sum,Product,Min,Max}.
  # The struct carries :axis and :reverse (both already resolved to non-negative
  # axis and boolean by Nx.cumulative_op).
  #
  # iattrs = [axis, reverse_int]  (0/1 booleans)
  # inclusive is always 1 (MLX inclusive mode matches Nx semantics).
  for {block_mod, op} <- [
        {Nx.Block.CumulativeSum, :cumulative_sum},
        {Nx.Block.CumulativeProduct, :cumulative_product},
        {Nx.Block.CumulativeMin, :cumulative_min},
        {Nx.Block.CumulativeMax, :cumulative_max}
      ] do
    defp expand_node(
           %T{
             type: out_type,
             data: %Nx.Defn.Expr{
               id: id,
               op: :block,
               args: [%unquote(block_mod){axis: axis, reverse: reverse}, [tensor], _default, _fun]
             }
           },
           state
         ) do
      ref = make_ref()
      tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
      norm_axis = if axis < 0, do: tuple_size(tensor.shape) + axis, else: axis
      reverse_int = if reverse, do: 1, else: 0

      state = %{
        state
        | instructions: [
            {ref, unquote(op), [tensor_ref], [norm_axis, reverse_int]} | state.instructions
          ]
      }

      {result_ref, state} = emit_cast_if_needed(ref, tensor.type, out_type, state)
      %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
    end
  end

  # ── fft / ifft ────────────────────────────────────────────────────────────

  # fft/ifft: args = [tensor, opts], opts has :length and :axis (already resolved).
  # iattrs = [axis, n]  where n is the FFT length (positive int).
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :fft, args: [tensor, opts]}},
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    axis = opts[:axis]
    n = opts[:length]

    %{
      state
      | instructions: [{ref, :fft, [tensor_ref], [axis, n]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :ifft, args: [tensor, opts]}},
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    axis = opts[:axis]
    n = opts[:length]

    %{
      state
      | instructions: [{ref, :ifft, [tensor_ref], [axis, n]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # ── Nx.Block.FFT2 / IFFT2 — recognize-struct path ─────────────────────────
  #
  # fft2/ifft2: Nx.Block.FFT2/IFFT2{lengths: [n0, n1], axes: [ax0, ax1]}.
  # iattrs = [ax0, ax1, n0, n1]
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.FFT2{lengths: lengths, axes: axes}, [tensor], _default, _fun]
           }
         },
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    [ax0, ax1] = axes || [-2, -1]
    rank = tuple_size(tensor.shape)
    ax0 = if ax0 < 0, do: rank + ax0, else: ax0
    ax1 = if ax1 < 0, do: rank + ax1, else: ax1
    [n0, n1] = lengths || [elem(tensor.shape, ax0), elem(tensor.shape, ax1)]

    %{
      state
      | instructions: [{ref, :fft2, [tensor_ref], [ax0, ax1, n0, n1]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.IFFT2{lengths: lengths, axes: axes}, [tensor], _default, _fun]
           }
         },
         state
       ) do
    ref = make_ref()
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    [ax0, ax1] = axes || [-2, -1]
    rank = tuple_size(tensor.shape)
    ax0 = if ax0 < 0, do: rank + ax0, else: ax0
    ax1 = if ax1 < 0, do: rank + ax1, else: ax1
    [n0, n1] = lengths || [elem(tensor.shape, ax0), elem(tensor.shape, ax1)]

    %{
      state
      | instructions: [{ref, :ifft2, [tensor_ref], [ax0, ax1, n0, n1]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # ── Nx.Block.LinAlg.* — recognize-struct native path ─────────────────────
  #
  # MLX provides native (CPU-only) linalg primitives. We emit a native op that
  # mirrors EMLX.Backend's eager path: cast the operand(s) to f32, run the
  # primitive, cast the result back to the block's output type. The op runs on
  # the CPU stream in C++ (pinned), so it composes inside the compiled graph.
  #
  # cholesky: single-output. operands = [a]; no attrs.
  defp expand_node(
         %T{
           type: out_type,
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.LinAlg.Cholesky{}, [tensor], _default, _fun]
           }
         },
         state
       ) do
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    {f32_ref, state} = emit_cast_if_needed(tensor_ref, tensor.type, {:f, 32}, state)

    ref = make_ref()
    state = %{state | instructions: [{ref, :cholesky, [f32_ref], []} | state.instructions]}

    {result_ref, state} = emit_cast_if_needed(ref, {:f, 32}, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # solve: single-output. operands = [a, b]; no attrs. Solves A x = b.
  defp expand_node(
         %T{
           type: out_type,
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.LinAlg.Solve{}, [a, b], _default, _fun]
           }
         },
         state
       ) do
    a_ref = Map.fetch!(state.node_to_ref, a.data.id)
    b_ref = Map.fetch!(state.node_to_ref, b.data.id)
    {a_f, state} = emit_cast_if_needed(a_ref, a.type, {:f, 32}, state)
    {b_f, state} = emit_cast_if_needed(b_ref, b.type, {:f, 32}, state)

    ref = make_ref()
    state = %{state | instructions: [{ref, :solve, [a_f, b_f], []} | state.instructions]}

    {result_ref, state} = emit_cast_if_needed(ref, {:f, 32}, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # qr (reduced mode): multi-output [q, r]. operands = [a]; no attrs.
  # :complete mode descends into default_expr (Householder + while) — falls back.
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.LinAlg.QR{mode: :reduced}, [tensor], _default, _fun]
           }
         },
         state
       ) do
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    {f32_ref, state} = emit_cast_if_needed(tensor_ref, tensor.type, {:f, 32}, state)

    q_ref = make_ref()
    r_ref = make_ref()

    state = %{
      state
      | instructions: [{[q_ref, r_ref], :qr, [f32_ref], []} | state.instructions]
    }

    %{state | node_to_ref: Map.put(state.node_to_ref, id, [q_ref, r_ref])}
  end

  # eigh: multi-output [eigenvalues, eigenvectors]. operands = [a]; lower triangle.
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.LinAlg.Eigh{}, [tensor], _default, _fun]
           }
         },
         state
       ) do
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    {f32_ref, state} = emit_cast_if_needed(tensor_ref, tensor.type, {:f, 32}, state)

    w_ref = make_ref()
    v_ref = make_ref()

    state = %{
      state
      | instructions: [{[w_ref, v_ref], :eigh, [f32_ref], []} | state.instructions]
    }

    %{state | node_to_ref: Map.put(state.node_to_ref, id, [w_ref, v_ref])}
  end

  # svd (full matrices): multi-output [u, s, vt]. operands = [a]; no attrs.
  # full_matrices?: false descends into default_expr (Jacobi + while) — falls back.
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.LinAlg.SVD{full_matrices?: true}, [tensor], _default, _fun]
           }
         },
         state
       ) do
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    {f32_ref, state} = emit_cast_if_needed(tensor_ref, tensor.type, {:f, 32}, state)

    u_ref = make_ref()
    s_ref = make_ref()
    vt_ref = make_ref()

    state = %{
      state
      | instructions: [{[u_ref, s_ref, vt_ref], :svd, [f32_ref], []} | state.instructions]
    }

    %{state | node_to_ref: Map.put(state.node_to_ref, id, [u_ref, s_ref, vt_ref])}
  end

  # lu: multi-output {P, L, U}. operands = [a]; no attrs.
  # MLX returns a pivot index vector; we rebuild the permutation matrix in-graph
  # via eye + take (mirroring EMLX.Backend.block/4 for LU).
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [%Nx.Block.LinAlg.LU{}, [tensor], _default, _fun]
           }
         },
         state
       ) do
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    {f32_ref, state} = emit_cast_if_needed(tensor_ref, tensor.type, {:f, 32}, state)

    piv_ref = make_ref()
    l_ref = make_ref()
    u_ref = make_ref()

    state = %{
      state
      | instructions: [{[piv_ref, l_ref, u_ref], :lu, [f32_ref], []} | state.instructions]
    }

    # P = take(eye(n), pivots, axis: 0). n is the trailing matrix dimension.
    n = elem(tensor.shape, tuple_size(tensor.shape) - 1)
    f32_int = Map.fetch!(@mlx_type_to_int, EMLX.Native.to_mlx_type({:f, 32}))

    eye_ref = make_ref()
    p_ref = make_ref()

    state = %{
      state
      | instructions: [
          {p_ref, :take, [eye_ref, piv_ref], [0]},
          {eye_ref, :eye, [], [f32_int, n, n]}
          | state.instructions
        ]
    }

    %{state | node_to_ref: Map.put(state.node_to_ref, id, [p_ref, l_ref, u_ref])}
  end

  # triangular_solve: single-output. Direct op node (not a block).
  # args = [a, b, opts]. Only the common configuration (left_side + no transform)
  # is lowered natively; other variants raise → Evaluator fallback.
  defp expand_node(
         %T{
           type: out_type,
           data: %Nx.Defn.Expr{id: id, op: :triangular_solve, args: [a, b, opts]}
         },
         state
       ) do
    left_side = Keyword.get(opts, :left_side, true)
    transform_a = Keyword.get(opts, :transform_a, :none)
    lower = Keyword.get(opts, :lower, true)

    unless left_side and transform_a == :none do
      raise ArgumentError,
            "does not yet lower op :triangular_solve with " <>
              "left_side=#{inspect(left_side)} transform_a=#{inspect(transform_a)}"
    end

    a_ref = Map.fetch!(state.node_to_ref, a.data.id)
    b_ref = Map.fetch!(state.node_to_ref, b.data.id)
    {a_f, state} = emit_cast_if_needed(a_ref, a.type, {:f, 32}, state)
    {b_f, state} = emit_cast_if_needed(b_ref, b.type, {:f, 32}, state)

    upper_int = if lower, do: 0, else: 1
    ref = make_ref()

    state = %{
      state
      | instructions: [{ref, :solve_triangular, [a_f, b_f], [upper_int]} | state.instructions]
    }

    {result_ref, state} = emit_cast_if_needed(ref, {:f, 32}, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # ── block fallback: descend into default_expr ─────────────────────────────
  #
  # For any Nx.Block.* struct not specifically recognized above (e.g. RFFT,
  # IRFFT, AllClose, Phase, unrecognized future blocks), lower the block's
  # traced default implementation instead of raising.
  #
  # The default_expr was traced by expr_block using fresh :parameter nodes as
  # stand-ins for the in_args. We map those inner params to the parent-scope
  # refs for in_args, then expand the inner scope's nodes inline.
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :block,
             args: [_struct, in_args, default_expr, _fun]
           }
         },
         state
       ) do
    expand_block_via_default(id, in_args, default_expr, state)
  end

  # ── cond: lower as nested :select ops ────────────────────────────────────
  #
  # All cond predicates and bodies are in the parent scope: Nx's apply_args
  # for :cond traverses everything (no :scope vs :all distinction), so every
  # pred and body tensor is already in node_to_ref when we reach the :cond
  # node in the topo order.
  #
  # Strategy: for each output element index i, right-fold the clauses:
  #   select(pred1, body1_i, select(pred2, body2_i, ..., select(predN, bodyN_i, last_i)))
  #
  # Single-tensor output  → store one ref in node_to_ref.
  # Tuple output (type {:tuple, n}) → store a list of n refs in node_to_ref.
  # :elem nodes that follow pick the correct element from that list.
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :cond, args: [clauses, last]}},
         state
       ) do
    last_refs = flat_refs(last, state)

    clause_ref_pairs =
      Enum.map(clauses, fn {pred, body} ->
        {Map.fetch!(state.node_to_ref, pred.data.id), flat_refs(body, state)}
      end)

    n = length(last_refs)

    {per_elem_refs, state} =
      Enum.reduce(0..(n - 1), {[], state}, fn i, {elem_results, st} ->
        last_ref_i = Enum.at(last_refs, i)

        # Right-fold: most-priority clause wraps the least-priority accumulator.
        {result_ref, st} =
          Enum.reduce(Enum.reverse(clause_ref_pairs), {last_ref_i, st}, fn {pred_ref,
                                                                             body_refs},
                                                                            {acc_ref, st2} ->
            body_ref_i = Enum.at(body_refs, i)
            ref = make_ref()
            st2 = %{st2 | instructions: [{ref, :select, [pred_ref, body_ref_i, acc_ref], []} | st2.instructions]}
            {ref, st2}
          end)

        {elem_results ++ [result_ref], st}
      end)

    node_val = if n == 1, do: hd(per_elem_refs), else: per_elem_refs
    %{state | node_to_ref: Map.put(state.node_to_ref, id, node_val)}
  end

  # ── elem: extract element from a tuple-output op (cond/while) ─────────────
  #
  # Tuple-output ops (e.g. a :cond or :while with tuple carry) store a LIST of
  # refs in node_to_ref. :elem picks the pos-th element from that list.
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :elem, args: [tuple_node, pos]}},
         state
       ) do
    case Map.fetch!(state.node_to_ref, tuple_node.data.id) do
      refs when is_list(refs) ->
        elem_ref = Enum.at(refs, pos)
        %{state | node_to_ref: Map.put(state.node_to_ref, id, elem_ref)}

      single_ref when pos == 0 ->
        # Degenerate single-element tuple — shouldn't normally appear.
        %{state | node_to_ref: Map.put(state.node_to_ref, id, single_ref)}
    end
  end

  # ── creation ops ─────────────────────────────────────────────────────────

  # iota: no tensor operands; all info in iattrs.
  # iattrs = [dtype_int, n_dims, axis_int, d0..dn-1]
  # axis_int = -1 encodes nil (flat enumeration across all dims).
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :iota, args: [axis]}} = node,
         state
       ) do
    ref = make_ref()
    shape = Tuple.to_list(node.shape)
    n_dims = length(shape)
    dtype_int = Map.fetch!(@mlx_type_to_int, EMLX.Native.to_mlx_type(node.type))
    axis_int = if axis == nil, do: -1, else: axis

    %{
      state
      | instructions: [
          {ref, :iota, [], [dtype_int, n_dims, axis_int | shape]} | state.instructions
        ],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # eye: no tensor operands; iattrs = [dtype_int, m, n].
  # Output shape is always {m, n}.
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :eye, args: []}} = node,
         state
       ) do
    ref = make_ref()
    [m, n] = Tuple.to_list(node.shape)
    dtype_int = Map.fetch!(@mlx_type_to_int, EMLX.Native.to_mlx_type(node.type))

    %{
      state
      | instructions: [{ref, :eye, [], [dtype_int, m, n]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
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
      | instructions: [
          {ref, op, [target_ref, indices_ref, updates_ref], iattrs} | state.instructions
        ],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # ── block-descent helper ──────────────────────────────────────────────────

  # Lower a :block node via its traced default_expr (the primitive decomposition).
  #
  # Nx.Defn.Expr.expr_block creates fresh :parameter nodes for the block's fun
  # and passes them into the fun instead of the real in_args.  The default_expr
  # therefore has inner :parameter nodes (at positions 0, 1, …) whose IDs are
  # distinct from the parent-scope in_args IDs.
  #
  # We:
  #   1. topo-sort the inner scope from default_expr.
  #   2. Find the inner :parameter nodes and map them → parent-scope refs.
  #   3. Expand the inner scope nodes (skipping the already-mapped params).
  #   4. Alias the block node's output to the default_expr's result ref.
  defp expand_block_via_default(id, in_args, default_expr, state) do
    inner_ordered = EMLX.Defn.Tree.post_order(default_expr)

    # Collect inner :parameter nodes; sort by position (args[0]).
    inner_params =
      inner_ordered
      |> Enum.filter(&(&1.data.op == :parameter))
      |> Enum.sort_by(fn t -> hd(t.data.args) end)

    # Map each inner param id → the corresponding parent-scope arg ref.
    parent_arg_refs = Enum.map(in_args, &Map.fetch!(state.node_to_ref, &1.data.id))

    inner_param_id_set =
      inner_params
      |> Enum.zip(parent_arg_refs)
      |> Enum.reduce(MapSet.new(), fn {param, _}, acc ->
        MapSet.put(acc, param.data.id)
      end)

    inner_param_ref_map =
      inner_params
      |> Enum.zip(parent_arg_refs)
      |> Map.new(fn {param, ref} -> {param.data.id, ref} end)

    # Extend node_to_ref with inner param → parent ref mappings.
    merged_node_to_ref = Map.merge(state.node_to_ref, inner_param_ref_map)
    inner_state = %{state | node_to_ref: merged_node_to_ref}

    # Expand inner scope, skipping inner :parameter nodes (already mapped).
    inner_state =
      Enum.reduce(inner_ordered, inner_state, fn node, st ->
        if MapSet.member?(inner_param_id_set, node.data.id) do
          st
        else
          expand_node(node, st)
        end
      end)

    result_ref = Map.fetch!(inner_state.node_to_ref, default_expr.data.id)
    %{inner_state | node_to_ref: Map.put(inner_state.node_to_ref, id, result_ref)}
  end

  # ── cond helper ───────────────────────────────────────────────────────────

  # Flatten a composite (single tensor or Elixir tuple of tensors) to a list
  # of refs looked up in node_to_ref, one per leaf tensor.
  defp flat_refs(composite, state) do
    Composite.flatten_list([composite])
    |> Enum.map(&Map.fetch!(state.node_to_ref, &1.data.id))
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
    # A `result` ref is indexed into the C++ flat results accumulator. Each
    # instruction contributes one entry (single-output) or several (multi-output
    # ops whose result field is a list of refs), so the flat index is tracked
    # separately from the instruction position.
    {op_names, operands, iattrs, ref_to_packed, _flat} =
      prog.instructions
      |> Enum.reduce({[], [], [], ref_to_packed, 0}, fn {id, op, operand_refs, attrs},
                                                        {ops, ors, ias, rmap, flat} ->
        wire_operands = Enum.map(operand_refs, &Map.fetch!(rmap, &1))

        {rmap2, flat2} =
          case id do
            ids when is_list(ids) ->
              Enum.reduce(ids, {rmap, flat}, fn one, {m, f} ->
                {Map.put(m, one, @kind_instr <<< @kind_shift ||| f), f + 1}
              end)

            one ->
              {Map.put(rmap, one, @kind_instr <<< @kind_shift ||| flat), flat + 1}
          end

        {[op | ops], [wire_operands | ors], [attrs | ias], rmap2, flat2}
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
        result = dispatch(op, args, attrs)

        # Multi-output ops carry a list of result refs; bind each to its output.
        case id do
          ids when is_list(ids) ->
            Enum.zip(ids, result)
            |> Enum.reduce(env, fn {one, val}, e -> Map.put(e, one, val) end)

          one ->
            Map.put(env, one, result)
        end
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

  # creation ops — no tensor operands

  # iota: attrs = [dtype_int, n_dims, axis_int, d0..dn-1]
  # axis_int = -1 means nil (flat enumeration).
  defp dispatch(:iota, [], [dtype_int, n_dims, axis_int | shape_rest]) do
    shape = shape_rest |> Enum.take(n_dims) |> List.to_tuple()
    type = Expr.int_to_nx_type(dtype_int)
    opts = [type: type, backend: EMLX.Backend]
    opts = if axis_int >= 0, do: Keyword.put(opts, :axis, axis_int), else: opts
    Nx.iota(shape, opts)
  end

  # eye: attrs = [dtype_int, m, n]
  defp dispatch(:eye, [], [dtype_int, m, n]) do
    type = Expr.int_to_nx_type(dtype_int)
    Nx.eye({m, n}, type: type, backend: EMLX.Backend)
  end

  # linalg — operands already f32 (lowerer casts). Mirror the native C++ ops.
  defp dispatch(:cholesky, [a], []), do: Nx.LinAlg.cholesky(a)
  defp dispatch(:solve, [a, b], []), do: Nx.LinAlg.solve(a, b)

  defp dispatch(:solve_triangular, [a, b], [upper_int]),
    do: Nx.LinAlg.triangular_solve(a, b, lower: upper_int == 0)

  defp dispatch(:qr, [a], []) do
    {q, r} = Nx.LinAlg.qr(a)
    [q, r]
  end

  defp dispatch(:eigh, [a], []) do
    {w, v} = Nx.LinAlg.eigh(a)
    [w, v]
  end

  defp dispatch(:svd, [a], []) do
    {u, s, vt} = Nx.LinAlg.svd(a)
    [u, s, vt]
  end

  # :lu returns the raw MLX outputs [pivots, l, u]; the lowered program rebuilds
  # the permutation matrix from `pivots` via separate :eye / :take instructions.
  defp dispatch(:lu, [a], []) do
    [piv, l, u] = EMLX.linalg_lu(EMLX.Backend.from_nx(a))
    [EMLX.Backend.to_nx(piv), EMLX.Backend.to_nx(l), EMLX.Backend.to_nx(u)]
  end

  defp dispatch(op, _args, _attrs),
    do: raise(ArgumentError, "Native.Expr.Interpreter: unknown op #{inspect(op)}")
end
