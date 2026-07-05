defmodule EMLX.Native.Expr do
  @moduledoc false
  import Bitwise

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T

  @compiler_debug Application.compile_env(:emlx, :compiler_debug, false)
  defmacrop maybe_debug_check(do: block) do
    if @compiler_debug do
      block
    else
      :ok
    end
  end

  @kind_input 0
  @kind_capture 1
  @kind_const 2
  @kind_instr 3
  @kind_shift 60

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

  # Must stay in sync with int_to_quant_mode() in emlx_compiler.cpp.
  @quant_mode_to_int %{
    "affine" => 0,
    "mxfp4" => 1,
    "mxfp8" => 2,
    "nvfp4" => 3
  }

  @enforce_keys [:inputs, :captures, :constants, :instructions, :outputs]
  defstruct [
    :inputs,
    :captures,
    :constants,
    :instructions,
    :outputs,
    hooks: [],
    runtime_calls: []
  ]

  @type node_ref :: reference()
  @type hook :: %{
          name: atom(),
          callback: (Nx.Container.t() -> term()),
          template: Nx.Container.t(),
          refs: [node_ref()]
        }
  @type runtime_call :: %{
          index: non_neg_integer(),
          callback: (Nx.Container.t(), keyword() -> Nx.Container.t()),
          args_template: Nx.Container.t(),
          arg_param_positions: [non_neg_integer() | nil],
          opts: keyword()
        }
  @type t :: %__MODULE__{
          inputs: [node_ref()],
          captures: [{node_ref(), Nx.Tensor.t()}],
          constants: [{node_ref(), number(), Nx.Type.t()}],
          instructions: [{node_ref(), atom(), [node_ref()], [integer()]}],
          outputs: [node_ref()],
          hooks: [hook()],
          runtime_calls: [runtime_call()]
        }

  # ── lowering ──────────────────────────────────────────────────────────────

  @doc false
  @spec lower(
          Nx.Container.t(),
          non_neg_integer() | nil,
          %{non_neg_integer() => EMLX.Quantization.Config.t()}
        ) :: t()
  def lower(output, num_inputs \\ nil, quant_signature \\ %{}) do
    ordered = EMLX.Defn.Tree.post_order(output, &scope_dependencies/1)

    # inputs is a map of pos → ref during lowering; densified to a list at the end.
    state = %{
      inputs: %{},
      captures: [],
      constants: [],
      instructions: [],
      node_to_ref: %{},
      hooks: [],
      runtime_calls: [],
      top_scope_ids: output |> Nx.Defn.Tree.scope_ids() |> Map.keys() |> MapSet.new(),
      quant_signature: quant_signature
    }

    state = Enum.reduce(ordered, state, &expand_node/2)

    max_referenced_pos = state.inputs |> Map.keys() |> Enum.max(fn -> -1 end)
    arity = max(num_inputs || 0, max_referenced_pos + 1)

    inputs_list =
      for pos <- 0..(arity - 1)//1 do
        Map.get_lazy(state.inputs, pos, &make_ref/0)
      end

    flat_outputs = Composite.flatten_list([output])
    output_refs = Enum.map(flat_outputs, &Map.fetch!(state.node_to_ref, &1.data.id))

    %__MODULE__{
      inputs: inputs_list,
      captures: Enum.reverse(state.captures),
      constants: Enum.reverse(state.constants),
      instructions: Enum.reverse(state.instructions),
      outputs: output_refs,
      hooks: Enum.reverse(state.hooks),
      runtime_calls: Enum.reverse(state.runtime_calls)
    }
  end

  # ── node expansion ────────────────────────────────────────────────────────

  @doc false
  def scope_dependencies(%T{
        data: %Nx.Defn.Expr{op: :metadata, args: [_inner, %{__EMLX__: %{operands: operands}}]}
      }) do
    {:ok, operands}
  end

  def scope_dependencies(_node), do: :default

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

    state = %{
      state
      | constants: [{ref, number, node.type} | state.constants]
    }

    if node.shape == {} do
      %{state | node_to_ref: Map.put(state.node_to_ref, id, ref)}
    else
      broadcast_ref = make_ref()
      shape_list = Tuple.to_list(node.shape)
      iattrs = [length(shape_list) | shape_list] ++ [0]

      %{
        state
        | instructions: [{broadcast_ref, :broadcast, [ref], iattrs} | state.instructions],
          node_to_ref: Map.put(state.node_to_ref, id, broadcast_ref)
      }
    end
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
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :metadata,
             args: [_inner, %{__EMLX__: %{op: opcode, operands: operands, attrs: attrs}}]
           }
         },
         state
       ) do
    ref = make_ref()
    operand_refs = Enum.map(operands, &Map.fetch!(state.node_to_ref, &1.data.id))

    %{
      state
      | instructions: [{ref, opcode, operand_refs, attrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :metadata, args: [inner, _meta]}},
         state
       ) do
    inner_ref =
      if is_tuple(inner) do
        inner |> Tuple.to_list() |> Enum.map(&Map.fetch!(state.node_to_ref, &1.data.id))
      else
        Map.fetch!(state.node_to_ref, inner.data.id)
      end

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

  # block: dispatch on struct — handles Nx.Block.LogicalNot.
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

  @binary_arithmetic_ops [:add, :subtract, :multiply, :pow, :left_shift]

  for op <- @binary_arithmetic_ops do
    defp expand_node(
           %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: unquote(op), args: [left, right]}},
           state
         ) do
      expand_binary_node(id, unquote(op), out_type, left, right, state)
    end
  end

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

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :pad, args: [tensor, pad_value, config]}},
         state
       ) do
    tensor_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    pad_value_ref = Map.fetch!(state.node_to_ref, pad_value.data.id)

    {result_ref, state} =
      if Enum.any?(config, fn {lo, hi, interior} -> lo < 0 or hi < 0 or interior > 0 end) do
        expand_pad_general(tensor_ref, pad_value_ref, Tuple.to_list(tensor.shape), config, state)
      else
        ref = make_ref()
        n_dims = length(config)
        iattrs = [n_dims | Enum.flat_map(config, fn {lo, hi, interior} -> [lo, hi, interior] end)]

        {ref,
         %{
           state
           | instructions: [{ref, :pad, [tensor_ref, pad_value_ref], iattrs} | state.instructions]
         }}
      end

    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
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

  defp expand_node(
         %T{
           type: out_type,
           shape: out_shape,
           data: %Nx.Defn.Expr{id: id, op: :reduce, args: [tensor, acc, opts, fun]}
         },
         state
       ) do
    expand_reduce_unroll(id, out_type, out_shape, tensor, acc, opts, fun, state)
  end

  # ── dot ─────────────────────────────────────────────────────────────────────

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
    if quantized_param_config(left, state.quant_signature) do
      raise ArgumentError,
            "does not yet lower op :dot with a quantized left operand. " <>
              "Dequantize it first with EMLX.dequantize/1."
    end

    case quantized_param_config(right, state.quant_signature) do
      nil ->
        expand_plain_dot(id, out_type, left, c_left, right, c_right, b_left, b_right, state)

      cfg ->
        expand_quantized_dot(id, out_type, left, c_left, b_left, right, c_right, cfg, state)
    end
  end

  # ── conv ─────────────────────────────────────────────────────────────────────

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

  defp expand_node(
         %T{
           type: out_type,
           shape: out_shape,
           data: %Nx.Defn.Expr{
             id: id,
             op: :window_reduce,
             args: [tensor, acc, window_dims_tuple, opts, fun]
           }
         },
         state
       ) do
    n_dims = tuple_size(window_dims_tuple)
    window_dims = Tuple.to_list(window_dims_tuple)
    {low_pads, high_pads} = Enum.unzip(opts[:padding])
    strides = opts[:strides] || List.duplicate(1, n_dims)
    dilations = opts[:window_dilations] || List.duplicate(1, n_dims)

    if Enum.any?(low_pads ++ high_pads, &(&1 < 0)) do
      raise ArgumentError, "does not yet lower op :window_reduce with negative padding"
    end

    [params, body, _mfa] = fun.data.args

    unless length(params) == 2 do
      raise ArgumentError, "does not yet lower op :window_reduce with a non-binary reducer"
    end

    # Cast input + acc to the reducer/output type before padding/folding.
    tensor_ref0 = Map.fetch!(state.node_to_ref, tensor.data.id)
    {tensor_ref, state} = emit_cast_if_needed(tensor_ref0, tensor.type, out_type, state)
    acc_ref0 = Map.fetch!(state.node_to_ref, acc.data.id)
    {acc_scalar_ref, state} = emit_cast_if_needed(acc_ref0, acc.type, out_type, state)

    # Pad with acc (interior 0). Padded shape drives the slice input-shape iattr.
    in_dims = Tuple.to_list(tensor.shape)

    padded_shape =
      [in_dims, low_pads, high_pads]
      |> Enum.zip()
      |> Enum.map(fn {d, lo, hi} -> d + lo + hi end)

    {padded_ref, state} =
      if Enum.all?(low_pads ++ high_pads, &(&1 == 0)) do
        {tensor_ref, state}
      else
        emit_pad_with(tensor_ref, acc_scalar_ref, low_pads, high_pads, state)
      end

    out_dims = Tuple.to_list(out_shape)

    {acc_ref, state} = emit_broadcast_to(acc_scalar_ref, out_dims, state)
    extent = Enum.product(window_dims)

    # :slice takes a span (stop = start + length); with a stride it yields
    spans = Enum.zip_with(out_dims, strides, fn d, s -> (d - 1) * s + 1 end)

    {final_ref, state} =
      Enum.reduce(0..(extent - 1)//1, {acc_ref, state}, fn k, {acc_k, st} ->
        offsets = window_offsets(k, window_dims)
        starts = Enum.zip_with(offsets, dilations, &(&1 * &2))
        {slice_ref, st} = emit_static_slice(padded_ref, padded_shape, starts, spans, strides, st)
        lower_fun_body(body, %{0 => slice_ref, 1 => acc_k}, st)
      end)

    %{state | node_to_ref: Map.put(state.node_to_ref, id, final_ref)}
  end

  # ── Nx.Block.Cumulative* — recognize-struct path ─────────────────────────
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
    upper = not lower

    a_ref = Map.fetch!(state.node_to_ref, a.data.id)
    b_ref = Map.fetch!(state.node_to_ref, b.data.id)
    {a_f, state} = emit_cast_if_needed(a_ref, a.type, {:f, 32}, state)
    {b_f, state} = emit_cast_if_needed(b_ref, b.type, {:f, 32}, state)

    a_rank = tuple_size(a.shape)

    {a_op, effective_upper, state} =
      case transform_a do
        :transpose ->
          {a_t, state} = emit_transpose_instr(a_f, swap_last_two_axes(a_rank), state)
          {a_t, not upper, state}

        _ ->
          {a_f, upper, state}
      end

    {ref, state} =
      if left_side do
        emit_solve_triangular_instr(a_op, b_f, effective_upper, state)
      else
        # Solve XA = B → A^T x = b (works for both 1D and 2D b).
        {a_t, state} = emit_transpose_instr(a_op, swap_last_two_axes(a_rank), state)
        b_rank = tuple_size(b.shape)

        if b_rank == 1 do
          emit_solve_triangular_instr(a_t, b_f, not effective_upper, state)
        else
          {b_t, state} = emit_transpose_instr(b_f, swap_last_two_axes(b_rank), state)
          {out_t, state} = emit_solve_triangular_instr(a_t, b_t, not effective_upper, state)
          emit_transpose_instr(out_t, swap_last_two_axes(b_rank), state)
        end
      end

    {result_ref, state} = emit_cast_if_needed(ref, {:f, 32}, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # ── block fallback: descend into default_expr ─────────────────────────────
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
          Enum.reduce(Enum.reverse(clause_ref_pairs), {last_ref_i, st}, fn {pred_ref, body_refs},
                                                                           {acc_ref, st2} ->
            body_ref_i = Enum.at(body_refs, i)
            ref = make_ref()

            st2 = %{
              st2
              | instructions: [
                  {ref, :select, [pred_ref, body_ref_i, acc_ref], []} | st2.instructions
                ]
            }

            {ref, st2}
          end)

        {elem_results ++ [result_ref], st}
      end)

    node_val = if n == 1, do: hd(per_elem_refs), else: per_elem_refs
    %{state | node_to_ref: Map.put(state.node_to_ref, id, node_val)}
  end

  # ── elem: extract element from a tuple-output op (cond/while) ─────────────
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

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :eye, args: []}} = node,
         state
       ) do
    ref = make_ref()
    shape_list = Tuple.to_list(node.shape)
    n_dims = length(shape_list)
    [m, n] = Enum.take(shape_list, -2)
    dtype_int = Map.fetch!(@mlx_type_to_int, EMLX.Native.to_mlx_type(node.type))

    state = %{
      state
      | instructions: [{ref, :eye, [], [dtype_int, m, n]} | state.instructions]
    }

    if n_dims == 2 do
      %{state | node_to_ref: Map.put(state.node_to_ref, id, ref)}
    else
      broadcast_ref = make_ref()
      axes = [n_dims - 2, n_dims - 1]
      iattrs = [n_dims | shape_list] ++ [length(axes) | axes]

      %{
        state
        | instructions: [{broadcast_ref, :broadcast, [ref], iattrs} | state.instructions],
          node_to_ref: Map.put(state.node_to_ref, id, broadcast_ref)
      }
    end
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: :fun}}, state), do: state

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :token, args: [%Nx.Defn.Token{hooks: hooks}]}},
         state
       ) do
    unless MapSet.member?(state.top_scope_ids, id) do
      raise ArgumentError,
            "cannot lower a hook nested inside a cond branch: EMLX's cond compiles by " <>
              "evaluating every branch unconditionally (:select), which would fire this " <>
              "hook on every call regardless of which branch is actually taken -- a " <>
              "behavior divergence from Nx.Defn.Evaluator (which only fires the selected " <>
              "branch's hook). Move the hook outside the cond."
    end

    Enum.reduce(hooks, state, fn
      %{callback: nil}, state ->
        state

      %{callback: callback, expr: expr, name: name}, state ->
        refs =
          [expr]
          |> Composite.flatten_list()
          |> Enum.map(&Map.fetch!(state.node_to_ref, &1.data.id))

        template = Composite.traverse(expr, &Nx.to_template/1)
        hook = %{name: name, callback: callback, template: template, refs: refs}
        %{state | hooks: [hook | state.hooks]}
    end)
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :attach_token, args: [_token, expr]}},
         state
       ) do
    ref = Map.fetch!(state.node_to_ref, expr.data.id)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, ref)}
  end

  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :runtime_call,
             args: [tensor_expr, callback, out_template, opts]
           }
         },
         state
       ) do
    unless MapSet.member?(state.top_scope_ids, id) do
      raise ArgumentError,
            "cannot lower a runtime_call nested inside a cond branch: EMLX's cond compiles " <>
              "by evaluating every branch unconditionally (:select), which would fire this " <>
              "runtime_call's callback on every call regardless of which branch is actually " <>
              "taken -- a behavior divergence from Nx.Defn.Evaluator (which only fires the " <>
              "selected branch's callback). Move the runtime_call outside the cond."
    end

    operand_leaves = Composite.flatten_list([tensor_expr])

    operand_refs = Enum.map(operand_leaves, &Map.fetch!(state.node_to_ref, &1.data.id))

    arg_param_positions =
      Enum.map(operand_leaves, fn
        %T{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}} -> pos
        _ -> nil
      end)

    output_templates =
      case out_template do
        %Nx.Tensor{} = t -> [t]
        container -> Composite.flatten_list([container])
      end

    callback_index = length(state.runtime_calls)

    iattrs =
      [callback_index, length(output_templates)] ++
        Enum.flat_map(output_templates, fn t ->
          dtype_int = Map.fetch!(@mlx_type_to_int, EMLX.Native.to_mlx_type(t.type))
          shape = Tuple.to_list(t.shape)
          [dtype_int, length(shape) | shape]
        end)

    runtime_call = %{
      index: callback_index,
      callback: callback,
      args_template: Composite.traverse(tensor_expr, &Nx.to_template/1),
      arg_param_positions: arg_param_positions,
      opts: opts
    }

    result_ref =
      case out_template do
        %Nx.Tensor{} -> make_ref()
        _ -> Enum.map(output_templates, fn _ -> make_ref() end)
      end

    %{
      state
      | instructions: [{result_ref, :runtime_call, operand_refs, iattrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, result_ref),
        runtime_calls: [runtime_call | state.runtime_calls]
    }
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :while, args: [initial, arg, condition, body]}},
         state
       ) do
    case detect_static_while_trip_count(initial, arg, condition, body) do
      {:ok, count} -> expand_while_unroll(id, initial, arg, body, count, state)
      :error -> raise ArgumentError, "does not yet lower op :while"
    end
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: op}}, _state) do
    raise ArgumentError, "does not yet lower op #{inspect(op)}"
  end

  @doc false
  @spec native_lowerable_block?(struct(), [Nx.Tensor.t()]) :: boolean()
  def native_lowerable_block?(%Nx.Block.LogicalNot{}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.Take{}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.TakeAlongAxis{}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.FFT2{}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.IFFT2{}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.LinAlg.Cholesky{}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.LinAlg.Solve{}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.LinAlg.QR{mode: :reduced}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.LinAlg.Eigh{}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.LinAlg.SVD{full_matrices?: true}, _in_args), do: true
  def native_lowerable_block?(%Nx.Block.LinAlg.LU{}, _in_args), do: true
  def native_lowerable_block?(_struct, _in_args), do: false

  # ── dot helpers ────────────────────────────────────────────────────────────

  defp quantized_param_config(
         %T{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}},
         quant_signature
       ),
       do: Map.get(quant_signature, pos)

  defp quantized_param_config(_node, _quant_signature), do: nil

  defp expand_plain_dot(id, out_type, left, c_left, right, c_right, b_left, b_right, state) do
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

  defp expand_quantized_dot(
         id,
         out_type,
         left,
         c_left,
         b_left,
         right,
         c_right,
         %EMLX.Quantization.Config{} = cfg,
         state
       ) do
    unless b_left == [] do
      raise ArgumentError,
            "does not yet lower op :dot with a quantized right operand and batch axes " <>
              "(mx::quantized_matmul does not support batching)"
    end

    unless c_left == [tuple_size(left.shape) - 1] do
      raise ArgumentError,
            "does not yet lower op :dot with a quantized right operand contracted " <>
              "on a non-last left axis"
    end

    unless match?([_], c_right) do
      raise ArgumentError,
            "does not yet lower op :dot with a quantized right operand contracted " <>
              "on more than one axis"
    end

    last_dim = tuple_size(right.shape) - 1

    transpose =
      case cfg.transpose do
        nil -> c_right == [last_dim]
        explicit -> explicit
      end

    left_ref = Map.fetch!(state.node_to_ref, left.data.id)
    right_ref = Map.fetch!(state.node_to_ref, right.data.id)

    {scales_ref, state} = emit_capture(cfg.scales, state)

    {operands, has_bias, state} =
      if cfg.biases do
        {biases_ref, state} = emit_capture(cfg.biases, state)
        {[left_ref, right_ref, scales_ref, biases_ref], 1, state}
      else
        {[left_ref, right_ref, scales_ref], 0, state}
      end

    mode_int = Map.fetch!(@quant_mode_to_int, cfg.mode)
    transpose_int = if transpose, do: 1, else: 0
    iattrs = [cfg.group_size, cfg.bits, transpose_int, mode_int, has_bias]

    qmm_ref = make_ref()

    state = %{
      state
      | instructions: [{qmm_ref, :quantized_matmul, operands, iattrs} | state.instructions]
    }

    # mx::quantized_matmul returns the activation's dtype (matching the eager
    {result_ref, state} = emit_cast_if_needed(qmm_ref, left.type, out_type, state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  defp emit_capture(%Nx.Tensor{} = tensor, state) do
    ref = make_ref()
    {ref, %{state | captures: [{ref, tensor} | state.captures]}}
  end

  # ── indexing helpers ──────────────────────────────────────────────────────

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

  defp expand_block_via_default(id, in_args, default_expr, state) do
    inner_ordered = EMLX.Defn.Tree.post_order(default_expr, &scope_dependencies/1)

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

    result_ref =
      if is_tuple(default_expr) do
        flat_refs(default_expr, inner_state)
      else
        Map.fetch!(inner_state.node_to_ref, default_expr.data.id)
      end

    %{inner_state | node_to_ref: Map.put(inner_state.node_to_ref, id, result_ref)}
  end

  # ── while (static unroll for counted range loops) ──────────────────────────
  defp detect_static_while_trip_count(initial, arg, condition, body) when is_tuple(arg) do
    index_param = elem(arg, 0)

    with {:ok, start} <- constant_value(elem(initial, 0)),
         {:ok, bound, le?} <- while_condition_bound(condition, index_param),
         {:ok, step} <- while_body_step(elem(body, 0), index_param),
         {:ok, count} <- static_trip_count(start, bound, step, le?) do
      {:ok, count}
    else
      _ -> :error
    end
  end

  defp detect_static_while_trip_count(_initial, _arg, _condition, _body), do: :error

  defp static_trip_count(start, bound, step, true) when step > 0 and bound >= start,
    do: {:ok, div(bound - start, step) + 1}

  defp static_trip_count(start, bound, step, false) when step < 0 and bound <= start,
    do: {:ok, div(start - bound, -step) + 1}

  defp static_trip_count(_start, _bound, _step, _le?), do: :error

  defp while_condition_bound(
         %T{data: %Nx.Defn.Expr{op: op, args: [%T{data: %Nx.Defn.Expr{id: pid}}, bound_node]}},
         %T{data: %Nx.Defn.Expr{id: pid}}
       )
       when op in [:less_equal, :greater_equal] do
    case constant_value(bound_node) do
      {:ok, bound} -> {:ok, bound, op == :less_equal}
      :error -> :error
    end
  end

  defp while_condition_bound(_condition, _index_param), do: :error

  defp while_body_step(
         %T{data: %Nx.Defn.Expr{op: :add, args: [a, b]}},
         %T{data: %Nx.Defn.Expr{id: pid}}
       ) do
    case {a, b} do
      {%T{data: %Nx.Defn.Expr{id: ^pid}}, step_node} -> constant_value(step_node)
      {step_node, %T{data: %Nx.Defn.Expr{id: ^pid}}} -> constant_value(step_node)
      _ -> :error
    end
  end

  defp while_body_step(_next_index, _index_param), do: :error

  defp constant_value(%T{data: %Nx.Defn.Expr{op: :constant, args: [n]}}) when is_integer(n),
    do: {:ok, n}

  defp constant_value(_node), do: :error

  defp expand_while_unroll(id, initial, arg, body, count, state) do
    initial_list = Tuple.to_list(initial)
    arg_list = Tuple.to_list(arg)
    body_list = Tuple.to_list(body)

    init_refs = Enum.map(initial_list, &Map.fetch!(state.node_to_ref, &1.data.id))

    {final_refs, state} =
      Enum.reduce(1..count//1, {init_refs, state}, fn _iteration, {carry_refs, acc_state} ->
        param_ref_by_pos =
          arg_list
          |> Enum.zip(carry_refs)
          |> Map.new(fn {param, ref} -> {hd(param.data.args), ref} end)

        lower_tuple_body(body_list, param_ref_by_pos, acc_state)
      end)

    %{state | node_to_ref: Map.put(state.node_to_ref, id, final_refs)}
  end

  defp lower_tuple_body(body_list, param_ref_by_pos, state) do
    body_tuple = List.to_tuple(body_list)
    state = merge_scope_ids(state, body_tuple)
    inner_ordered = EMLX.Defn.Tree.post_order(body_tuple, &scope_dependencies/1)

    param_id_to_ref =
      inner_ordered
      |> Enum.filter(&(&1.data.op == :parameter))
      |> Map.new(fn p -> {p.data.id, Map.fetch!(param_ref_by_pos, hd(p.data.args))} end)

    param_id_set = MapSet.new(Map.keys(param_id_to_ref))
    local_base = Map.merge(state.node_to_ref, param_id_to_ref)

    inner_state =
      Enum.reduce(inner_ordered, %{state | node_to_ref: local_base}, fn node, st ->
        if MapSet.member?(param_id_set, node.data.id), do: st, else: expand_node(node, st)
      end)

    result_refs = Enum.map(body_list, &Map.fetch!(inner_state.node_to_ref, &1.data.id))

    {result_refs,
     %{
       state
       | instructions: inner_state.instructions,
         captures: inner_state.captures,
         constants: inner_state.constants,
         inputs: inner_state.inputs,
         hooks: inner_state.hooks
     }}
  end

  # ── custom-fun reduce (static unroll) ──────────────────────────────────────
  defp expand_reduce_unroll(id, out_type, out_shape, tensor, acc, opts, fun, state) do
    in_rank = tuple_size(tensor.shape)

    reduce_axes =
      case opts[:axes] do
        nil -> Enum.to_list(0..(in_rank - 1)//1)
        axes -> Enum.sort(normalize_axes(axes, in_rank))
      end

    if in_rank == 0 or reduce_axes == [] do
      raise ArgumentError, "does not yet lower op :reduce with no reduction axes"
    end

    [params, body, _mfa] = fun.data.args

    unless length(params) == 2 do
      raise ArgumentError, "does not yet lower op :reduce with a non-binary reducer"
    end

    reduce_set = MapSet.new(reduce_axes)
    kept_axes = Enum.reject(0..(in_rank - 1)//1, &MapSet.member?(reduce_set, &1))
    in_dims = Tuple.to_list(tensor.shape)
    kept_shape = Enum.map(kept_axes, &Enum.at(in_dims, &1))
    reduce_extent = reduce_axes |> Enum.map(&Enum.at(in_dims, &1)) |> Enum.product()

    # Cast input + initial acc to the reducer/output type.
    tensor_ref0 = Map.fetch!(state.node_to_ref, tensor.data.id)
    {tensor_ref, state} = emit_cast_if_needed(tensor_ref0, tensor.type, out_type, state)
    acc_ref0 = Map.fetch!(state.node_to_ref, acc.data.id)
    {acc_scalar_ref, state} = emit_cast_if_needed(acc_ref0, acc.type, out_type, state)

    # Move reduce axes last, then collapse them into a single trailing axis.
    perm = kept_axes ++ reduce_axes

    {perm_ref, state} =
      if perm == Enum.to_list(0..(in_rank - 1)//1) do
        {tensor_ref, state}
      else
        emit_transpose_instr(tensor_ref, perm, state)
      end

    combined_shape = kept_shape ++ [reduce_extent]
    {combined_ref, state} = emit_reshape_instr(perm_ref, combined_shape, state)

    # Seed acc broadcast to the kept shape, then fold over the extent.
    {acc_ref, state} = emit_broadcast_to(acc_scalar_ref, kept_shape, state)

    {final_ref, state} =
      Enum.reduce(0..(reduce_extent - 1)//1, {acc_ref, state}, fn i, {acc_i, st} ->
        {elem_ref, st} = emit_reduce_slice(combined_ref, combined_shape, kept_shape, i, st)
        lower_fun_body(body, %{0 => elem_ref, 1 => acc_i}, st)
      end)

    # Reshape to the declared output shape (restores keep_axes 1-dims).
    {out_ref, state} = emit_reshape_instr(final_ref, Tuple.to_list(out_shape), state)
    %{state | node_to_ref: Map.put(state.node_to_ref, id, out_ref)}
  end

  # Inline fun/while bodies sit outside top-level scope_ids; extend before lowering hooks.
  defp merge_scope_ids(state, body) do
    extra = body |> Nx.Defn.Tree.scope_ids() |> Map.keys() |> MapSet.new()
    %{state | top_scope_ids: MapSet.union(state.top_scope_ids, extra)}
  end

  defp lower_fun_body(body, param_ref_by_pos, state) do
    state = merge_scope_ids(state, body)
    inner_ordered = EMLX.Defn.Tree.post_order(body, &scope_dependencies/1)

    param_id_to_ref =
      inner_ordered
      |> Enum.filter(&(&1.data.op == :parameter))
      |> Map.new(fn p -> {p.data.id, Map.fetch!(param_ref_by_pos, hd(p.data.args))} end)

    param_id_set = MapSet.new(Map.keys(param_id_to_ref))
    local_base = Map.merge(state.node_to_ref, param_id_to_ref)

    inner_state =
      Enum.reduce(inner_ordered, %{state | node_to_ref: local_base}, fn node, st ->
        if MapSet.member?(param_id_set, node.data.id), do: st, else: expand_node(node, st)
      end)

    result_ref = Map.fetch!(inner_state.node_to_ref, body.data.id)

    # Carry forward everything the body produced, but discard its node_to_ref.
    {result_ref,
     %{
       state
       | instructions: inner_state.instructions,
         captures: inner_state.captures,
         constants: inner_state.constants,
         inputs: inner_state.inputs,
         hooks: inner_state.hooks
     }}
  end

  # Emit a :reshape instruction; returns {new_ref, state}.
  defp emit_reshape_instr(ref, shape_list, state) do
    new_ref = make_ref()

    {new_ref,
     %{state | instructions: [{new_ref, :reshape, [ref], shape_list} | state.instructions]}}
  end

  # Broadcast a scalar ref to `shape_list` (no-op when the target is scalar).
  defp emit_broadcast_to(ref, [], state), do: {ref, state}

  defp emit_broadcast_to(ref, shape_list, state) do
    new_ref = make_ref()
    iattrs = [length(shape_list) | shape_list] ++ [0]

    {new_ref,
     %{state | instructions: [{new_ref, :broadcast, [ref], iattrs} | state.instructions]}}
  end

  # Slice element `i` along the collapsed trailing axis then squeeze it away,
  defp emit_reduce_slice(ref, combined_shape, kept_shape, i, state) do
    n_dims = length(combined_shape)
    last_axis = n_dims - 1
    lengths = kept_shape ++ [1]
    strides = List.duplicate(1, n_dims)
    starts = List.duplicate(0, length(kept_shape)) ++ [i]
    iattrs = [n_dims, 0] ++ combined_shape ++ lengths ++ strides ++ starts

    slice_ref = make_ref()
    state = %{state | instructions: [{slice_ref, :slice, [ref], iattrs} | state.instructions]}
    squeeze_ref = make_ref()

    {squeeze_ref,
     %{
       state
       | instructions: [{squeeze_ref, :squeeze, [slice_ref], [last_axis]} | state.instructions]
     }}
  end

  defp emit_pad_with(ref, pad_value_ref, low_pads, high_pads, state) do
    new_ref = make_ref()
    n_dims = length(low_pads)

    iattrs =
      [n_dims | Enum.flat_map(Enum.zip(low_pads, high_pads), fn {lo, hi} -> [lo, hi, 0] end)]

    {new_ref,
     %{
       state
       | instructions: [{new_ref, :pad, [ref, pad_value_ref], iattrs} | state.instructions]
     }}
  end

  defp emit_static_slice(ref, input_shape, starts, lengths, strides, state) do
    new_ref = make_ref()
    n_dims = length(input_shape)
    iattrs = [n_dims, 0] ++ input_shape ++ lengths ++ strides ++ starts

    {new_ref, %{state | instructions: [{new_ref, :slice, [ref], iattrs} | state.instructions]}}
  end

  defp expand_pad_general(tensor_ref, pad_value_ref, in_dims, config, state) do
    interior_list = Enum.map(config, fn {_lo, _hi, interior} -> interior end)

    {interior_ref, state} =
      if Enum.all?(interior_list, &(&1 == 0)) do
        {tensor_ref, state}
      else
        emit_interior_padding(tensor_ref, pad_value_ref, in_dims, interior_list, state)
      end

    interior_shape =
      Enum.zip(in_dims, interior_list)
      |> Enum.map(fn {d, interior} -> d + max(d - 1, 0) * interior end)

    {cropped_ref, _cropped_shape, state} =
      if Enum.any?(config, fn {lo, hi, _interior} -> lo < 0 or hi < 0 end) do
        emit_negative_crop(interior_ref, interior_shape, config, state)
      else
        {interior_ref, interior_shape, state}
      end

    low_pads = Enum.map(config, fn {lo, _hi, _interior} -> max(lo, 0) end)
    high_pads = Enum.map(config, fn {_lo, hi, _interior} -> max(hi, 0) end)

    if Enum.all?(low_pads ++ high_pads, &(&1 == 0)) do
      {cropped_ref, state}
    else
      emit_pad_with(cropped_ref, pad_value_ref, low_pads, high_pads, state)
    end
  end

  defp emit_interior_padding(ref, pad_value_ref, in_dims, interior_list, state) do
    rank = length(in_dims)
    shape0 = in_dims ++ [1]
    {ref, state} = emit_reshape_instr(ref, shape0, state)

    {final_ref, _final_shape, state} =
      interior_list
      |> Enum.with_index()
      |> Enum.reduce({ref, shape0, state}, fn
        {0, _axis_index}, {acc_ref, shape, st} ->
          {acc_ref, shape, st}

        {interior, axis_index}, {acc_ref, shape, st} ->
          next_axis = axis_index + 1
          axis_size = Enum.at(shape, axis_index)
          next_axis_size = Enum.at(shape, next_axis)

          pad_lows = List.duplicate(0, rank + 1)
          pad_highs = List.replace_at(pad_lows, next_axis, next_axis_size * interior)
          {padded_ref, st} = emit_pad_with(acc_ref, pad_value_ref, pad_lows, pad_highs, st)

          new_axis_size = axis_size + axis_size * interior

          reshaped_shape =
            shape
            |> List.replace_at(axis_index, new_axis_size)
            |> List.replace_at(next_axis, next_axis_size)

          {reshaped_ref, st} = emit_reshape_instr(padded_ref, reshaped_shape, st)

          sliced_shape = List.replace_at(reshaped_shape, axis_index, new_axis_size - interior)
          starts = List.duplicate(0, rank + 1)
          strides = List.duplicate(1, rank + 1)

          {sliced_ref, st} =
            emit_static_slice(reshaped_ref, reshaped_shape, starts, sliced_shape, strides, st)

          {sliced_ref, sliced_shape, st}
      end)

    squeeze_ref = make_ref()

    {squeeze_ref,
     %{state | instructions: [{squeeze_ref, :squeeze, [final_ref], [rank]} | state.instructions]}}
  end

  defp emit_negative_crop(ref, shape, config, state) do
    starts = Enum.map(config, fn {lo, _hi, _interior} -> max(-lo, 0) end)

    lengths =
      [shape, config, starts]
      |> Enum.zip_with(fn [d, {_lo, hi, _interior}, start] ->
        stop = if hi < 0, do: d + hi, else: d
        stop - start
      end)

    strides = List.duplicate(1, length(shape))
    {new_ref, state} = emit_static_slice(ref, shape, starts, lengths, strides, state)
    {new_ref, lengths, state}
  end

  defp window_offsets(k, dims) do
    {digits, _} =
      dims
      |> Enum.reverse()
      |> Enum.reduce({[], k}, fn d, {acc, n} -> {[rem(n, d) | acc], div(n, d)} end)

    digits
  end

  @doc false
  @spec quantizable_param_positions(Nx.Container.t()) :: MapSet.t(non_neg_integer())
  def quantizable_param_positions(output) do
    output
    |> EMLX.Defn.Tree.post_order(&scope_dependencies/1)
    |> Enum.reduce(MapSet.new(), fn
      %T{data: %Nx.Defn.Expr{op: :dot, args: [left, _c_left, _b_left, right, _c_right, _b_right]}},
      acc ->
        acc
        |> maybe_put_param_position(left)
        |> maybe_put_param_position(right)

      _, acc ->
        acc
    end)
  end

  defp maybe_put_param_position(acc, %T{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}}),
    do: MapSet.put(acc, pos)

  defp maybe_put_param_position(acc, _node), do: acc

  @doc false
  @spec f64_bits(number()) :: integer()
  def f64_bits(v) when is_number(v) do
    <<bits::signed-64>> = <<v * 1.0::float-64>>
    bits
  end

  @doc false
  @spec bits_to_f64(integer()) :: float()
  def bits_to_f64(bits) when is_integer(bits) do
    <<v::float-64>> = <<bits::signed-64>>
    v
  end

  # ── cond helper ───────────────────────────────────────────────────────────

  defp flat_refs(composite, state) do
    Composite.flatten_list([composite])
    |> Enum.map(&Map.fetch!(state.node_to_ref, &1.data.id))
  end

  # ── binary lowering helpers ────────────────────────────────────────────────

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

  defp swap_last_two_axes(rank) do
    {front, [x, y]} = Enum.split(Enum.to_list(0..(rank - 1)), rank - 2)
    front ++ [y, x]
  end

  # Emit a :solve_triangular instruction; returns {result_ref, updated_state}.
  defp emit_solve_triangular_instr(a_ref, b_ref, upper, state) do
    ref = make_ref()
    upper_int = if upper, do: 1, else: 0

    {ref,
     %{
       state
       | instructions: [
           {ref, :solve_triangular, [a_ref, b_ref], [upper_int]} | state.instructions
         ]
     }}
  end

  # Move the second element (channels) to the last position.
  defp move_channels_last([head | [second | rest]]) do
    [head | rest] ++ [second]
  end

  # ── wire serialisation ────────────────────────────────────────────────────

  @doc false
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

    maybe_debug_check do
      expected_size = map_size(input_map) + map_size(capture_map) + map_size(constant_map)

      if map_size(ref_to_packed) != expected_size do
        raise ArgumentError,
              "EMLX.Native.Expr.to_wire: ref id collision across inputs/captures/constants -- " <>
                "#{map_size(input_map)} input(s), #{map_size(capture_map)} capture(s), " <>
                "#{map_size(constant_map)} constant(s) should merge to #{expected_size} distinct " <>
                "refs, but only #{map_size(ref_to_packed)} survived Map.merge/2. This means two " <>
                "refs of different categories share the same id, silently dropping one from the " <>
                "wire map -- inputs: #{inspect(Map.keys(input_map))}, " <>
                "captures: #{inspect(Map.keys(capture_map))}, " <>
                "constants: #{inspect(Map.keys(constant_map))}"
      end
    end

    {op_names, operands, iattrs, ref_to_packed, _flat} =
      prog.instructions
      |> Enum.reduce({[], [], [], ref_to_packed, 0}, fn {id, op, operand_refs, attrs},
                                                        {ops, ors, ias, rmap, flat} ->
        wire_operands = Enum.map(operand_refs, &Map.fetch!(rmap, &1))

        maybe_debug_check do
          for packed <- wire_operands,
              packed >>> @kind_shift == @kind_instr,
              (packed &&& (1 <<< @kind_shift) - 1) >= flat do
            raise ArgumentError,
                  "EMLX.Native.Expr.to_wire: instruction #{inspect(op)} (id=#{inspect(id)}) " <>
                    "references result index #{packed &&& (1 <<< @kind_shift) - 1} of the " <>
                    "flat results accumulator, but only #{flat} result(s) have been produced " <>
                    "so far -- this is a forward/self reference bug in program lowering, not " <>
                    "a valid program. Full instruction list: #{inspect(prog.instructions)}"
          end
        end

        maybe_debug_check do
          for one <- List.wrap(id), Map.has_key?(rmap, one) do
            raise ArgumentError,
                  "EMLX.Native.Expr.to_wire: instruction #{inspect(op)} produces result ref " <>
                    "#{inspect(one)}, but that ref is already bound (to " <>
                    "#{inspect(Map.fetch!(rmap, one))}) -- the same node id was lowered twice, " <>
                    "silently overwriting its earlier binding for every prior instruction that " <>
                    "already referenced it. Full instruction list: #{inspect(prog.instructions)}"
          end
        end

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

    hook_refs = Enum.flat_map(prog.hooks, & &1.refs)
    wire_outputs = Enum.map(prog.outputs ++ hook_refs, &Map.fetch!(ref_to_packed, &1))

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
