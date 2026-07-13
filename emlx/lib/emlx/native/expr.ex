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

  @enforce_keys [:inputs, :captures, :constants, :instructions, :outputs]
  defstruct [
    :inputs,
    :captures,
    :constants,
    :instructions,
    :outputs,
    keepalive_refs: [],
    runtime_calls: []
  ]

  @type node_ref :: reference()
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
          # Hooks (`:token`/`:attach_token`) lower to inline `:runtime_call`s
          # (see the `:token` clause of `expand_node/2`) chained through a
          # single running "keepalive" ref per program: each hook's runtime
          # call both fires the real Elixir side effect *and* threads a
          # scalar forward, so (a) MLX's lazy evaluator is forced to actually
          # run a hook whose value would otherwise be unreferenced by the
          # real output (matching a hook's own token/attach_token becoming
          # reachable, without needing that specific hook's value to be),
          # and (b) hooks that aren't already ordered by a genuine data
          # dependency still fire in declaration order, without relying on
          # unspecified scheduling order between independent primitives.
          # `keepalive_refs` holds the *final* tip of that chain (0 or 1
          # refs) -- appended to `outputs` on the wire (see `to_native/1`,
          # which also records the real/keepalive boundary in
          # `EMLX.Native.Program.num_real_outputs`) and dropped natively by
          # `eval_program` before it returns (the real firing already
          # happened inline, during evaluation) -- Elixir never sees it.
          keepalive_refs: [node_ref()],
          runtime_calls: [runtime_call()]
        }

  # ── lowering ──────────────────────────────────────────────────────────────

  @doc false
  def lower(output, num_inputs \\ nil, quant_signature \\ %{}) do
    ordered = EMLX.Defn.Tree.post_order(output, &scope_dependencies/1)
    top_scope_ids = top_scope_ids(output, ordered)

    # inputs is a map of pos → ref during lowering; densified to a list at the end.
    state = %{
      inputs: %{},
      captures: [],
      constants: [],
      instructions: [],
      node_to_ref: %{},
      runtime_calls: [],
      top_scope_ids: top_scope_ids,
      quant_signature: quant_signature,
      while_nesting_depth: 0,
      stable_positions: %{},
      hook_chain_ref: nil,
      hooked_value_refs: %{},
      refcounts: compute_refcounts(ordered)
    }

    state = Enum.reduce(ordered, state, &expand_node/2)

    max_referenced_pos = state.inputs |> Map.keys() |> Enum.max(fn -> -1 end)
    arity = max(num_inputs || 0, max_referenced_pos + 1)

    inputs_list =
      for pos <- 0..(arity - 1)//1 do
        Map.get_lazy(state.inputs, pos, &make_ref/0)
      end

    flat_outputs = Composite.flatten_list([output])

    # A `__EMLX_QUANT__`-tagged leaf (see `EMLX.Quantization.quantize/2`)
    # is one *logical* output but has no single physical ref of its own
    # (see the `:metadata` `expand_node/2` clause above) — emit its 2-3
    # underlying decomposition refs directly instead of looking the leaf's
    # own id up in `node_to_ref`. Every other leaf still contributes
    # exactly one ref, as before.
    output_refs =
      Enum.flat_map(flat_outputs, fn
        %T{
          data: %Nx.Defn.Expr{
            op: :metadata,
            args: [_inner, %{__EMLX_QUANT__: %{weight: weight, scales: scales, biases: biases}}]
          }
        } ->
          refs = [
            Map.fetch!(state.node_to_ref, weight.data.id),
            Map.fetch!(state.node_to_ref, scales.data.id)
          ]

          if biases, do: refs ++ [Map.fetch!(state.node_to_ref, biases.data.id)], else: refs

        leaf ->
          [Map.fetch!(state.node_to_ref, leaf.data.id)]
      end)

    %__MODULE__{
      inputs: inputs_list,
      captures: Enum.reverse(state.captures),
      constants: Enum.reverse(state.constants),
      instructions: Enum.reverse(state.instructions),
      outputs: output_refs,
      keepalive_refs: List.wrap(state.hook_chain_ref),
      runtime_calls: Enum.reverse(state.runtime_calls)
    }
  end

  # `Nx.Defn.Tree.scope_ids/1` only walks a node's *generic* args — it has
  # no notion of `__EMLX_QUANT__`'s custom `scope_dependencies/1` redirection
  # (see `EMLX.Quantization.quantize/2`), so a quantize-traced node's
  # decomposition `:runtime_call` (reachable only through that metadata
  # payload) is invisible to it and never marked top-scope, even when the
  # metadata node itself is. Backfill it here so the `:runtime_call`
  # `expand_node/2` clause's "not inside a cond" check doesn't misfire on a
  # decomposition call that is, in fact, top-scope.
  defp top_scope_ids(output, ordered) do
    base = output |> Nx.Defn.Tree.scope_ids() |> Map.keys() |> MapSet.new()

    Enum.reduce(ordered, base, fn
      %T{
        data: %Nx.Defn.Expr{
          id: id,
          op: :metadata,
          args: [_inner, %{__EMLX_QUANT__: %{weight: %T{data: %Nx.Defn.Expr{args: [root, _]}}}}]
        }
      },
      acc ->
        if MapSet.member?(acc, id), do: MapSet.put(acc, root.data.id), else: acc

      _, acc ->
        acc
    end)
  end

  # Same-scope reference count per node id — how many *other* nodes in
  # `ordered` (or a `:while` subprogram's own `inner_ordered`) reference a
  # given node as one of their `args`, via the same generic `Tree.apply_args`
  # walk `EMLX.Defn.Tree.post_order/2` itself uses. Deliberately does **not**
  # count a node being a top-level/subprogram *output leaf* (that's resolved
  # via `node_to_ref` after the whole reduce, not via another node's args —
  # see `lower/3`'s/`lower_while_subprogram/3`'s own `output_refs`/leaf
  # lookups) — only used by the `:kv_cache_sdpa_update` peephole fusion
  # (below) to verify it's safe to delete a node's already-emitted
  # instruction (i.e. confirm the fused SDPA node is genuinely its *only*
  # same-scope consumer) without needing to patch up some other, unrelated
  # consumer that a naive fusion would otherwise leave dangling.
  defp compute_refcounts(ordered) do
    Enum.reduce(ordered, %{}, fn
      # Mirrors `EMLX.Defn.Tree.visit_scope_deps/3`'s own two special cases
      # exactly (`:fun` has no same-scope deps; `scope_dependencies/1`
      # redirects `:metadata`/`__EMLX__` nodes to their `operands` list
      # instead of their literal `args`) -- generic `Tree.apply_args/4` alone
      # doesn't know about either, so it would (wrongly) report 0 references
      # to a `:metadata` node's `__EMLX__` operands (e.g. an SDPA node's own
      # Q/K/V), even though `post_order/2` *does* visit them as this node's
      # dependencies. Must stay in lockstep with that module's traversal or
      # the `kv_cache_sdpa_update` fusion's refcount==1 safety checks below
      # silently never pass.
      %T{data: %Nx.Defn.Expr{op: :fun}}, acc ->
        acc

      node, acc ->
        deps =
          case scope_dependencies(node) do
            {:ok, deps} -> deps
            :default -> apply_args_deps(node)
          end

        Enum.reduce(deps, acc, fn dep, a -> Map.update(a, dep.data.id, 1, &(&1 + 1)) end)
    end)
  end

  defp apply_args_deps(node) do
    {_, deps} =
      Nx.Defn.Tree.apply_args(node, :scope, [], fn dep, acc -> {dep, [dep | acc]} end)

    deps
  end

  # ── kv_cache_sdpa_update peephole fusion helpers ────────────────────────────
  #
  # Used by the `:fast_sdpa_causal_key_masked` `expand_node/2` clause below —
  # kept out of the `expand_node/2` clause run itself so as not to break up
  # that group (see this module's other private helpers, similarly kept
  # outside the run).

  @head_transpose_axes [0, 2, 1, 3]

  defp match_kv_cache_sdpa_fusion(q_t, k_t, v_t, state) do
    with {:ok, q} <- match_head_transpose(q_t),
         {:ok, k_put_slice_chain, k_cache, k_offset, new_k} <- match_cache_put_slice(k_t),
         {:ok, v_put_slice_chain, v_cache, v_offset, new_v} <- match_cache_put_slice(v_t),
         true <- not match?(%T{data: %Nx.Defn.Expr{op: :constant}}, k_offset),
         true <- k_offset.data.id == v_offset.data.id,
         true <- Map.get(state.refcounts, q_t.data.id, 0) == 1,
         true <- Map.get(state.refcounts, k_t.data.id, 0) == 1,
         true <- Map.get(state.refcounts, v_t.data.id, 0) == 1,
         true <- Map.get(state.refcounts, hd(k_put_slice_chain).data.id, 0) == 1,
         true <- Map.get(state.refcounts, hd(v_put_slice_chain).data.id, 0) == 1 do
      {:ok, q, new_k, new_v, k_cache, v_cache, k_offset, k_put_slice_chain, v_put_slice_chain}
    else
      _ -> :no_match
    end
  end

  defp match_head_transpose(%T{data: %Nx.Defn.Expr{op: :transpose, args: [inner, axes]}}) do
    if Enum.to_list(axes) == @head_transpose_axes, do: {:ok, inner}, else: :no_match
  end

  defp match_head_transpose(_), do: :no_match

  # `k_t`/`v_t` (already known to be `transpose(_, [0,2,1,3])` by the time
  # this is called) must wrap exactly `put_slice(cache, [0, offset, 0, 0],
  # new_val)` — a dynamic (tensor-valued) `offset` at the T axis (axis 1,
  # Bumblebee-native layout) and static 0 everywhere else. A static-integer
  # `offset` (e.g. a one-shot prefill call, not inside a decode `:while`)
  # intentionally does not match — fusion only pays off across many
  # iterations, and requiring a dynamic offset here is a cheap, sufficient
  # proxy for "this is a decode loop write" without inspecting scope depth.
  defp match_cache_put_slice(%T{
         data: %Nx.Defn.Expr{op: :transpose, args: [put_slice_node, axes]}
       }) do
    if Enum.to_list(axes) == @head_transpose_axes do
      case unwrap_generic_metadata(put_slice_node) do
        {:ok, aliases,
         %T{data: %Nx.Defn.Expr{op: :put_slice, args: [cache, [ax0, offset, ax2, ax3], new_val]}} =
             raw} ->
          if static_zero?(ax0) and not static_zero?(offset) and static_zero?(ax2) and
               static_zero?(ax3) do
            {:ok, [raw | aliases], cache, offset, new_val}
          else
            :no_match
          end

        _ ->
          :no_match
      end
    else
      :no_match
    end
  end

  defp match_cache_put_slice(_), do: :no_match

  # Bumblebee/Axon wraps a named layer's output (e.g.
  # `update_attention_cache`'s `put_slice`, whose result threads into both
  # this SDPA read *and* the decode loop's carried cache state) in a
  # generic `:metadata` node (`%{axon_layer: ...}`, no `__EMLX__`/
  # `__EMLX_QUANT__` key) purely for introspection -- `expand_node/2`'s
  # catch-all `:metadata` clause below aliases its id straight to the
  # wrapped node's ref without emitting an instruction of its own. See
  # through any number of these to find the real op, collecting every
  # wrapper id along the way so `fuse_kv_cache_sdpa_instr/15` can redirect
  # *all* of them (not just the raw `:put_slice`'s id) to the fused op's
  # output ref -- otherwise a consumer reading through a wrapper id would
  # keep resolving to the now-pruned `:put_slice` instruction's stale ref.
  defp unwrap_generic_metadata(node, acc \\ [])

  defp unwrap_generic_metadata(
         %T{data: %Nx.Defn.Expr{op: :metadata, args: [inner, meta]}} = node,
         acc
       )
       when not is_map_key(meta, :__EMLX__) and not is_map_key(meta, :__EMLX_QUANT__) do
    unwrap_generic_metadata(inner, [node | acc])
  end

  defp unwrap_generic_metadata(node, acc), do: {:ok, acc, node}

  # Traced `Nx.put_slice(cache, [0, offset, 0, 0], new)` calls normalize
  # *every* `start_indices` entry to an `Nx.Defn.Expr` (constant tensors for
  # the literal `0`s, not left as plain host integers) -- so a static "0" here
  # is `%T{data: %Nx.Defn.Expr{op: :constant, args: [0]}}`, never a bare `0`.
  defp static_zero?(0), do: true
  defp static_zero?(%T{data: %Nx.Defn.Expr{op: :constant, args: [0]}}), do: true
  defp static_zero?(_), do: false

  # Test-observable fusion counter — bumped once per successful fusion
  # (typically once per attention layer per model *compile*, not per token:
  # a `:while` decode-loop body is lowered once and replayed natively). Cheap
  # enough to leave unconditional (rare-write `:persistent_term`, and this
  # only ever runs during `EMLX.Native.Expr.lower/3`, not on the hot decode
  # path itself) — lets `EMLX.Native.ExprTest` assert the peephole actually
  # fired, rather than silently falling through to the unfused path the
  # whole time and only ever exercising `match_kv_cache_sdpa_fusion/4`'s
  # `:no_match` branch.
  @doc false
  def kv_cache_fusion_count, do: :persistent_term.get(:emlx_kv_cache_fusion_count, 0)

  @doc false
  def reset_kv_cache_fusion_count, do: :persistent_term.put(:emlx_kv_cache_fusion_count, 0)

  defp bump_kv_cache_fusion_count! do
    :persistent_term.put(:emlx_kv_cache_fusion_count, kv_cache_fusion_count() + 1)
  end

  defp fuse_kv_cache_sdpa_instr(
         id,
         q,
         new_k,
         new_v,
         k_cache,
         v_cache,
         offset,
         key_mask,
         attrs,
         q_t,
         k_t,
         v_t,
         k_put_slice_chain,
         v_put_slice_chain,
         state
       ) do
    bump_kv_cache_fusion_count!()

    operands =
      [q, new_k, new_v, k_cache, v_cache, offset, key_mask]
      |> Enum.map(&Map.fetch!(state.node_to_ref, &1.data.id))

    # Refs of the 5 now-dead instructions (Q's transpose; K/V's transpose +
    # put_slice each) to strip from `state.instructions` -- otherwise the
    # interpreter would still dispatch all 6 original ops *plus* this new
    # fused one, a net dispatch-count *increase*, not the intended decrease.
    # `hd/1` of each chain is enough here: every alias in a chain still
    # resolves (pre-fusion) to the same single put_slice ref.
    dead_refs =
      MapSet.new(
        Enum.map(
          [q_t, k_t, v_t, hd(k_put_slice_chain), hd(v_put_slice_chain)],
          &Map.fetch!(state.node_to_ref, &1.data.id)
        )
      )

    pruned_instructions =
      Enum.reject(state.instructions, fn {ref_or_refs, _op, _operands, _attrs} ->
        case ref_or_refs do
          refs when is_list(refs) -> Enum.any?(refs, &MapSet.member?(dead_refs, &1))
          ref -> MapSet.member?(dead_refs, ref)
        end
      end)

    attn_ref = make_ref()
    k_upd_ref = make_ref()
    v_upd_ref = make_ref()

    # Every id in a put_slice chain (the raw `:put_slice` plus any generic
    # `:metadata` wrapper ids around it) must be redirected to the new
    # fused ref, not just the raw node's -- a wrapper id may be the one a
    # *different* consumer (e.g. the decode loop's carried cache state)
    # actually reads through.
    node_to_ref =
      state.node_to_ref
      |> Map.put(id, attn_ref)
      |> then(
        &Enum.reduce(k_put_slice_chain, &1, fn n, acc -> Map.put(acc, n.data.id, k_upd_ref) end)
      )
      |> then(
        &Enum.reduce(v_put_slice_chain, &1, fn n, acc -> Map.put(acc, n.data.id, v_upd_ref) end)
      )

    %{
      state
      | instructions: [
          {[attn_ref, k_upd_ref, v_upd_ref], :kv_cache_sdpa_update, operands, attrs}
          | pruned_instructions
        ],
        node_to_ref: node_to_ref
    }
  end

  defp emit_metadata_instr(id, opcode, operands, attrs, state) do
    operand_refs = Enum.map(operands, &Map.fetch!(state.node_to_ref, &1.data.id))

    refs =
      case {opcode, Enum.at(attrs, 5)} do
        {:plugin, count} when is_integer(count) and count > 1 ->
          for _ <- 1..count, do: make_ref()

        _ ->
          make_ref()
      end

    %{
      state
      | instructions: [{refs, opcode, operand_refs, attrs} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, refs)
    }
  end

  # ── node expansion ────────────────────────────────────────────────────────

  @doc false
  def scope_dependencies(%T{
        data: %Nx.Defn.Expr{op: :metadata, args: [_inner, %{__EMLX__: %{operands: operands}}]}
      }) do
    {:ok, operands}
  end

  # `EMLX.Quantization.quantize/2`'s traced representation (see its
  # `quantize_traced/2`) — redirects traversal to the decomposition leaves
  # (weight/scales/biases, each a real dense `Nx.runtime_call` result) so
  # `inner` (a plain-tensor fallback for non-EMLX compilers only) is never
  # visited/lowered by this compiler at all.
  def scope_dependencies(%T{
        data: %Nx.Defn.Expr{
          op: :metadata,
          args: [_inner, %{__EMLX_QUANT__: %{weight: weight, scales: scales, biases: biases}}]
        }
      }) do
    {:ok, if(biases, do: [weight, scales, biases], else: [weight, scales])}
  end

  def scope_dependencies(_node), do: :default

  defp expand_node(%T{data: %Nx.Defn.Expr{id: id, op: :parameter, args: [pos]}}, state) do
    ref = make_ref()

    %{
      state
      | inputs: Map.put(state.inputs, pos, ref),
        node_to_ref: Map.put(state.node_to_ref, id, ref),
        # Real top-level position, in this program's own :parameter numbering
        # (matches `tensors` in EMLX.handle_runtime_call/6) -- see
        # `expand_while_native/6`'s stable-carry propagation for how this
        # reaches a `:runtime_call` operand nested inside a `:while` body.
        stable_positions: Map.put(state.stable_positions, id, pos)
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
      attrs = [length(shape_list) | shape_list] ++ [0]

      %{
        state
        | instructions: [{broadcast_ref, :broadcast, [ref], attrs} | state.instructions],
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

  # ── kv_cache_sdpa_update peephole fusion ────────────────────────────────────
  # EMLXAxon's `build_sdpa_layer/5` always wraps its causal+key_mask SDPA
  # call as `transpose(sdpa(transpose(q), transpose(k), transpose(v), ...))`
  # (head-transpose in, head-transpose out — see that function), and its K/V
  # arguments trace back to `Bumblebee.Layers.Decoder.update_attention_cache/5`
  # (i.e. `Nx.put_slice(cache, [0, offset, 0, 0], new)`) whenever this SDPA
  # node sits directly downstream of a KV-cache write, as in any decode loop.
  # When that exact shape is confirmed (see `match_kv_cache_sdpa_fusion/4`,
  # including refcount==1 safety checks — this is purely an optimization, so
  # any mismatch silently falls through to the generic lowering below, never
  # an error), rewrite `put_slice(K)` + `put_slice(V)` + `transpose` ×3 (Q,
  # post-write K, post-write V) + this SDPA node — 6 native-IR instructions —
  # into one `multi_op_registry["kv_cache_sdpa_update"]` (`emlx_compiler.cpp`)
  # instruction.
  defp expand_node(
         %T{
           data: %Nx.Defn.Expr{
             id: id,
             op: :metadata,
             args: [
               _inner,
               %{
                 __EMLX__: %{
                   op: :fast_sdpa_causal_key_masked,
                   operands: [q_t, k_t, v_t, key_mask],
                   attrs: attrs
                 }
               }
             ]
           }
         },
         state
       ) do
    case match_kv_cache_sdpa_fusion(q_t, k_t, v_t, state) do
      {:ok, q, new_k, new_v, k_cache, v_cache, offset, k_put_slice, v_put_slice} ->
        fuse_kv_cache_sdpa_instr(
          id,
          q,
          new_k,
          new_v,
          k_cache,
          v_cache,
          offset,
          key_mask,
          attrs,
          q_t,
          k_t,
          v_t,
          k_put_slice,
          v_put_slice,
          state
        )

      :no_match ->
        emit_metadata_instr(
          id,
          :fast_sdpa_causal_key_masked,
          [q_t, k_t, v_t, key_mask],
          attrs,
          state
        )
    end
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
    emit_metadata_instr(id, opcode, operands, attrs, state)
  end

  # `EMLX.Quantization.quantize/2`'s traced representation — see
  # `scope_dependencies/1`'s matching clause above. This node has no single
  # physical ref of its own: `lower/2`'s output-ref builder reads
  # `__EMLX_QUANT__` straight off an output leaf and emits its 2-3
  # underlying refs directly, bypassing `node_to_ref` entirely for this id.
  # Feeding this value into anything *other* than a direct output (e.g.
  # `Nx.dot`) isn't supported yet — that consumer's own `Map.fetch!` on
  # `node_to_ref` raises a `KeyError` instead of silently reading a
  # meaningless ref.
  defp expand_node(
         %T{data: %Nx.Defn.Expr{op: :metadata, args: [_inner, %{__EMLX_QUANT__: _quant}]}},
         state
       ) do
    state
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

  # reshape: attrs = new shape dims (flat list); shape from the output tensor.
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

  # squeeze: attrs = axes to remove (non-negative).
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

  # transpose: attrs = axis permutation (non-negative).
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

  # bitcast: attrs = [target_dtype]. Target type from the output tensor.
  defp expand_node(
         %T{type: out_type, data: %Nx.Defn.Expr{id: id, op: :bitcast, args: [tensor]}},
         state
       ) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    mlx_type = EMLX.Native.to_mlx_type(out_type)

    %{
      state
      | instructions: [{ref, :bitcast, [operand_ref], [mlx_type]} | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, ref)
    }
  end

  # broadcast: attrs = [n_shape, d0…, n_axes, a0…] (both shape and axes, length-delimited).
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :broadcast, args: [tensor, shape, axes]}},
         state
       ) do
    ref = make_ref()
    operand_ref = Map.fetch!(state.node_to_ref, tensor.data.id)
    shape_list = Tuple.to_list(shape)
    n_shape = length(shape_list)
    n_axes = length(axes)
    attrs = [n_shape | shape_list] ++ [n_axes | axes]

    %{
      state
      | instructions: [{ref, :broadcast, [operand_ref], attrs} | state.instructions],
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
        attrs = [n_dims | Enum.flat_map(config, fn {lo, hi, interior} -> [lo, hi, interior] end)]

        {ref,
         %{
           state
           | instructions: [{ref, :pad, [tensor_ref, pad_value_ref], attrs} | state.instructions]
         }}
      end

    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_ref)}
  end

  # reverse: attrs = axes to flip (non-negative).
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
      attrs = [keep_axes | normalize_axes(axes, tuple_size(tensor.shape))]

      state = %{
        state
        | instructions: [{ref, unquote(op), [operand_ref], attrs} | state.instructions]
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
      attrs = [keep_axes | normalize_axes(axes, tuple_size(tensor.shape))]

      %{
        state
        | instructions: [{ref, unquote(op), [operand_ref], attrs} | state.instructions],
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
    if quantized_param_config(left, state.quant_signature, state.stable_positions) do
      raise ArgumentError,
            "does not yet lower op :dot with a quantized left operand. " <>
              "Dequantize it first with EMLX.dequantize/1."
    end

    case quantized_param_config(right, state.quant_signature, state.stable_positions) do
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

    attrs =
      [n_dims, dynamic_mask] ++
        input_shape ++
        lengths ++
        strides ++
        static_vals

    operands = [tensor_ref | dyn_operand_refs]

    %{
      state
      | instructions: [{ref, :slice, operands, attrs} | state.instructions],
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

    attrs = [n_dims, dynamic_mask] ++ input_shape ++ lengths ++ static_vals
    operands = [input_ref, slice_ref | dyn_operand_refs]

    %{
      state
      | instructions: [{ref, :put_slice, operands, attrs} | state.instructions],
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

    attrs =
      [n_gather_axes | axes] ++
        [n_tensor_dims | slice_sizes] ++
        [length(out_shape_list) | out_shape_list]

    %{
      state
      | instructions: [{ref, :gather, [tensor_ref, indices_ref], attrs} | state.instructions],
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

      attrs =
        [n_dims, op_int] ++
          Enum.flat_map(0..(n_dims - 1), fn i ->
            [Enum.at(low_pads, i), Enum.at(high_pads, i)]
          end) ++
          strides ++ window_dims ++ window_dilations

      state = %{
        state
        | instructions: [{ref, unquote(op), [tensor_ref], attrs} | state.instructions]
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

      attrs =
        [n_dims] ++
          Enum.flat_map(0..(n_dims - 1), fn i ->
            [Enum.at(low_pads, i), Enum.at(high_pads, i)]
          end) ++
          strides ++ window_dims

      %{
        state
        | instructions: [
            {ref, unquote(op), [t_ref, src_ref, init_ref], attrs} | state.instructions
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
    f32_int = EMLX.Native.to_mlx_type({:f, 32})

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
    dtype_int = EMLX.Native.to_mlx_type(node.type)
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
    dtype_int = EMLX.Native.to_mlx_type(node.type)

    state = %{
      state
      | instructions: [{ref, :eye, [], [dtype_int, m, n]} | state.instructions]
    }

    if n_dims == 2 do
      %{state | node_to_ref: Map.put(state.node_to_ref, id, ref)}
    else
      broadcast_ref = make_ref()
      axes = [n_dims - 2, n_dims - 1]
      attrs = [n_dims | shape_list] ++ [length(axes) | axes]

      %{
        state
        | instructions: [{broadcast_ref, :broadcast, [ref], attrs} | state.instructions],
          node_to_ref: Map.put(state.node_to_ref, id, broadcast_ref)
      }
    end
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: :fun}}, state), do: state

  # A hook lowers to an inline `:runtime_call` (see `emit_hook_runtime_call/3`)
  # instead of the old "extra program output fired once after the whole
  # compiled program returns" scheme: that older scheme couldn't represent
  # "once per iteration" inside a native `:while`, which is exactly why
  # `native_eligible_node?/1` used to reject `:token` outright. Firing inline
  # gets per-iteration semantics for free (same mechanism validated for
  # `:runtime_call` generally -- see `EMLXRuntimeCall`'s moduledoc in
  # emlx_compiler.cpp), and each hook's runtime call also threads a
  # `hook_chain_ref` scalar through `state`: this is what still makes a hook
  # fire even when *this specific token's* hook value has no other consumer
  # (matching Evaluator's "the token, not the individual hook expr, is what
  # attach_token makes reachable" semantics -- see `Nx.Defn.Kernel.hook/3`'s
  # doc), and forces a deterministic firing order between hooks that aren't
  # already ordered by a genuine data dependency (MLX's own execution order
  # for independent primitives is not a documented guarantee). See
  # `EMLX.Native.Expr.t/0`'s `keepalive_refs` doc for where the final tip of
  # this chain ends up.
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

    # `token.hooks` stores the most-recently-declared hook at the head (see
    # Nx.Defn.Token's moduledoc) -- reverse to chain them in true declaration
    # order.
    Enum.reduce(Enum.reverse(hooks), state, fn
      %{callback: nil}, state ->
        state

      %{callback: callback, expr: expr}, state ->
        emit_hook_runtime_call(expr, callback, state)
    end)
  end

  # Resolves to the hooked (post-side-effect) value when `expr` is one of
  # *this* token's own hooked expressions -- see `emit_hook_runtime_call/3`'s
  # `hooked_value_refs` doc -- falling back to the plain ref otherwise (a
  # name-only hook with no callback, or `attach_token` wrapping something
  # that isn't itself hooked).
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :attach_token, args: [_token, expr]}},
         state
       ) do
    ref =
      case Map.fetch(state.hooked_value_refs, expr.data.id) do
        {:ok, hooked_ref} -> hooked_ref
        :error -> Map.fetch!(state.node_to_ref, expr.data.id)
      end

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

    output_templates =
      case out_template do
        %Nx.Tensor{} = t -> [t]
        container -> Composite.flatten_list([container])
      end

    {result_refs, state} =
      case Keyword.pop(opts, :__emlx_native_multi_op__) do
        {nil, _opts} ->
          # A bare `%T{op: :parameter}` leaf is only a genuine top-level
          # position when it IS a top-level parameter; inside a `:while`
          # sub-program, `arg`'s parameters have their own local 0..N-1
          # numbering, unrelated to this program's real input list.
          # `stable_positions` (populated by the top-level :parameter clause
          # and propagated through invariant `while` carries by
          # `expand_while_native/6`) resolves the correct position in both
          # cases, and is `nil` when no such stable identity exists (e.g. the
          # operand is computed, or a while carry that actually changes per
          # iteration) -- see EMLX.handle_runtime_call/6's `positions` doc.
          arg_param_positions =
            Enum.map(operand_leaves, &Map.get(state.stable_positions, &1.data.id))

          args_template = Composite.traverse(tensor_expr, &Nx.to_template/1)

          emit_runtime_call_refs(
            operand_refs,
            arg_param_positions,
            args_template,
            callback,
            opts,
            output_templates,
            state
          )

        {{native_opcode, native_attrs}, _opts} ->
          # Escape hatch for a genuine `multi_op_registry` C++ op (no BEAM
          # round-trip, unlike the `EMLXRuntimeCall`/`invoke_runtime_call`
          # path above) that still wants `Nx.runtime_call`'s existing
          # multi-output container/`:elem` machinery and eager/grad-fallback
          # callback for free, instead of duplicating that plumbing. `callback`
          # is kept as this node's eager/`Nx.Defn.Evaluator`/`Nx.Defn.Grad`
          # fallback (never invoked on this native-compiled path); only the
          # opcode + static attrs reach the interpreter.
          emit_native_multi_op_refs(
            operand_refs,
            native_opcode,
            native_attrs,
            output_templates,
            state
          )
      end

    result_id =
      case out_template do
        %Nx.Tensor{} -> hd(result_refs)
        _ -> result_refs
      end

    %{state | node_to_ref: Map.put(state.node_to_ref, id, result_id)}
  end

  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :while, args: [initial, arg, condition, body]}},
         state
       ) do
    case detect_static_while_trip_count(initial, arg, condition, body) do
      {:ok, count} ->
        expand_while_unroll(id, initial, arg, body, count, state)

      :error ->
        if native_while_eligible?(condition, body) do
          expand_while_native(id, initial, arg, condition, body, state)
        else
          raise ArgumentError, "does not yet lower op :while"
        end
    end
  end

  defp expand_node(%T{data: %Nx.Defn.Expr{op: op}}, _state) do
    raise ArgumentError, "does not yet lower op #{inspect(op)}"
  end

  # Emits one `:runtime_call` instruction from already-resolved operand refs
  # (as opposed to the `:runtime_call` op's own `expand_node` clause, which
  # additionally has to resolve those refs from a real `Nx.Defn.Expr` --
  # `emit_hook_runtime_call/3` builds its operands from a hook's `expr`
  # instead). Shared so both call sites stay byte-for-byte identical in how
  # they build the wire `attrs`/`runtime_call` map. Always returns a *list*
  # of result refs (even for a single output) -- safe for the instruction's
  # own result identifier (`register_result_refs/3` treats a 1-element list
  # the same as a bare ref), but callers that need `node_to_ref` to hold a
  # bare ref for a bare-tensor output must unwrap it themselves (see the
  # `:runtime_call` clause above).
  defp emit_runtime_call_refs(
         operand_refs,
         arg_param_positions,
         args_template,
         callback,
         opts,
         output_templates,
         state
       ) do
    callback_index = length(state.runtime_calls)

    attrs =
      [callback_index, length(output_templates)] ++
        Enum.flat_map(output_templates, fn t ->
          dtype_int = EMLX.Native.to_mlx_type(t.type)
          shape = Tuple.to_list(t.shape)
          [dtype_int, length(shape) | shape]
        end)

    runtime_call = %{
      index: callback_index,
      callback: callback,
      args_template: args_template,
      arg_param_positions: arg_param_positions,
      opts: opts
    }

    result_refs = Enum.map(output_templates, fn _ -> make_ref() end)

    state = %{
      state
      | instructions: [{result_refs, :runtime_call, operand_refs, attrs} | state.instructions],
        runtime_calls: [runtime_call | state.runtime_calls]
    }

    {result_refs, state}
  end

  # Sibling of `emit_runtime_call_refs/7` for the `:__emlx_native_multi_op__`
  # escape hatch (see the `:runtime_call` `expand_node/2` clause above):
  # emits one plain multi-output instruction dispatched to `opcode`
  # (expected to resolve via `emlx_compiler.cpp`'s `multi_op_registry`, the
  # same mechanism already backing `qr`/`svd`/`lu` -- pure native C++, no
  # `EMLXRuntimeCall`/BEAM round-trip, unlike `emit_runtime_call_refs/7`).
  # `attrs` are static (baked in at trace time, like every other op's attrs);
  # any per-iteration-dynamic values must instead be threaded as extra
  # `operand_refs` (see `:put_slice`'s dynamic-start-index pattern).
  defp emit_native_multi_op_refs(operand_refs, opcode, attrs, output_templates, state) do
    result_refs = Enum.map(output_templates, fn _ -> make_ref() end)

    state = %{
      state
      | instructions: [{result_refs, opcode, operand_refs, attrs} | state.instructions]
    }

    {result_refs, state}
  end

  # Emits a scalar (shape `{}`) constant, mirroring the `:constant` op's own
  # shape-`{}` case (see that `expand_node/2` clause) -- used to seed the
  # hook keepalive chain (see `emit_hook_runtime_call/3`) when there's no
  # earlier hook in the current scope to chain from.
  defp emit_scalar_constant(number, type, state) do
    ref = make_ref()
    {ref, %{state | constants: [{ref, number, type} | state.constants]}}
  end

  @hook_chain_type {:u, 32}

  # Converts one hook (`%{expr: ..., callback: ...}` from a `:token` node's
  # hook list) into an inline `:runtime_call`. The callback's *real* job
  # (calling the user's function for its side effect) is wrapped in a
  # closure that also passes the value through unchanged and advances the
  # keepalive chain -- see `EMLX.Native.Expr.t/0`'s `keepalive_refs` doc for
  # why the chain exists at all.
  #
  # `state.hooked_value_refs` (id of the *hooked* leaf -> its post-side-effect
  # ref) is separate from `state.node_to_ref` -- deliberately: `expr`'s own
  # `node_to_ref` entry (already computed by ordinary `expand_node`
  # processing before this `:token` node is reached) must be left untouched,
  # since some *other*, non-hooked use of the same underlying value elsewhere
  # in the graph must not be delayed behind this hook's side effect. Only
  # `:attach_token`'s own clause consults `hooked_value_refs`, which is
  # exactly the one place Nx's semantics say the hooked identity should be
  # visible.
  defp emit_hook_runtime_call(hook_expr, callback, state) do
    leaves = Composite.flatten_list([hook_expr])
    leaf_refs = Enum.map(leaves, &Map.fetch!(state.node_to_ref, &1.data.id))
    leaf_positions = Enum.map(leaves, &Map.get(state.stable_positions, &1.data.id))
    hook_template = Composite.traverse(hook_expr, &Nx.to_template/1)
    chain_template = Nx.template({}, @hook_chain_type)

    {chain_in_ref, state} =
      case state.hook_chain_ref do
        nil -> emit_scalar_constant(0, @hook_chain_type, state)
        ref -> {ref, state}
      end

    args_template = {hook_template, chain_template}
    output_templates = Composite.flatten_list([args_template])

    wrapped_callback = fn {value, chain_in}, _opts ->
      callback.(value)
      {value, Nx.add(chain_in, 1)}
    end

    {result_refs, state} =
      emit_runtime_call_refs(
        leaf_refs ++ [chain_in_ref],
        leaf_positions ++ [nil],
        args_template,
        wrapped_callback,
        [],
        output_templates,
        state
      )

    {value_refs, [chain_out_ref]} = Enum.split(result_refs, length(result_refs) - 1)

    hooked_value_refs =
      leaves
      |> Enum.zip(value_refs)
      |> Enum.reduce(state.hooked_value_refs, fn {leaf, ref}, acc ->
        Map.put(acc, leaf.data.id, ref)
      end)

    %{state | hook_chain_ref: chain_out_ref, hooked_value_refs: hooked_value_refs}
  end

  @doc false
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

  # Resolves via `stable_positions` (id -> true top-level parameter position,
  # see `EMLX.Native.Expr.t/0`'s stable-carry propagation doc) rather than
  # matching `op: :parameter, args: [pos]` and reading `pos` directly: a
  # `:dot` operand living inside a `:while` body/condition is a `:parameter`
  # node too, but scoped to that while's own local carry numbering, not the
  # top-level program's -- reading its `pos` directly would look up the wrong
  # (or a coincidentally-colliding) `quant_signature` key. `stable_positions`
  # already carries a top-level parameter's own position (see `expand_node/2`'s
  # `op: :parameter` clause) plus, for a while carry, whatever position the
  # carry is transitively "stable" (loop-invariant) at (see
  # `propagate_stable_carry_positions/4`), so this one lookup handles both.
  defp quantized_param_config(%T{data: %Nx.Defn.Expr{id: id}}, quant_signature, stable_positions) do
    case Map.fetch(stable_positions, id) do
      {:ok, pos} -> Map.get(quant_signature, pos)
      :error -> nil
    end
  end

  defp expand_plain_dot(id, out_type, left, c_left, right, c_right, b_left, b_right, state) do
    left_ref0 = Map.fetch!(state.node_to_ref, left.data.id)
    right_ref0 = Map.fetch!(state.node_to_ref, right.data.id)

    computation_type =
      if Nx.Type.integer?(out_type), do: Nx.Type.to_floating(out_type), else: out_type

    {left_ref, state} = emit_cast_if_needed(left_ref0, left.type, computation_type, state)
    {right_ref, state} = emit_cast_if_needed(right_ref0, right.type, computation_type, state)

    attrs =
      [length(c_left) | c_left] ++
        [length(c_right) | c_right] ++
        [length(b_left) | b_left] ++
        [length(b_right) | b_right]

    dot_ref = make_ref()

    state = %{
      state
      | instructions: [{dot_ref, :dot, [left_ref, right_ref], attrs} | state.instructions]
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

    mode_int = String.to_atom(cfg.mode)
    transpose_int = if transpose, do: 1, else: 0
    attrs = [cfg.group_size, cfg.bits, transpose_int, mode_int, has_bias]

    qmm_ref = make_ref()

    state = %{
      state
      | instructions: [{qmm_ref, :quantized_matmul, operands, attrs} | state.instructions]
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

    attrs = [length(axes) | axes] ++ [length(updates_shape) | updates_shape]

    %{
      state
      | instructions: [
          {ref, op, [target_ref, indices_ref, updates_ref], attrs} | state.instructions
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
         runtime_calls: inner_state.runtime_calls,
         hook_chain_ref: inner_state.hook_chain_ref,
         hooked_value_refs: inner_state.hooked_value_refs
     }}
  end

  # ── while (native lowering for dynamic-trip-count loops) ───────────────────
  #
  # See emlx_compiler.cpp's EMLXWhile primitive and emlx_compiler.hpp's
  # SubProgram/RefKind::Carry. Instead of unrolling (impossible: the trip
  # count is data-dependent) or splitting the graph and driving the loop
  # from Elixir (the old `emlx.ex` `build_while_base_eval_fn`/`run_while_loop`
  # path, still used as a fallback -- see `native_while_eligible?/2`), this
  # lowers straight to a single `:while` instruction carrying two
  # self-contained cond/body sub-programs, so the whole loop runs natively
  # inside one `eval_program` NIF call.

  # A `:while` loop is native-lowerable unconditionally now: `:runtime_call`
  # and `:token`/hooks (which lower to an inline `:runtime_call`, see
  # `emit_hook_runtime_call/3`) are both native-eligible inside a while's
  # condition/body -- see `native_eligible_node?/1`. `native_while_eligible?/2`
  # is kept as an explicit, named check (rather than inlining `true`) so
  # `EMLX.ex`'s `split_point?/1` has a single shared source of truth to call,
  # in case a future node type needs to reintroduce a real restriction here.
  # Nested `:while` *is* supported: EMLXWhile's cond/body
  # sub-programs are interpreted by the same generic `interpret_instructions`
  # helper used for the top-level program, so a nested `:while` instruction
  # just recurses into another `EMLXWhile` primitive from inside the outer
  # one's `eval()` -- validated safe (2- and 3-level nesting, both default
  # CPU and GPU devices) against the checkpoint (a) spike's reentrant-`eval()`
  # finding, which generalizes because C++ call-stack depth here scales with
  # *lexical* nesting depth (fixed at compile time), not iteration count.
  # `expand_while_native/6` below still guards against pathologically deep
  # (e.g. generated) nesting with `@max_while_nesting_depth`, purely to turn
  # a native stack overflow into a clean Elixir error -- not because
  # correctness is in doubt at any realistic depth. Shared between here and
  # `EMLX.ex`'s `split_point?/1` so the eligibility criteria can't drift
  # between the two call sites (each does its own cheap traversal; only the
  # *logic* is required to stay in sync).
  @doc false
  def native_while_eligible?(condition, body) do
    Enum.all?([condition, body], &while_scope_native_eligible?/1)
  end

  defp while_scope_native_eligible?(container) do
    container
    |> EMLX.Defn.Tree.post_order(&scope_dependencies/1)
    |> Enum.all?(&native_eligible_node?/1)
  end

  # Every node is native-eligible: `:runtime_call` is backed by a genuine
  # `mx::core::Primitive` (`EMLXRuntimeCall`, emlx_compiler.cpp) that re-fires
  # on every replay of the compiled tape, including replays driven by
  # `EMLXWhile`'s own per-iteration `mlx::core::eval()` calls (validated safe:
  # both are CPU-pinned primitives triggering nested `eval()` from inside a
  # primitive's own `eval()`, the same pattern already validated for
  # while-inside-while nesting -- see `EMLXWhile`'s moduledoc comment in
  # emlx_compiler.cpp). See `propagate_stable_carry_positions/4` for how a
  # `:runtime_call` operand recovers its true top-level position when it's a
  # `while`-carried value instead of a direct top-level parameter. `:token`
  # (hooks) lowers to an inline `:runtime_call` too (see
  # `emit_hook_runtime_call/3`), same per-iteration semantics -- the only
  # remaining restriction is structural, not eligibility-based: a hook whose
  # value isn't already the condition's own boolean result can't sit inside
  # a `:while` *condition* (fixed at exactly one output -- see
  # `expand_while_native/6`'s keepalive-widening raise).
  defp native_eligible_node?(%T{}), do: true

  # Defensive cap on lexical `:while`-inside-`:while` nesting depth: each
  # extra level adds a handful of native C++ call-stack frames (outer
  # `EMLXWhile::eval` -> `interpret_instructions` -> inner `EMLXWhile::eval`,
  # reentered via `mlx::core::eval()`), so unbounded nesting from generated
  # code could in principle exhaust the stack. 64 levels is far beyond any
  # hand-written loop nest while leaving ample stack headroom.
  @max_while_nesting_depth 64

  defp expand_while_native(id, initial, arg, condition, body, state) do
    depth = state.while_nesting_depth

    if depth >= @max_while_nesting_depth do
      raise ArgumentError,
            "while loops nested #{@max_while_nesting_depth} levels deep exceed EMLX's " <>
              "native lowering depth limit (a defensive guard against unbounded native " <>
              "C++ call-stack recursion -- not a real Nx.Defn.while restriction)"
    end

    initial_list = while_leaf_list(initial)
    arg_list = while_leaf_list(arg)
    body_list = while_leaf_list(body)

    init_refs = Enum.map(initial_list, &Map.fetch!(state.node_to_ref, &1.data.id))
    carry_refs = Enum.map(arg_list, fn _ -> make_ref() end)

    param_ref_by_pos =
      arg_list
      |> Enum.zip(carry_refs)
      |> Map.new(fn {param, ref} -> {hd(param.data.args), ref} end)

    stable_positions =
      propagate_stable_carry_positions(arg_list, body_list, initial_list, state.stable_positions)

    # A hook inside cond/body needs its own scope-local keepalive chain (see
    # `EMLX.Native.Expr.t/0`'s `keepalive_refs` doc) -- `Nx.Defn.while`'s
    # closure rules mean cond/body can't reference anything outside their own
    # carry/captures/constants (enforced structurally: a sub-program's own
    # `{:result, i}` numbering is local, see `to_native_subprogram/3`), so an
    # *ambient* chain from outside this `:while` can't simply be read inside
    # -- it has to come in as one more (fixed-width, invariant-shape) carry
    # slot instead, exactly like any other loop-invariant value threaded
    # through untouched. Only paid for when a hook is actually present.
    needs_keepalive? = while_scope_contains_hook?(condition) or while_scope_contains_hook?(body)

    {keepalive_seed, init_refs, sub_carry_refs, state} =
      if needs_keepalive? do
        {seed_ref, state} =
          case state.hook_chain_ref do
            nil -> emit_scalar_constant(0, @hook_chain_type, state)
            ref -> {ref, state}
          end

        keepalive_carry_ref = make_ref()
        {keepalive_carry_ref, init_refs ++ [seed_ref], carry_refs ++ [keepalive_carry_ref], state}
      else
        {nil, init_refs, carry_refs, state}
      end

    nested_state = %{
      state
      | while_nesting_depth: depth + 1,
        stable_positions: stable_positions,
        hook_chain_ref: keepalive_seed,
        hooked_value_refs: %{}
    }

    {cond_sub, state} =
      lower_while_subprogram([condition], sub_carry_refs, param_ref_by_pos, nested_state)

    if cond_sub.hook_chain_ref != keepalive_seed do
      raise ArgumentError,
            "cannot lower a hook inside a :while condition unless its value is exactly the " <>
              "condition's own boolean result -- EMLXWhile's condition sub-program is fixed " <>
              "at exactly one output (see EMLXWhile::evaluate_predicate in " <>
              "emlx_compiler.cpp), so there is no way to force evaluation of a hook whose " <>
              "value the returned boolean doesn't already depend on. Move the hook into the " <>
              "while's body instead."
    end

    {body_sub, state} =
      lower_while_subprogram(body_list, sub_carry_refs, param_ref_by_pos, %{
        state
        | while_nesting_depth: depth + 1,
          stable_positions: stable_positions,
          hook_chain_ref: keepalive_seed,
          hooked_value_refs: %{}
      })

    state = %{state | while_nesting_depth: depth}

    body_sub =
      if needs_keepalive? do
        %{body_sub | outputs: body_sub.outputs ++ [body_sub.hook_chain_ref]}
      else
        body_sub
      end

    result_refs = Enum.map(init_refs, fn _ -> make_ref() end)

    {real_result_refs, outer_chain_ref} =
      if needs_keepalive? do
        {real_refs, [chain_ref]} = Enum.split(result_refs, length(result_refs) - 1)
        {real_refs, chain_ref}
      else
        {result_refs, nil}
      end

    result_id = if match?([_], real_result_refs), do: hd(real_result_refs), else: real_result_refs

    instr = {result_refs, :while, init_refs, [], [cond_sub, body_sub]}

    %{
      state
      | instructions: [instr | state.instructions],
        node_to_ref: Map.put(state.node_to_ref, id, result_id),
        hook_chain_ref: outer_chain_ref
    }
  end

  # Cheap pre-scan (separate from `native_eligible_node?`'s own traversal,
  # since we need this decided *before* lowering starts, to size the carry --
  # see `expand_while_native/6`) for whether a `:while` scope needs the extra
  # keepalive carry slot at all. A name-only hook (nil callback) never emits
  # a runtime_call (see `emit_hook_runtime_call/3`'s caller), so it doesn't
  # count.
  defp while_scope_contains_hook?(container) do
    container
    |> EMLX.Defn.Tree.post_order(&scope_dependencies/1)
    |> Enum.any?(fn
      %T{data: %Nx.Defn.Expr{op: :token, args: [%Nx.Defn.Token{hooks: hooks}]}} ->
        Enum.any?(hooks, &(&1.callback != nil))

      _ ->
        false
    end)
  end

  defp while_leaf_list(container) when is_tuple(container), do: Tuple.to_list(container)
  defp while_leaf_list(leaf), do: [leaf]

  # A carry slot is "stable" (its runtime value is identical on every
  # iteration, including the first) exactly when the loop body returns it
  # completely unmodified -- i.e. `body`'s i-th output IS `arg`'s i-th
  # parameter (same node id, not just an equal value). For such a slot, a
  # `:runtime_call` deep inside the loop (see the `:runtime_call` clause of
  # `expand_node/2`) can safely resolve its operand back to `initial`'s own
  # stable position (if any), instead of the carry's local, while-scoped
  # parameter numbering -- required for `EMLX.handle_runtime_call/6`'s
  # quantized-tensor fast path to recover the right argument when a quantized
  # weight is threaded through a `while` unchanged (e.g. `dequantize/1` called
  # once per iteration on a carried `qw`). Chains transitively through nested
  # whiles for free: `initial_i` may itself be a stable position propagated
  # from an enclosing while's own carry. Slots that are genuinely recomputed
  # per iteration get no entry, so `Map.get/2` on `stable_positions` for them
  # is `nil` -- correctly forcing the safe (if lossy-for-sub-byte-types)
  # binary-decode fallback instead of reusing a stale Elixir tensor.
  defp propagate_stable_carry_positions(arg_list, body_list, initial_list, stable_positions) do
    [arg_list, body_list, initial_list]
    |> Enum.zip()
    |> Enum.reduce(stable_positions, fn {arg_i, body_i, initial_i}, acc ->
      if body_i.data.id == arg_i.data.id do
        case Map.fetch(stable_positions, initial_i.data.id) do
          {:ok, pos} -> Map.put(acc, arg_i.data.id, pos)
          :error -> acc
        end
      else
        acc
      end
    end)
  end

  # Lowers a `:while` sub-scope (the condition wrapped in a 1-element list,
  # or the body leaf list) into an isolated internal sub-program: unlike
  # `lower_tuple_body/3` (used for *unrolling*, which splices the inner
  # instructions into the outer `state.instructions` stream once per
  # iteration), the inner instructions here stay local to the returned
  # sub-program map -- they're converted to a wire `EMLX.Native.SubProgram`
  # later, in `to_native_subprogram/3`. Only captures/constants/inputs/hooks
  # merge back into the outer state, since those are shared global tables
  # (see EMLX.Native.SubProgram's moduledoc note on shared captures/consts).
  # `carry_refs` (this while's fresh per-slot refs, positionally matching
  # `arg`/`initial`) is stashed on the returned map so `to_native_subprogram/3`
  # knows which internal refs become `{:carry, i}` on the wire.
  defp lower_while_subprogram(exprs, carry_refs, param_ref_by_pos, state) do
    exprs_tuple = List.to_tuple(exprs)
    state = merge_scope_ids(state, exprs_tuple)
    inner_ordered = EMLX.Defn.Tree.post_order(exprs_tuple, &scope_dependencies/1)

    param_id_to_ref =
      inner_ordered
      |> Enum.filter(&(&1.data.op == :parameter))
      |> Map.new(fn p -> {p.data.id, Map.fetch!(param_ref_by_pos, hd(p.data.args))} end)

    param_id_set = MapSet.new(Map.keys(param_id_to_ref))
    local_base = Map.merge(state.node_to_ref, param_id_to_ref)

    inner_state =
      Enum.reduce(
        inner_ordered,
        %{
          state
          | node_to_ref: local_base,
            instructions: [],
            refcounts: compute_refcounts(inner_ordered)
        },
        fn node, st ->
          if MapSet.member?(param_id_set, node.data.id), do: st, else: expand_node(node, st)
        end
      )

    output_refs = Enum.map(exprs, &Map.fetch!(inner_state.node_to_ref, &1.data.id))

    sub_program = %{
      carry_refs: carry_refs,
      instructions: Enum.reverse(inner_state.instructions),
      outputs: output_refs,
      hook_chain_ref: inner_state.hook_chain_ref
    }

    {sub_program,
     %{
       state
       | captures: inner_state.captures,
         constants: inner_state.constants,
         inputs: inner_state.inputs,
         runtime_calls: inner_state.runtime_calls
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
         runtime_calls: inner_state.runtime_calls,
         hook_chain_ref: inner_state.hook_chain_ref,
         hooked_value_refs: inner_state.hooked_value_refs
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
    attrs = [length(shape_list) | shape_list] ++ [0]

    {new_ref, %{state | instructions: [{new_ref, :broadcast, [ref], attrs} | state.instructions]}}
  end

  # Slice element `i` along the collapsed trailing axis then squeeze it away,
  defp emit_reduce_slice(ref, combined_shape, kept_shape, i, state) do
    n_dims = length(combined_shape)
    last_axis = n_dims - 1
    lengths = kept_shape ++ [1]
    strides = List.duplicate(1, n_dims)
    starts = List.duplicate(0, length(kept_shape)) ++ [i]
    attrs = [n_dims, 0] ++ combined_shape ++ lengths ++ strides ++ starts

    slice_ref = make_ref()
    state = %{state | instructions: [{slice_ref, :slice, [ref], attrs} | state.instructions]}
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

    attrs =
      [n_dims | Enum.flat_map(Enum.zip(low_pads, high_pads), fn {lo, hi} -> [lo, hi, 0] end)]

    {new_ref,
     %{
       state
       | instructions: [{new_ref, :pad, [ref, pad_value_ref], attrs} | state.instructions]
     }}
  end

  defp emit_static_slice(ref, input_shape, starts, lengths, strides, state) do
    new_ref = make_ref()
    n_dims = length(input_shape)
    attrs = [n_dims, 0] ++ input_shape ++ lengths ++ strides ++ starts

    {new_ref, %{state | instructions: [{new_ref, :slice, [ref], attrs} | state.instructions]}}
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

  # Only ever finds *bound top-level parameter* positions (propagated
  # through invariant `while` carries) — a quantized value produced
  # in-graph by `EMLX.Quantization.quantize/2` (a `:metadata`/
  # `__EMLX_QUANT__` node, see its moduledoc's "Limitation" section) has no
  # such position and is never recognized as a `:dot` operand here, even
  # when fed straight into one. Extending quantized `:dot` dispatch to
  # cover that case would mean recognizing `__EMLX_QUANT__` nodes here too,
  # and reworking `expand_quantized_dot/8` to consume scales/biases as
  # graph-flowing operands (via their own `node_to_ref` entries) instead of
  # `emit_capture/2`-ing real, call-time-known tensors — out of scope today.
  @doc false
  def quantizable_param_positions(output) do
    {positions, _stable} = quantizable_positions_in_scope(output, %{})
    positions
  end

  # Walks a single lexical scope (the top-level program, or a `:while`
  # condition/body) collecting the top-level parameter positions consumed as
  # a `:dot` operand anywhere in `container` -- descending into nested
  # `:while` sub-scopes (which `EMLX.Defn.Tree.post_order/2` otherwise treats
  # as opaque, see its moduledoc) so a quantized weight threaded unchanged
  # through a `while` carry is still recognized as quantizable. Without this,
  # e.g. Bumblebee's `Nx.Defn.while`-based generation loop (which carries
  # `params` -- including quantized projection weights -- unchanged through
  # the whole decode loop, see `Bumblebee.Text.Generation`) would never
  # register any position here, `quant_signature/2` (emlx.ex) would come back
  # empty, and every `:dot` against those weights would lower as a plain
  # (non-quantized) `:dot` against the tensor's physical *packed* shape --
  # producing a `[tensordot] a and b must have the same shape on the
  # contracted axes` NIF error instead of a `:quantized_matmul`.
  #
  # Mirrors `propagate_stable_carry_positions/4` (used by the real lowering
  # path) but as a standalone static pre-pass: this runs *before*
  # `EMLX.Native.Expr.lower/3`'s own stateful traversal, directly on
  # `output_expr`, with no `state` to reuse.
  #
  # Returns `{positions, stable_positions}`; `stable_positions` maps each
  # node id in `container`'s own scope back to its top-level parameter
  # position (only for slots ultimately backed by a genuine top-level
  # parameter, directly or via an outer while's own stable carry) -- returned
  # so processing a `:while`'s condition and body (both closing over the same
  # carries) can share one computation, and so a nested `:while` can chain
  # through its outer scope's stable positions.
  defp quantizable_positions_in_scope(container, stable_positions) do
    container
    |> EMLX.Defn.Tree.post_order(&scope_dependencies/1)
    |> Enum.reduce({MapSet.new(), stable_positions}, fn
      %T{data: %Nx.Defn.Expr{id: id, op: :parameter, args: [pos]}}, {positions, stable} ->
        {positions, Map.put(stable, id, pos)}

      %T{data: %Nx.Defn.Expr{op: :dot, args: [left, _c_left, _b_left, right, _c_right, _b_right]}},
      {positions, stable} ->
        positions =
          positions
          |> maybe_put_stable_position(left, stable)
          |> maybe_put_stable_position(right, stable)

        {positions, stable}

      %T{data: %Nx.Defn.Expr{op: :while, args: [initial, arg, condition, body]}},
      {positions, stable} ->
        initial_list = while_leaf_list(initial)
        arg_list = while_leaf_list(arg)
        body_list = while_leaf_list(body)

        inner_stable =
          propagate_stable_carry_positions(arg_list, body_list, initial_list, stable)

        {cond_positions, _} = quantizable_positions_in_scope(condition, inner_stable)
        {body_positions, _} = quantizable_positions_in_scope(body, inner_stable)

        {positions |> MapSet.union(cond_positions) |> MapSet.union(body_positions), stable}

      _, acc ->
        acc
    end)
  end

  defp maybe_put_stable_position(positions, %T{data: %Nx.Defn.Expr{id: id}}, stable) do
    case Map.fetch(stable, id) do
      {:ok, pos} -> MapSet.put(positions, pos)
      :error -> positions
    end
  end

  @doc false
  def f64_bits(v) when is_number(v) do
    <<bits::signed-64>> = <<v * 1.0::float-64>>
    bits
  end

  @doc false
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
    instr = {cast_ref, :astype, [ref], [mlx_type]}
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
  def to_native(%__MODULE__{} = prog) do
    # Build ref → wire-ref map for all non-instruction nodes. A wire ref is a
    # tagged tuple ({:input, i} / {:capture, i} / {:const, i} / {:result, i})
    # instead of a bit-packed int — see EMLX.Native.Instruction.ref/0.
    input_map =
      prog.inputs
      |> Enum.with_index()
      |> Map.new(fn {ref, i} -> {ref, {:input, i}} end)

    capture_map =
      prog.captures
      |> Enum.with_index()
      |> Map.new(fn {{ref, _t}, i} -> {ref, {:capture, i}} end)

    constant_map =
      prog.constants
      |> Enum.with_index()
      |> Map.new(fn {{ref, _v, _t}, i} -> {ref, {:const, i}} end)

    ref_to_wire = Map.merge(input_map, Map.merge(capture_map, constant_map))

    maybe_debug_check do
      expected_size = map_size(input_map) + map_size(capture_map) + map_size(constant_map)

      if map_size(ref_to_wire) != expected_size do
        raise ArgumentError,
              "EMLX.Native.Expr.to_native: ref id collision across inputs/captures/constants -- " <>
                "#{map_size(input_map)} input(s), #{map_size(capture_map)} capture(s), " <>
                "#{map_size(constant_map)} constant(s) should merge to #{expected_size} distinct " <>
                "refs, but only #{map_size(ref_to_wire)} survived Map.merge/2. This means two " <>
                "refs of different categories share the same id, silently dropping one from the " <>
                "wire map -- inputs: #{inspect(Map.keys(input_map))}, " <>
                "captures: #{inspect(Map.keys(capture_map))}, " <>
                "constants: #{inspect(Map.keys(constant_map))}"
      end
    end

    {instructions, ref_to_wire, _flat} =
      to_native_instructions(prog.instructions, ref_to_wire, capture_map, constant_map)

    wire_outputs = Enum.map(prog.outputs ++ prog.keepalive_refs, &Map.fetch!(ref_to_wire, &1))
    num_real_outputs = length(prog.outputs)

    capture_nif_refs =
      Enum.map(prog.captures, fn {_ref, %Nx.Tensor{data: %EMLX.Backend{ref: {_, nif_ref}}}} ->
        nif_ref
      end)

    wire_constants =
      Enum.map(prog.constants, fn {_, v, t} -> {v * 1.0, EMLX.Native.to_mlx_type(t)} end)

    %EMLX.Native.Program{
      num_inputs: length(prog.inputs),
      captures: capture_nif_refs,
      constants: wire_constants,
      instructions: Enum.reverse(instructions),
      outputs: wire_outputs,
      num_real_outputs: num_real_outputs
    }
  end

  # Converts one flat internal instruction list (a Program's top-level
  # `instructions`, or -- recursively -- a `:while` sub-program's own local
  # `instructions`, see to_native_subprogram/3) into wire
  # `EMLX.Native.Instruction`s, threading a `ref_to_wire` map + flat
  # `{:result, i}` counter that starts wherever `seed_ref_to_wire` leaves off
  # (0 for a fresh sub-program interpretation, matching emlx_compiler.hpp's
  # SubProgram semantics). `capture_map`/`constant_map` are always the
  # *outer* Program's tables, threaded down unchanged: a `:while`
  # sub-program's `{:capture, i}`/`{:const, i}` refs resolve against the same
  # shared tables as the parent program, never a duplicated per-sub-program
  # copy (see EMLX.Native.SubProgram).
  defp to_native_instructions(instructions, seed_ref_to_wire, capture_map, constant_map) do
    Enum.reduce(instructions, {[], seed_ref_to_wire, 0}, fn
      {id, :while, operand_refs, attrs, [cond_sub, body_sub]}, {instrs, rmap, flat} ->
        wire_operands = Enum.map(operand_refs, &Map.fetch!(rmap, &1))
        wire_cond = to_native_subprogram(cond_sub, capture_map, constant_map)
        wire_body = to_native_subprogram(body_sub, capture_map, constant_map)

        maybe_debug_check do
          for one <- List.wrap(id), Map.has_key?(rmap, one) do
            raise ArgumentError,
                  "EMLX.Native.Expr.to_native: instruction :while produces result ref " <>
                    "#{inspect(one)}, but that ref is already bound (to " <>
                    "#{inspect(Map.fetch!(rmap, one))}) -- the same node id was lowered twice, " <>
                    "silently overwriting its earlier binding for every prior instruction that " <>
                    "already referenced it."
          end
        end

        {rmap2, flat2} = register_result_refs(id, rmap, flat)

        instr = %EMLX.Native.Instruction{
          op: :while,
          operands: wire_operands,
          attrs: attrs,
          subprograms: [wire_cond, wire_body]
        }

        {[instr | instrs], rmap2, flat2}

      {id, op, operand_refs, attrs}, {instrs, rmap, flat} ->
        wire_operands = Enum.map(operand_refs, &Map.fetch!(rmap, &1))

        maybe_debug_check do
          for {:result, idx} <- wire_operands, idx >= flat do
            raise ArgumentError,
                  "EMLX.Native.Expr.to_native: instruction #{inspect(op)} (id=#{inspect(id)}) " <>
                    "references result index #{idx} of the flat results accumulator, but only " <>
                    "#{flat} result(s) have been produced so far -- this is a forward/self " <>
                    "reference bug in program lowering, not a valid program. Full instruction " <>
                    "list: #{inspect(instructions)}"
          end
        end

        maybe_debug_check do
          for one <- List.wrap(id), Map.has_key?(rmap, one) do
            raise ArgumentError,
                  "EMLX.Native.Expr.to_native: instruction #{inspect(op)} produces result ref " <>
                    "#{inspect(one)}, but that ref is already bound (to " <>
                    "#{inspect(Map.fetch!(rmap, one))}) -- the same node id was lowered twice, " <>
                    "silently overwriting its earlier binding for every prior instruction that " <>
                    "already referenced it."
          end
        end

        {rmap2, flat2} = register_result_refs(id, rmap, flat)

        instr = %EMLX.Native.Instruction{op: op, operands: wire_operands, attrs: attrs}
        {[instr | instrs], rmap2, flat2}
    end)
  end

  # Registers `id` (a single ref, or a list of refs for a multi-output
  # instruction) into `rmap` as the next `{:result, i}` slot(s), advancing
  # the flat counter by one per ref. Shared between the top-level Program
  # walk and the `:while` sub-program walk in to_native_instructions/4.
  defp register_result_refs(id, rmap, flat) do
    case id do
      ids when is_list(ids) ->
        Enum.reduce(ids, {rmap, flat}, fn one, {m, f} ->
          {Map.put(m, one, {:result, f}), f + 1}
        end)

      one ->
        {Map.put(rmap, one, {:result, flat}), flat + 1}
    end
  end

  # Converts a `:while` instruction's internal cond/body sub-scope (built by
  # lower_while_subprogram/3) into a wire `EMLX.Native.SubProgram`. Its own
  # `{:result, i}` numbering starts fresh at 0 (local to this sub-program,
  # matching emlx_compiler.hpp's SubProgram semantics) -- the seed map only
  # ever contains `{:carry, i}` (the sub-scope's own carry parameters) plus
  # the outer program's shared `{:capture, i}`/`{:const, i}` entries, never
  # the outer `{:input, i}`/`{:result, i}` entries: `Nx.Defn.while`'s closure
  # rules guarantee cond/body never reference anything else from the parent
  # scope, so this also structurally enforces that rule rather than merely
  # trusting it.
  defp to_native_subprogram(
         %{carry_refs: carry_refs, instructions: instructions, outputs: outputs},
         capture_map,
         constant_map
       ) do
    carry_map =
      carry_refs
      |> Enum.with_index()
      |> Map.new(fn {ref, i} -> {ref, {:carry, i}} end)

    seed_ref_to_wire = Map.merge(capture_map, Map.merge(constant_map, carry_map))

    {wire_instructions, local_ref_to_wire, _flat} =
      to_native_instructions(instructions, seed_ref_to_wire, capture_map, constant_map)

    wire_outputs = Enum.map(outputs, &Map.fetch!(local_ref_to_wire, &1))

    %EMLX.Native.SubProgram{instructions: Enum.reverse(wire_instructions), outputs: wire_outputs}
  end
end
