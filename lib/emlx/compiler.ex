defmodule EMLX.Compiler do
  @moduledoc """
  EMLX-native JIT compiler that builds an MLX lazy graph once and replays it
  cheaply on every subsequent call via `graph_capture` / `graph_replay`.

  ## Compilation strategy

  Graph capture is deferred to the **first actual call** (not compile time).
  On the first call:

  1. Run `Nx.Defn.Evaluator` with the real input tensors — this builds the
     correct lazy MLX DAG, including quantized-matmul kernels for 4-bit models.
  2. While the result is still lazy, call `graph_capture/3` to bake the tape.
  3. Store the `compiled_ref`; return the evaluator result for this first call.

  On every subsequent call `graph_replay/2` substitutes the new inputs into
  the baked tape — no NIF dispatch per node, no Nx expression walk.

  Falls back to the evaluator permanently when `graph_capture` fails, e.g.
  for defns containing `:runtime_call` (ETS / KV-cache) nodes.

  ## Deferred vs placeholder tracing

  An earlier iteration created zero-filled placeholder tensors at compile time
  and traced with those.  That approach silently breaks quantized models:
  placeholders are plain bf16 tensors, so the evaluator dispatches regular
  matmul; at replay time the actual weights are packed uint32, producing
  garbage logits.  Deferring to the first real call avoids this entirely.

  ## Note on ref formats

  `Backend.from_nx/1` returns `{device, ref}` tuples, but `graph_capture` /
  `graph_replay` NIFs use `enif_get_resource` on bare refs.  We extract bare
  refs before NIF calls and re-wrap with the device atom after.
  """

  require Logger

  alias EMLX.Backend
  alias Nx.Defn.Composite

  @doc """
  Returns a closure that replays a baked MLX graph on every call.

  The graph is captured on the **first invocation** of the returned closure
  using the actual runtime tensors.  The closure itself is returned immediately
  (no blocking at compile time beyond building the evaluator closure).
  """
  def compile(key, vars, fun, opts, queue) do
    evaluator_fn = Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)
    device = (queue && queue.device) || Keyword.get(opts, :device) || EMLX.default_device()

    # Lazy state: :uncompiled | {:replay, compiled_ref, output_template} | :fallback
    # Agent.start (not start_link) so the cell outlives any transient caller.
    {:ok, cell} = Agent.start(fn -> :uncompiled end)

    fn [params] ->
      case Agent.get(cell, & &1) do
        {:replay, compiled_ref, output_template} ->
          do_replay(compiled_ref, output_template, params, device, queue)

        state ->
          # Evaluator path — always produces the correct result.
          result_list = run_with_queue(queue, fn -> evaluator_fn.([params]) end)

          if state == :uncompiled do
            # While result tensors are still lazy, attempt graph_capture.
            # Thunks are called here (in the caller's process) so that any
            # queue-context requirements are met.
            inputs = Enum.map(params, fn f -> f.() end)

            input_bare_refs =
              Composite.flatten_list(inputs)
              |> Enum.map(&bare_ref(Backend.from_nx(&1)))

            new_state =
              try do
                [result] = result_list
                {output_bare_refs, tmpl} = extract_refs_and_template(result)
                {:ok, captured} = EMLX.NIF.graph_capture(input_bare_refs, output_bare_refs, false)
                {:replay, captured, tmpl}
              rescue
                e ->
                  Logger.warning(
                    "EMLX.Compiler: graph_capture failed, falling back to evaluator. " <>
                      "Reason: #{Exception.message(e)}"
                  )

                  :fallback
              end

            # First Agent.update caller wins; concurrent first calls keep the
            # evaluator result and only one of them commits the compiled state.
            Agent.update(cell, fn
              :uncompiled -> new_state
              existing -> existing
            end)
          end

          result_list
      end
    end
  end

  # ── Private helpers ─────────────────────────────────────────────────────────

  defp do_replay(compiled_ref, output_template, params, device, queue) do
    inputs = Enum.map(params, fn f -> f.() end)

    input_bare_refs =
      Composite.flatten_list(inputs)
      |> Enum.map(&bare_ref(Backend.from_nx(&1)))

    result =
      run_with_queue(queue, fn ->
        {:ok, new_bare_refs} = EMLX.NIF.graph_replay(compiled_ref, input_bare_refs)
        # Re-wrap bare refs with the device atom for to_nx.
        new_device_refs = Enum.map(new_bare_refs, fn r -> {device, r} end)
        # Force-eval while the queue context is active — replayed tensors are
        # bound to this queue's Metal stream and cannot be evaluated from another
        # thread's stream context (e.g. the default worker). After eval the data
        # is materialised; subsequent to_blob calls become cheap data reads.
        Enum.each(new_device_refs, &EMLX.eval/1)
        reconstruct_output(output_template, new_device_refs)
      end)

    [result]
  end

  defp run_with_queue(nil, fun), do: fun.()
  defp run_with_queue(queue, fun), do: EMLX.CommandQueue.with_queue(queue, fun)

  # Extracts the bare Erlang reference from a {device, ref} device_ref tuple.
  defp bare_ref({_device, ref}), do: ref

  # Flattens an Nx.Container into {[bare_ref], template_container}.
  defp extract_refs_and_template(container) do
    {template, bare_refs} =
      Composite.traverse(container, [], fn tensor, acc ->
        bare = bare_ref(Backend.from_nx(tensor))
        tmpl = Nx.to_template(tensor)
        {tmpl, [bare | acc]}
      end)

    {Enum.reverse(bare_refs), template}
  end

  # Puts new device_refs back into the template container, producing tensors.
  defp reconstruct_output(template, new_device_refs) do
    {result, []} =
      Composite.traverse(template, new_device_refs, fn tmpl, [device_ref | rest] ->
        {Backend.to_nx(device_ref, tmpl), rest}
      end)

    result
  end

end
