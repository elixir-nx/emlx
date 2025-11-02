defmodule EMLX.NIFError do
  defexception [:message]
end

defmodule EMLX.Macro do
  @moduledoc false

  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__)
      Module.register_attribute(EMLX, :mlx_function, accumulate: true)

      @before_compile EMLX.Macro
    end
  end

  @doc false
  defmacro __before_compile__(env) do
    mlx_functions = Module.get_attribute(env.module, :mlx_function)

    quote do
      def __mlx_functions__ do
        unquote(mlx_functions)
      end
    end
  end

  @doc """
  Function that receives a device and allocates a tensor.
  """
  defmacro defdevice(call) do
    {name, args} = Macro.decompose_call(call)

    unless has_device?(args) do
      raise("At least one argument of defdevice function should be named 'device'.")
    end

    tensors =
      case tensors(args) do
        [] -> :ok
        tensors -> quote do: {unquote(tensors), _} = prepare_tensors!(unquote(tensors))
      end

    quote do
      @mlx_function {unquote(name), unquote(length(args))}
      def unquote(name)(unquote_splicing(args)) do
        unquote(tensors)
        {user_device, index} = normalize_device!(var!(device))
        var!(device) = mlx_device!(user_device, index)

        EMLX.NIF.unquote(name)(unquote_splicing(args))
        |> unwrap_tensor!(user_device)
      end
    end
  end

  @doc """
  Generates a call that returns a tensor (or a tuple/list of tensors).

  All tensor variables must start with the name tensor.
  """
  defmacro deftensor(call) do
    defcall(call, :unwrap_tensor!, [Macro.var(:device, __MODULE__)])
  end

  @doc """
  Generates a call that returns a value (not a tensor).

  All tensor variables must start with the name tensor.
  """
  defmacro defvalue(call) do
    defcall(call, :unwrap!, [])
  end

  defp defcall(call, unwrapper, extra) do
    {name, args} = Macro.decompose_call(call)
    tensors = tensors(args)

    if tensors == [] do
      raise ArgumentError, "at least one tensor required in #{name}/#{length(args)}"
    end

    quote do
      @mlx_function {unquote(name), unquote(length(args) + length(extra))}
      def unquote(name)(unquote_splicing(args)) do
        {unquote(tensors), device} = prepare_tensors!(unquote(tensors))

        EMLX.NIF.unquote(name)(unquote_splicing(args ++ extra))
        |> unquote(unwrapper)(unquote_splicing(extra))
      end
    end
  end

  defp has_device?(args) do
    Enum.any?(args, &match?({:device, _, nil}, &1))
  end

  defp tensors(args) do
    Enum.filter(args, fn {name, _, _} -> match?("tensor" <> _, Atom.to_string(name)) end)
  end
end

defmodule EMLX do
  use EMLX.Macro

  defguard is_tensor(device, ref) when is_reference(ref) and is_atom(device)

  ## Macro callbacks

  defp normalize_device!({device, index}) when is_atom(device) and is_integer(index),
    do: {device, index}

  defp normalize_device!(device) when is_atom(device),
    do: {device, -1}

  defp normalize_device!(device),
    do: raise(ArgumentError, "expected device to be {atom, index} or atom, got: #{device}")

  defp mlx_device!(device, _index) do
    case device do
      :cpu -> :cpu
      :gpu -> :gpu
      _ -> raise ArgumentError, "unknown device #{inspect(device)}"
    end
  end

  ## Creation / conversion
  defdevice eye(m, n, type, device)
  defdevice from_blob(blob, shape, type, device)
  defdevice scalar_tensor(scalar, type, device)
  defdevice ones(shape, type, device)
  defdevice full(value, shape, type, device)
  defdevice arange(start, stop, step, integer?, device)

  ## Manipulation
  deftensor reshape(tensor, shape)
  deftensor broadcast_to(tensor, shape)
  deftensor astype(tensor, type)
  deftensor as_strided(tensor, shape, strides, offset)
  deftensor view(tensor, type)

  ## Binary ops
  deftensor add(tensorA, tensorB)
  deftensor subtract(tensorA, tensorB)
  deftensor multiply(tensorA, tensorB)
  deftensor pow(tensorA, tensorB)
  deftensor remainder(tensorA, tensorB)
  deftensor divide(tensorA, tensorB)
  deftensor atan2(tensorA, tensorB)
  deftensor bitwise_and(tensorA, tensorB)
  deftensor bitwise_or(tensorA, tensorB)
  deftensor bitwise_xor(tensorA, tensorB)
  deftensor bitwise_not(tensor)
  deftensor left_shift(tensorA, tensorB)
  deftensor right_shift(tensorA, tensorB)
  deftensor minimum(tensorA, tensorB)
  deftensor maximum(tensorA, tensorB)
  deftensor quotient(tensorA, tensorB)
  deftensor equal(tensorA, tensorB)
  deftensor not_equal(tensorA, tensorB)
  deftensor greater(tensorA, tensorB)
  deftensor less(tensorA, tensorB)
  deftensor greater_equal(tensorA, tensorB)
  deftensor less_equal(tensorA, tensorB)
  deftensor logical_and(tensorA, tensorB)
  deftensor logical_or(tensorA, tensorB)
  deftensor logical_xor(tensorA, tensorB)

  deftensor fft(tensor, n, axis)
  deftensor ifft(tensor, n, axis)
  deftensor fft2(tensor, s, axes)
  deftensor ifft2(tensor, s, axes)

  deftensor allclose(tensorA, tensorB, rtol, atol, equal_nan)
  deftensor isclose(tensorA, tensorB, rtol, atol, equal_nan)

  deftensor tensordot(tensorA, tensorB, axesA, axesB)
  deftensor einsum(tensorA, tensorB, spec_string)
  deftensor transpose(tensor, axes)
  deftensor pad(tensor, axes, low_pad_size, high_pad_size, tensor_pad_value)
  deftensor sort(tensor, axis)
  deftensor argsort(tensor, axis)
  deftensor tri_inv(tensor, upper)

  deftensor conv_general(
              tensor_input,
              tensor_kernel,
              strides,
              padding_low,
              padding_high,
              kernel_dilation,
              input_dilation,
              feature_group_count
            )

  ## Unary ops
  deftensor abs(tensor)
  deftensor ceil(tensor)
  deftensor conjugate(tensor)
  deftensor floor(tensor)
  deftensor negate(tensor)
  deftensor round(tensor)
  deftensor sign(tensor)
  deftensor real(tensor)
  deftensor imag(tensor)
  deftensor is_nan(tensor)
  deftensor is_infinity(tensor)
  deftensor logical_not(tensor)
  deftensor sigmoid(tensor)

  deftensor asin(tensor)
  deftensor asinh(tensor)
  deftensor acos(tensor)
  deftensor acosh(tensor)
  deftensor atan(tensor)
  deftensor atanh(tensor)
  deftensor cos(tensor)
  deftensor cosh(tensor)
  deftensor erf(tensor)
  deftensor erf_inv(tensor)
  deftensor exp(tensor)
  deftensor expm1(tensor)
  deftensor log(tensor)
  deftensor log1p(tensor)
  deftensor rsqrt(tensor)
  deftensor sin(tensor)
  deftensor sinh(tensor)
  deftensor sqrt(tensor)
  deftensor tan(tensor)
  deftensor tanh(tensor)

  ## Aggregation
  deftensor all(tensor, axes, keep_axes)
  deftensor any(tensor, axes, keep_axes)
  deftensor sum(tensor, axes, keep_axes)
  deftensor product(tensor, axes, keep_axes)
  deftensor argmax(tensor, keep_axes)
  deftensor argmax(tensor, axes, keep_axes)
  deftensor argmin(tensor, keep_axes)
  deftensor argmin(tensor, axes, keep_axes)
  deftensor cumulative_sum(tensor, axis, reverse, inclusive)
  deftensor cumulative_product(tensor, axis, reverse, inclusive)
  deftensor cumulative_max(tensor, axis, reverse, inclusive)
  deftensor cumulative_min(tensor, axis, reverse, inclusive)
  deftensor stack(tensors, axis)
  deftensor where(tensorPred, tensorTrue, tensorFalse)
  deftensor concatenate(tensors, axis)
  deftensor take_along_axis(tensor, tensorIndices, axis)
  deftensor take(tensor, tensorIndices, axis)
  deftensor gather(tensor, indices, axes, slice_sizes)
  deftensor scatter_add(tensor, indices, tensor_updates, axes)
  deftensor scatter(tensor, indices, tensor_updates, axes)
  deftensor max(tensor, axes, keep_axes)
  deftensor min(tensor, axes, keep_axes)
  deftensor clip(tensor, tensor_min, tensor_max)

  ## Dirty non-tensor return values
  defvalue scalar_type(tensor)
  defvalue shape(tensor)

  def to_blob({device, ref} = tensor) when is_tensor(device, ref) do
    # Two-step to_blob: eval on main scheduler, then copy on dirty scheduler
    eval(tensor)
    EMLX.NIF.to_blob(ref) |> unwrap!()
  end

  def to_blob({device, ref} = tensor, limit) when is_tensor(device, ref) do
    # Two-step to_blob: eval on main scheduler, then copy on dirty scheduler
    eval(tensor)
    EMLX.NIF.to_blob(ref, limit) |> unwrap!()
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise(EMLX.NIFError, List.to_string(error))

  defp unwrap_tensor!(tagged_result, device) do
    case unwrap!(tagged_result) do
      ref when is_reference(ref) ->
        {device, ref}

      list when is_list(list) ->
        Enum.map(list, &{device, &1})

      tuple when is_tuple(tuple) ->
        tuple |> Tuple.to_list() |> Enum.map(&{device, &1}) |> List.to_tuple()
    end
  end

  defp prepare_tensors_list!(tensors_list, device) do
    Enum.map_reduce(tensors_list, device, fn
      {dev, ref}, device when is_tensor(dev, ref) ->
        {ref, merge_device(device, dev)}

      bad_tensor, _device ->
        raise ArgumentError, "expected a EMLX tensor, got: #{inspect(bad_tensor)}"
    end)
  end

  defp prepare_tensors!(tensors) do
    Enum.map_reduce(tensors, :cpu, fn
      {dev, ref}, device when is_tensor(dev, ref) ->
        {ref, merge_device(device, dev)}

      [{dev, ref} | _] = tensors, device when is_tensor(dev, ref) ->
        prepare_tensors_list!(tensors, device)

      bad_tensor, _device ->
        raise ArgumentError, "expected a EMLX tensor, got: #{inspect(bad_tensor)}"
    end)
  end

  defp merge_device(:gpu, _), do: :gpu
  defp merge_device(_, :gpu), do: :gpu
  defp merge_device(_, _), do: :cpu

  defvalue deallocate(tensor_ref)
  defvalue eval(tensor)

  deftensor slice(tensor, starts, stops, strides)
  deftensor slice_update(tensor, tensor_updates, starts, stops)
  deftensor squeeze(tensor, axes)
  defvalue item(tensor)
  defvalue strides(tensor)

  @behaviour Nx.Defn.Compiler

  @impl Nx.Defn.Compiler
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl Nx.Defn.Compiler
  def __compile__(key, vars, fun, opts) do
    backend = Nx.default_backend()

    target_backend =
      case backend do
        EMLX.Backend ->
          backend

        {EMLX.Backend, _} ->
          backend

        Nx.BinaryBackend ->
          EMLX.Backend

        {Nx.BinaryBackend, _} ->
          EMLX.Backend

        other ->
          raise ArgumentError,
                "EMLX can only be used with the EMLX.Backend or Nx.BinaryBackend, got: #{inspect(other)}"
      end

    # Build the expression once with the vars
    expr = fun.(vars)

    fn [args] ->
      # Extract MLX array references and determine device
      {devices, nif_args} =
        Enum.map(args, fn arg ->
          case arg.() do
            %Nx.Tensor{data: %EMLX.Backend{ref: {device, ref}}} ->
              {device, ref}

            %Nx.Tensor{data: %Nx.BinaryBackend{}} = t ->
              %Nx.Tensor{data: %EMLX.Backend{ref: {device, ref}}} =
                Nx.backend_copy(t, target_backend)

              {device, ref}

            other ->
              %Nx.Tensor{data: %EMLX.Backend{ref: {device, ref}}} = Nx.to_tensor(other)
              {device, ref}
          end
        end)
        |> Enum.unzip()

      device =
        Enum.reduce_while(devices, :cpu, fn
          :gpu, _ -> {:halt, :gpu}
          _, acc -> {:cont, acc}
        end)

      cache_key = {__MODULE__, :compiled_fun, key}

      compiled_fun =
        case :persistent_term.get(cache_key, :not_found) do
          :not_found ->
            eval_fun = Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)

            # Start a task that will handle the compilation
            # This keeps eval_fun in the caller process instead of copying it to the runner
            caller_pid = self()

            task =
              Task.async(fn ->
                # The callback receives MLX array references wrapped in a tuple
                # It sends the refs back to the caller process for evaluation
                callback = fn {refs} ->
                  callback_ref = make_ref()
                  runner_pid = self()

                  # Send refs to the caller process for evaluation
                  send(caller_pid, {:eval_defn, callback_ref, refs, device, runner_pid})

                  # Wait for the result
                  receive do
                    {:eval_result, ^callback_ref, result_refs} -> result_refs
                  after
                    5_000 -> raise "Timeout waiting for eval_defn result"
                  end
                end

                # Register callback and compile
                tag = EMLX.Runner.register(EMLX.Runner, callback)

                try do
                  fun = nif_compile(nif_args, tag)
                  fun
                after
                  # Unregister the callback after compilation completes
                  EMLX.Runner.unregister(EMLX.Runner, tag)
                end
              end)

            # Wait for compilation and handle eval_defn requests
            fun = await_with_eval_handler(task, eval_fun, device)
            :persistent_term.put(cache_key, fun)
            fun

          cached_fun ->
            cached_fun
        end

      # Call the compiled MLX function with the current arguments
      nif_result =
        case device do
          :cpu -> EMLX.NIF.call_compiled_cpu(compiled_fun, nif_args)
          :gpu -> EMLX.NIF.call_compiled_gpu(compiled_fun, nif_args)
        end

      # Convert results back to Nx tensors
      results =
        nif_result
        |> unwrap!()
        |> Enum.map(fn ref -> EMLX.Backend.to_nx({device, ref}) end)

      # Reconstruct the output structure
      {result, []} =
        Nx.Defn.Composite.traverse(expr, results, fn _node, [h | t] ->
          {h, t}
        end)

      [result]
    end
  end

  # Helper function to await a task while handling eval_defn messages
  defp await_with_eval_handler(%Task{ref: task_ref} = task, eval_fun, device) do
    monitor_ref = Process.monitor(task.pid)

    result =
      await_with_eval_loop(task, eval_fun, device, task_ref, monitor_ref)

    Process.demonitor(monitor_ref, [:flush])
    result
  end

  defp await_with_eval_loop(task, eval_fun, device, task_ref, monitor_ref) do
    receive do
      {:eval_defn, callback_ref, refs, ^device, reply_to} ->
        # Convert refs back to tensors for evaluation on EMLX.Backend
        arg_list =
          Enum.map(refs, fn ref ->
            fn -> EMLX.Backend.to_nx({device, ref}) end
          end)

        # Evaluate in this process (keeps eval_fun here)
        # Nx.Defn.Evaluator.__compile__/4 returns a function expecting [params]
        result = eval_fun.([arg_list])

        # Extract the refs from the result tensors
        result_refs =
          result
          |> Nx.Defn.Composite.flatten_list()
          |> Enum.map(fn %Nx.Tensor{data: %{ref: {_device, ref}}} -> ref end)

        # Send result back to the callback
        send(reply_to, {:eval_result, callback_ref, result_refs})

        # Continue handling messages
        await_with_eval_loop(task, eval_fun, device, task_ref, monitor_ref)

      {:DOWN, ^monitor_ref, :process, _pid, :normal} ->
        # Task completed normally; keep waiting for its result message
        await_with_eval_loop(task, eval_fun, device, task_ref, monitor_ref)

      {:DOWN, ^monitor_ref, :process, _pid, reason} ->
        raise "Task failed: #{inspect(reason)}"

      {:EXIT, _pid, reason} ->
        raise "Task exited: #{inspect(reason)}"

      {^task_ref, result} ->
        # This is the Task reply - successful completion
        result
    after
      5_000 ->
        Task.shutdown(task, :brutal_kill)
        raise "Timeout waiting for compilation"
    end
  end

  defp nif_compile(nif_args, tag) do
    nif_args
    |> EMLX.NIF.compile(tag)
    |> unwrap!()
  end

  @impl Nx.Defn.Compiler
  defdelegate __partitions_options__(opts), to: Nx.Defn.Evaluator

  @impl Nx.Defn.Compiler
  def __to_backend__(opts) do
    device = Keyword.get(opts, :device, :gpu)
    {EMLX.Backend, device: device}
  end
end
