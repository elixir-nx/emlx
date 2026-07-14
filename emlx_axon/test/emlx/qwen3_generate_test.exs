defmodule EMLXAxon.Qwen3GenerateTest do
  use ExUnit.Case, async: true

  alias EMLXAxon.Qwen3.{Attention, DenseLoader, Generate, Model}

  test "greedy decode token id forward matches tensor token forward" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    kv_cache_for_tensor = Model.init_kv_cache(state, 8)
    kv_cache_for_id = Model.init_kv_cache(state, 8)
    input_ids = Nx.tensor([[1]], type: :s64, backend: {EMLX.Backend, device: :gpu})

    {token, _kv_cache} = Model.forward_greedy(input_ids, kv_cache_for_tensor, 0, state)
    {token_id, _kv_cache} = Model.forward_greedy_decode_token_id(1, kv_cache_for_id, 0, state)

    assert token_id == Nx.to_number(token)
  end

  test "native KV cache uses EMLX tensors for plugin generation" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [{k_cache, v_cache}] = Model.init_native_kv_cache(state, 8)
    assert %Nx.Tensor{data: %EMLX.Backend{ref: {:gpu, k_ref}}} = k_cache
    assert %Nx.Tensor{data: %EMLX.Backend{ref: {:gpu, v_ref}}} = v_cache
    assert is_reference(k_ref)
    assert is_reference(v_ref)

    {token_id, _kv_cache} =
      Model.forward_greedy_decode_token_id(1, [{k_cache, v_cache}], 0, state)

    assert is_integer(token_id)
  end

  test "dense attention forward accepts DenseLoader projection layout" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [
      {norm1, _norm2, q_norm, k_norm, q_proj, k_proj, v_proj, o_proj, _gate_proj, _up_proj,
       _down_proj}
    ] = state.layers

    hidden =
      Nx.broadcast(Nx.tensor(0.1, type: :f16), {1, 1, state.config.hidden_size})
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    [{k_cache, v_cache}] = Model.init_kv_cache(state, 8)

    assert {attn_out, k_cache_updated, v_cache_updated} =
             Attention.forward(
               hidden,
               norm1,
               k_cache,
               v_cache,
               0,
               q_proj,
               k_proj,
               v_proj,
               o_proj,
               q_norm,
               k_norm,
               state.config
             )

    assert Nx.shape(attn_out) == {1, 1, state.config.hidden_size}
    assert Nx.shape(k_cache_updated) == Nx.shape(k_cache)
    assert Nx.shape(v_cache_updated) == Nx.shape(v_cache)
  end

  test "end host sync returns the same greedy token ids as sync after each token" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {per_token, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 3,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    {end_sync, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 3,
        max_len: 8,
        sampler: :greedy,
        host_sync: :end,
        profile_timing: false
      )

    assert end_sync == per_token
  end

  test "dense end host sync batches the remaining tokens in one native chunk" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    state = %{state | config: Map.put(state.config, :eos_token_id, -1)}

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    parent = self()
    runner = fake_native_chunk_runner(parent, :dense)

    {tokens, metadata} =
      Generate.generate_with_native_chunk_runner(
        input_ids,
        state,
        [
          max_new_tokens: 4,
          max_len: 4,
          sampler: :greedy,
          host_sync: :end,
          profile_timing: false
        ],
        runner
      )

    assert length(tokens) == 4
    assert metadata.finish_reason == :length
    assert_receive {:native_chunk, :dense, 1, 3}
    refute_receive {:native_chunk, :dense, _, _}
  end

  test "end host sync subdivides more than 4096 generalized decode tokens" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    state = %{
      state
      | config:
          state.config
          |> Map.put(:dense_layers?, false)
          |> Map.put(:eos_token_id, -1)
    }

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    parent = self()
    runner = fake_native_chunk_runner(parent, :generalized)

    {tokens, metadata} =
      Generate.generate_with_native_chunk_runner(
        input_ids,
        state,
        [
          max_new_tokens: 4098,
          max_len: 4098,
          sampler: :greedy,
          host_sync: :end,
          profile_timing: false
        ],
        runner
      )

    assert length(tokens) == 4098
    assert metadata.finish_reason == :length
    assert_receive {:native_chunk, :generalized, 1, 4096}
    assert_receive {:native_chunk, :generalized, 4097, 1}
    refute_receive {:native_chunk, :generalized, _, _}
  end

  test "per-token and end host sync stop when any EOS token is emitted" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {[first_token], _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 1,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    state = %{state | config: %{state.config | eos_token_id: [first_token, 151_645]}}

    for host_sync <- [:per_token, :end] do
      {tokens, metadata} =
        Generate.generate(input_ids, state,
          max_new_tokens: 5,
          max_len: 8,
          sampler: :greedy,
          host_sync: host_sync,
          profile_timing: false
        )

      assert tokens == [first_token]
      assert metadata.finish_reason == :stop
    end
  end

  test "end host sync falls back to sync after each token when streaming callback is configured" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    parent = self()

    {tokens, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 2,
        max_len: 8,
        sampler: :greedy,
        host_sync: :end,
        profile_timing: false,
        token_callback: fn token_id -> send(parent, {:token, token_id}) end
      )

    assert_receive {:token, first_token}
    assert first_token == hd(tokens)
  end

  test "profile timing reports microsecond-resolution millisecond values" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    for host_sync <- [:per_token, :end, {:chunk, 2}] do
      {_tokens, %{timing: timing}} =
        Generate.generate(input_ids, state,
          max_new_tokens: 3,
          max_len: 8,
          sampler: :greedy,
          host_sync: host_sync,
          profile_timing: true
        )

      assert is_float(timing.prefill_ms)
      assert is_float(timing.total_ms)
      assert length(timing.per_token_ms) == 2
      assert Enum.all?(timing.per_token_ms, &is_float/1)
    end
  end

  test "chunked host sync returns the same greedy token ids as sync after each token" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {per_token, per_token_metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false,
        return_kv_cache: true
      )

    {chunked, chunked_metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 2},
        profile_timing: false,
        return_kv_cache: true
      )

    assert chunked == per_token

    Enum.zip(per_token_metadata.kv_cache, chunked_metadata.kv_cache)
    |> Enum.each(fn {{per_token_k, per_token_v}, {chunked_k, chunked_v}} ->
      assert_cache_close(per_token_k, chunked_k)
      assert_cache_close(per_token_v, chunked_v)
    end)
  end

  test "chunked host sync stops at the first EOS token" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {[first_token], _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 1,
        max_len: 8,
        sampler: :greedy,
        profile_timing: false
      )

    state = %{state | config: %{state.config | eos_token_id: first_token}}

    {tokens, metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 4},
        profile_timing: false
      )

    assert tokens == [first_token]
    assert metadata.finish_reason == :stop
  end

  test "dense native chunk supports token counts across the former plugin output limit" do
    for count <- [1022, 1023] do
      {:ok, state} =
        DenseLoader.from_model_info(%{
          params: %Axon.ModelState{data: params()},
          spec: spec()
        })

      input_ids =
        Nx.tensor([[1]], type: :s64)
        |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

      kv_cache = Model.init_native_kv_cache(state, count)
      {tokens, updated_cache} = Model.forward_greedy_chunk(input_ids, kv_cache, 0, count, state)

      assert Nx.shape(tokens) == {count}
      assert length(updated_cache) == 1
      assert Enum.all?(Nx.to_flat_list(tokens), &is_integer/1)
    end
  end

  test "dense native chunk output contract scales with layers rather than token count" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    state = %{state | layers: List.duplicate(hd(state.layers), 28)}

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {tokens, updated_cache} =
      Model.forward_greedy_chunk(
        input_ids,
        Model.init_native_kv_cache(state, 3),
        0,
        3,
        state
      )

    assert Nx.shape(tokens) == {3}
    assert length(updated_cache) == 28
  end

  test "generation rejects batched input_ids with end host sync" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2], [3, 4]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/batch size 1, got batch size 2/, fn ->
      Generate.generate(input_ids, state,
        max_new_tokens: 2,
        max_len: 8,
        sampler: :greedy,
        host_sync: :end,
        profile_timing: false
      )
    end
  end

  test "generation rejects batched input_ids with chunked host sync" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2], [3, 4]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/batch size 1, got batch size 2/, fn ->
      Generate.generate(input_ids, state,
        max_new_tokens: 2,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 2},
        profile_timing: false
      )
    end
  end

  test "chunked host sync emits host token chunks" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    parent = self()

    {tokens, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 2},
        profile_timing: false,
        chunk_callback: fn token_ids -> send(parent, {:chunk, token_ids}) end
      )

    assert_receive {:chunk, first_chunk}
    assert_receive {:chunk, second_chunk}
    assert_receive {:chunk, final_chunk}
    refute_receive {:chunk, _extra}

    assert [first_chunk, second_chunk, final_chunk] |> List.flatten() == tokens
    assert length(first_chunk) == 2
    assert length(second_chunk) == 2
    assert length(final_chunk) == 1
  end

  test "chunked host sync preserves a callback boundary larger than 4096" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    state = %{state | config: Map.put(state.config, :eos_token_id, -1)}

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    parent = self()
    runner = fake_native_chunk_runner(parent, :dense)

    {tokens, metadata} =
      Generate.generate_with_native_chunk_runner(
        input_ids,
        state,
        [
          max_new_tokens: 5000,
          max_len: 5000,
          sampler: :greedy,
          host_sync: {:chunk, 5000},
          profile_timing: false,
          chunk_callback: fn token_ids -> send(parent, {:chunk, token_ids}) end
        ],
        runner
      )

    assert length(tokens) == 5000
    assert metadata.finish_reason == :length
    assert_receive {:native_chunk, :dense, 1, 4096}
    assert_receive {:native_chunk, :dense, 4097, 903}
    assert_receive {:chunk, chunk}
    assert length(chunk) == 5000
    assert chunk == tokens
    refute_receive {:native_chunk, :dense, _, _}
    refute_receive {:chunk, _extra}
  end

  test "chunked host sync can emit the first token immediately" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {per_token, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    parent = self()

    {chunked, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 2},
        profile_timing: false,
        chunk_callback_first_token: true,
        chunk_callback: fn token_ids -> send(parent, {:chunk, token_ids}) end
      )

    assert_receive {:chunk, first_chunk}
    assert_receive {:chunk, second_chunk}
    assert_receive {:chunk, third_chunk}
    refute_receive {:chunk, _extra}

    assert length(first_chunk) == 1
    assert first_chunk == [hd(per_token)]
    assert [first_chunk, second_chunk, third_chunk] |> List.flatten() == chunked
    assert chunked == per_token
  end

  test "chunked host sync falls back to sync after each token when streaming callback is configured" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    parent = self()

    {tokens, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 2,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 2},
        profile_timing: false,
        token_callback: fn token_id -> send(parent, {:token, token_id}) end
      )

    assert_receive {:token, first_token}
    assert first_token == hd(tokens)
  end

  test "chunked host sync requires a positive integer chunk size" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/positive integer/, fn ->
      Generate.generate(input_ids, state,
        max_new_tokens: 2,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 0},
        profile_timing: false
      )
    end
  end

  test "generation caps max_new_tokens to KV cache capacity" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {tokens, metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 3,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    assert length(tokens) == 2
    assert metadata.finish_reason == :length
  end

  test "generation caps max_new_tokens to provided KV cache capacity" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    kv_cache = Model.init_kv_cache(state, 3)

    {tokens, metadata} =
      Generate.generate(input_ids, state,
        kv_cache: kv_cache,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    assert length(tokens) == 2
    assert metadata.finish_reason == :length
  end

  test "generation rejects inputs longer than KV cache capacity" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2, 3]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/input length/, fn ->
      Generate.generate(input_ids, state,
        max_new_tokens: 1,
        max_len: 2,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )
    end
  end

  test "native greedy forward rejects invalid layer tuples" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [layer] = state.layers
    bad_state = %{state | layers: [Tuple.delete_at(layer, 10)]}
    kv_cache = Model.init_native_kv_cache(bad_state, 8)

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/expected a Qwen3 layer tuple with 11 tensors/, fn ->
      Model.forward_greedy(input_ids, kv_cache, 0, bad_state)
    end
  end

  test "native greedy forward rejects mismatched layer and cache counts" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/layers and KV cache must have the same nonzero length/, fn ->
      Model.forward_greedy(input_ids, [], 0, state)
    end
  end

  test "native greedy forward rejects projection widths not divisible by head dim" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [layer] = state.layers

    bad_q_proj =
      Nx.iota({4, 5}, type: :f16)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    bad_state = %{state | layers: [put_elem(layer, 4, bad_q_proj)]}
    kv_cache = Model.init_native_kv_cache(bad_state, 8)

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise EMLX.NIFError, ~r/q_proj output width must be divisible by head_dim/, fn ->
      Model.forward_greedy(input_ids, kv_cache, 0, bad_state)
    end
  end

  test "native greedy forward rejects deallocated layer tensor refs" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [layer] = state.layers
    q_proj = elem(layer, 4)
    kv_cache = Model.init_native_kv_cache(state, 8)

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    Nx.backend_deallocate(q_proj)

    assert_raise EMLX.NIFError, ~r/Tensor has been deallocated/, fn ->
      Model.forward_greedy(input_ids, kv_cache, 0, state)
    end
  end

  test "native greedy forward rejects deallocated kv cache refs" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [{k_cache, _v_cache}] = kv_cache = Model.init_native_kv_cache(state, 8)

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    Nx.backend_deallocate(k_cache)

    assert_raise EMLX.NIFError, ~r/Tensor has been deallocated/, fn ->
      Model.forward_greedy(input_ids, kv_cache, 0, state)
    end
  end

  test "native greedy forward rejects deallocated final norm refs" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    kv_cache = Model.init_native_kv_cache(state, 8)

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    Nx.backend_deallocate(state.norm)

    assert_raise EMLX.NIFError, ~r/Tensor has been deallocated/, fn ->
      Model.forward_greedy(input_ids, kv_cache, 0, state)
    end
  end

  test "native greedy forward rejects cache capacity overflow" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    kv_cache = Model.init_native_kv_cache(state, 2)

    assert_raise EMLX.NIFError, ~r/KV cache capacity 2 is smaller than required length 3/, fn ->
      Model.forward_greedy(input_ids, kv_cache, 2, state)
    end
  end

  test "native chunk forward rejects batched inputs before plugin dispatch" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1], [2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/must have shape \{1, 1\}/, fn ->
      Model.forward_greedy_chunk(input_ids, [], 0, 1, %{state | layers: []})
    end
  end

  test "a direct native chunk call rejects counts above the plugin limit" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/supports at most 4096 tokens/, fn ->
      Model.forward_greedy_chunk(input_ids, [], 0, 4097, state)
    end
  end

  test "generation rejects unsupported host sync modes" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError, ~r/expected :host_sync/, fn ->
      Generate.generate(input_ids, state,
        max_new_tokens: 2,
        max_len: 8,
        sampler: :greedy,
        host_sync: :sometimes,
        profile_timing: false
      )
    end
  end

  test "generation rejects zero, negative, string, or decimal max_new_tokens" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    for max_new_tokens <- [0, -1, 1.5, "1"] do
      assert_raise ArgumentError, ~r/expected :max_new_tokens to be a positive integer/, fn ->
        Generate.generate(input_ids, state,
          max_new_tokens: max_new_tokens,
          max_len: 8,
          sampler: :greedy,
          profile_timing: false
        )
      end
    end
  end

  defp spec do
    %Bumblebee.Text.Qwen3{
      architecture: :for_causal_language_modeling,
      vocab_size: 16,
      hidden_size: 4,
      intermediate_size: 8,
      attention_head_size: 2,
      num_blocks: 1,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      rotary_embedding_base: 10_000,
      layer_norm_epsilon: 1.0e-6,
      tie_word_embeddings: true
    }
  end

  defp params do
    %{}
    |> put("embedder.token_embedding", "kernel", Nx.iota({16, 4}, type: :f16))
    |> put("language_modeling_head.output", "kernel", Nx.iota({16, 4}, type: :f16))
    |> put("output_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.self_attention_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.output_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.self_attention.query_norm", "weight", ones({2}))
    |> put("decoder.blocks.0.self_attention.key_norm", "weight", ones({2}))
    |> put("decoder.blocks.0.self_attention.query", "kernel", Nx.iota({4, 4}, type: :f16))
    |> put("decoder.blocks.0.self_attention.key", "kernel", Nx.iota({4, 2}, type: :f16))
    |> put("decoder.blocks.0.self_attention.value", "kernel", Nx.iota({4, 2}, type: :f16))
    |> put("decoder.blocks.0.self_attention.output", "kernel", Nx.iota({4, 4}, type: :f16))
    |> put("decoder.blocks.0.ffn.gate", "kernel", Nx.iota({4, 8}, type: :f16))
    |> put("decoder.blocks.0.ffn.intermediate", "kernel", Nx.iota({4, 8}, type: :f16))
    |> put("decoder.blocks.0.ffn.output", "kernel", Nx.iota({8, 4}, type: :f16))
  end

  defp ones(shape), do: Nx.broadcast(Nx.tensor(1.0, type: :f16), shape)

  defp fake_native_chunk_runner(parent, kind) do
    fn input_ids, kv_cache, offset, count, _state ->
      send(parent, {:native_chunk, kind, offset, count})
      token = Nx.squeeze(input_ids)
      {Nx.broadcast(token, {count}), kv_cache}
    end
  end

  defp assert_cache_close(left, right) do
    left = Nx.backend_transfer(left, Nx.BinaryBackend)
    right = Nx.backend_transfer(right, Nx.BinaryBackend)

    assert Nx.shape(left) == Nx.shape(right)
    assert Nx.all_close(left, right, atol: 1.0e-3, rtol: 1.0e-3) |> Nx.to_number() == 1
  end

  defp put(params, scope, name, tensor) do
    Map.update(params, scope, %{name => tensor}, &Map.put(&1, name, tensor))
  end
end
