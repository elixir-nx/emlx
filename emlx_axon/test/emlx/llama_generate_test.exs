defmodule EMLXAxon.LlamaGenerateTest do
  use ExUnit.Case, async: true

  alias EMLXAxon.Llama.{DenseLoader, Generate, Model, Sampler}

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

  test "native KV cache uses raw refs for greedy dense generation" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [{k_cache, v_cache}] = Model.init_native_kv_cache(state, 8)
    assert {:gpu, k_ref} = k_cache
    assert {:gpu, v_ref} = v_cache
    assert is_reference(k_ref)
    assert is_reference(v_ref)

    {token_id, _kv_cache} =
      Model.forward_greedy_decode_token_id(1, [{k_cache, v_cache}], 0, state)

    assert is_integer(token_id)
  end

  test "KV cache dtype follows dense state dtype from Bumblebee params" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params(:f32)},
        spec: spec()
      })

    [{k_cache, v_cache}] = Model.init_kv_cache(state, 8)
    [{k_ref, v_ref}] = Model.init_native_kv_cache(state, 8)

    assert Nx.type(state.embed_tokens) == {:f, 32}
    assert Nx.type(k_cache) == {:f, 32}
    assert Nx.type(v_cache) == {:f, 32}
    assert EMLX.scalar_type(k_ref) == :float32
    assert EMLX.scalar_type(v_ref) == :float32
  end

  test "dense transformer layer accepts DenseLoader projection layout" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [layer] = state.layers

    hidden =
      Nx.broadcast(Nx.tensor(0.1, type: :f16), {1, 1, state.config.hidden_size})
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    [{k_cache, v_cache}] = Model.init_kv_cache(state, 8)

    assert {hidden_out, k_cache_updated, v_cache_updated} =
             Model.transformer_layer(hidden, k_cache, v_cache, 0, layer, state.config)

    assert Nx.shape(hidden_out) == {1, 1, state.config.hidden_size}
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

  test "native greedy wrappers match tensor-step greedy generation" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    tensor_step_state = tensor_step_state(state)

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    for host_sync <- [:per_token, :end, {:chunk, 2}] do
      {native_tokens, native_metadata} =
        Generate.generate(input_ids, state,
          max_new_tokens: 5,
          max_len: 8,
          sampler: :greedy,
          host_sync: host_sync,
          profile_timing: false
        )

      {tensor_step_tokens, tensor_step_metadata} =
        Generate.generate(input_ids, tensor_step_state,
          max_new_tokens: 5,
          max_len: 8,
          sampler: :greedy,
          host_sync: host_sync,
          profile_timing: false
        )

      assert native_tokens == tensor_step_tokens
      assert native_metadata.finish_reason == tensor_step_metadata.finish_reason
    end
  end

  test "native greedy generation is stable across repeated runs" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {expected_tokens, expected_metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 4,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 2},
        profile_timing: false
      )

    for _run <- 1..25 do
      {tokens, metadata} =
        Generate.generate(input_ids, state,
          max_new_tokens: 4,
          max_len: 8,
          sampler: :greedy,
          host_sync: {:chunk, 2},
          profile_timing: false
        )

      assert tokens == expected_tokens
      assert metadata.finish_reason == expected_metadata.finish_reason
    end
  end

  test "generation stops when any token in eos_token_id list is emitted" do
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

    state = %{state | config: %{state.config | eos_token_id: [first_token, 2]}}

    {tokens, metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    assert tokens == [first_token]
    assert metadata.finish_reason == :stop
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

    {per_token, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    {chunked, _metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: {:chunk, 2},
        profile_timing: false
      )

    assert chunked == per_token
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

  test "generation rejects invalid max_len before cache allocation" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    for max_len <- [0, -1, 1.5, "8"] do
      assert_raise ArgumentError, ~r/expected :max_len to be a positive integer/, fn ->
        Generate.generate(input_ids, state,
          max_new_tokens: 1,
          max_len: max_len,
          sampler: :greedy,
          host_sync: :per_token,
          profile_timing: false
        )
      end
    end
  end

  test "native greedy forward rejects invalid layer tuples" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    [layer] = state.layers
    bad_state = %{state | layers: [Tuple.delete_at(layer, 8)]}
    kv_cache = Model.init_native_kv_cache(bad_state, 8)

    input_ids =
      Nx.tensor([[1]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise FunctionClauseError, fn ->
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

    assert_raise EMLX.NIFError, ~r/layers and kv_cache length mismatch/, fn ->
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
    bad_q_proj = Nx.iota({4, 5}, type: :f16)
    bad_state = %{state | layers: [put_elem(layer, 2, bad_q_proj)]}
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
    q_proj = elem(layer, 2)
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

    EMLX.deallocate(k_cache)

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

  test "native token id return path rejects batched inputs" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1], [2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    assert_raise ArgumentError,
                 ~r/llama_forward_greedy_ids_token_id requires batch size 1, got batch size 2/,
                 fn ->
                   EMLX.Native.Llama.forward_greedy_ids_token_id(
                     EMLX.Backend.from_nx(input_ids),
                     EMLX.Backend.from_nx(state.embed_tokens),
                     [],
                     [],
                     EMLX.Backend.from_nx(state.norm),
                     EMLX.Backend.from_nx(state.lm_head),
                     0,
                     1.0 / :math.sqrt(state.config.head_dim),
                     state.config.head_dim,
                     EMLX.Backend.from_nx(state.rope_freqs),
                     state.config.rms_norm_eps
                   )
                 end
  end

  test "native chunk token id return path rejects batched inputs" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1], [2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    embed_ref = EMLX.Backend.from_nx(state.embed_tokens)

    assert_raise ArgumentError, ~r/llama_forward_greedy_ids_chunk requires batch size 1/, fn ->
      EMLX.Native.Llama.forward_greedy_ids_chunk(
        EMLX.Backend.from_nx(input_ids),
        embed_ref,
        [],
        [],
        EMLX.Backend.from_nx(state.norm),
        EMLX.Backend.from_nx(state.lm_head),
        0,
        1,
        1.0 / :math.sqrt(state.config.head_dim),
        state.config.head_dim,
        EMLX.Backend.from_nx(state.rope_freqs),
        state.config.rms_norm_eps
      )
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

  test "generation rejects invalid sampler options before decode" do
    {:ok, state} =
      DenseLoader.from_model_info(%{
        params: %Axon.ModelState{data: params()},
        spec: spec()
      })

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    cases = [
      {[sampler: :sometimes], ~r/expected :sampler/},
      {[sampler: :top_p_cpu, temperature: 0.0], ~r/expected :temperature/},
      {[sampler: :top_p_cpu, temperature: -1.0], ~r/expected :temperature/},
      {[sampler: :top_p_cpu, temperature: :invalid], ~r/expected :temperature/},
      {[sampler: :top_p_cpu, top_p: 0.0], ~r/expected :top_p/},
      {[sampler: :top_p_cpu, top_p: 1.5], ~r/expected :top_p/},
      {[sampler: :top_p_cpu, top_p: :invalid], ~r/expected :top_p/},
      {[sampler: :top_p_gpu, temperature: 0.0], ~r/expected :temperature/}
    ]

    for {opts, message} <- cases do
      assert_raise ArgumentError, message, fn ->
        Generate.generate(
          input_ids,
          state,
          Keyword.merge(
            [
              max_new_tokens: 1,
              max_len: 8,
              profile_timing: false
            ],
            opts
          )
        )
      end
    end
  end

  test "samplers reject invalid temperature and top_p values" do
    logits = Nx.tensor([[10.0, 1.0, 0.0]], type: :f32)

    assert_raise ArgumentError, ~r/expected temperature/, fn ->
      Sampler.top_p_cpu(logits, 0.0, 0.9)
    end

    assert_raise ArgumentError, ~r/expected top_p/, fn ->
      Sampler.top_p_cpu(logits, 1.0, 0.0)
    end

    assert_raise ArgumentError, ~r/expected temperature/, fn ->
      Sampler.top_p_gpu(logits, Nx.Random.key(42), temperature: -1.0)
    end
  end

  defp spec do
    %Bumblebee.Text.Llama{
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

  defp params(type \\ :f16) do
    %{}
    |> put("embedder.token_embedding", "kernel", Nx.iota({16, 4}, type: type))
    |> put("language_modeling_head.output", "kernel", Nx.iota({16, 4}, type: type))
    |> put("output_norm", "weight", ones({4}, type))
    |> put("decoder.blocks.0.self_attention_norm", "weight", ones({4}, type))
    |> put("decoder.blocks.0.output_norm", "weight", ones({4}, type))
    |> put("decoder.blocks.0.self_attention.query", "kernel", Nx.iota({4, 4}, type: type))
    |> put("decoder.blocks.0.self_attention.key", "kernel", Nx.iota({4, 2}, type: type))
    |> put("decoder.blocks.0.self_attention.value", "kernel", Nx.iota({4, 2}, type: type))
    |> put("decoder.blocks.0.self_attention.output", "kernel", Nx.iota({4, 4}, type: type))
    |> put("decoder.blocks.0.ffn.gate", "kernel", Nx.iota({4, 8}, type: type))
    |> put("decoder.blocks.0.ffn.intermediate", "kernel", Nx.iota({4, 8}, type: type))
    |> put("decoder.blocks.0.ffn.output", "kernel", Nx.iota({8, 4}, type: type))
  end

  defp ones(shape, type), do: Nx.broadcast(Nx.tensor(1.0, type: type), shape)

  defp tensor_step_state(state), do: %{state | config: %{state.config | dense_layers?: false}}

  defp put(params, scope, name, tensor) do
    Map.update(params, scope, %{name => tensor}, &Map.put(&1, name, tensor))
  end
end
