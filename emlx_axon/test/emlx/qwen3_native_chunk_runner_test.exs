defmodule EMLXAxon.Qwen3NativeChunkRunnerTest do
  use ExUnit.Case, async: true

  alias EMLXAxon.Qwen3.NativeChunkRunner

  test "keeps an exact 4096-token request in one native call" do
    parent = self()

    forward = fn last_token, calls, offset, count ->
      send(parent, {:native_call, last_token, offset, count})
      tokens = Nx.add(Nx.iota({count}), offset + 1)
      {tokens, [{offset, count} | calls]}
    end

    assert {[tokens], last_token, calls, 4096} =
             NativeChunkRunner.run(Nx.tensor(0), [], 0, 4096, forward)

    assert Nx.to_flat_list(tokens) == Enum.to_list(1..4096)
    assert Nx.to_number(last_token) == 4096
    assert calls == [{0, 4096}]
    assert_receive {:native_call, initial_token, 0, 4096}
    assert Nx.to_number(initial_token) == 0
    refute_receive {:native_call, _, _, _}
  end

  test "splits 4097 tokens without changing order or KV offsets" do
    parent = self()

    forward = fn last_token, calls, offset, count ->
      send(parent, {:native_call, last_token, offset, count})
      tokens = Nx.add(Nx.iota({count}), offset + 1)
      {tokens, calls ++ [{offset, count}]}
    end

    assert {chunks, last_token, calls, 4107} =
             NativeChunkRunner.run(Nx.tensor(10), [], 10, 4097, forward)

    assert Enum.flat_map(chunks, &Nx.to_flat_list/1) == Enum.to_list(11..4107)
    assert Nx.to_number(last_token) == 4107
    assert calls == [{10, 4096}, {4106, 1}]
    assert_receive {:native_call, first_token, 10, 4096}
    assert_receive {:native_call, second_token, 4106, 1}
    assert Nx.to_number(first_token) == 10
    assert Nx.to_number(second_token) == 4106
    refute_receive {:native_call, _, _, _}
  end

  test "applies the same subdivision before dense and generalized dispatch" do
    for kind <- [:dense, :generalized] do
      forward = fn last_token, calls, offset, count ->
        tokens = Nx.broadcast(Nx.add(last_token, 1), {count})
        {tokens, calls ++ [{kind, offset, count}]}
      end

      assert {chunks, last_token, calls, 4097} =
               NativeChunkRunner.run(Nx.tensor(0), [], 0, 4097, forward)

      assert Enum.map(chunks, &Nx.shape/1) == [{4096}, {1}]
      assert Nx.to_number(last_token) == 2
      assert calls == [{kind, 0, 4096}, {kind, 4096, 1}]
    end
  end

  test "preserves small chunks and rejects invalid counts" do
    forward = fn last_token, kv_cache, offset, count ->
      {Nx.add(Nx.iota({count}), Nx.add(last_token, 1)), [{offset, count} | kv_cache]}
    end

    assert {[tokens], last_token, [{0, 3}], 3} =
             NativeChunkRunner.run(Nx.tensor(0), [], 0, 3, forward)

    assert Nx.to_flat_list(tokens) == [1, 2, 3]
    assert Nx.to_number(last_token) == 3

    for count <- [0, -1, 1.5, "1"] do
      assert_raise ArgumentError, ~r/positive integer/, fn ->
        NativeChunkRunner.run(0, [], 0, count, forward)
      end
    end
  end

  test "rejects malformed native results before advancing state" do
    assert_raise ArgumentError, ~r/returned tensor shape \{1\} tokens, expected 2/, fn ->
      NativeChunkRunner.run(Nx.tensor(0), :kv, 0, 2, fn _last, kv, _offset, _count ->
        {Nx.tensor([1]), kv}
      end)
    end

    assert_raise ArgumentError, ~r/returned a non-tensor value/, fn ->
      NativeChunkRunner.run(Nx.tensor(0), :kv, 0, 1, fn _last, kv, _offset, _count ->
        {:bad, kv}
      end)
    end
  end
end
