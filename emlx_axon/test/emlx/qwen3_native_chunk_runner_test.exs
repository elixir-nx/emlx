defmodule EMLXAxon.Qwen3NativeChunkRunnerTest do
  use ExUnit.Case, async: true

  alias EMLXAxon.Qwen3.NativeChunkRunner

  test "keeps an exact 4096-token request in one native call" do
    parent = self()

    forward = fn last_token, calls, offset, count ->
      send(parent, {:native_call, last_token, offset, count})
      tokens = Enum.to_list((offset + 1)..(offset + count))
      {tokens, [{offset, count} | calls]}
    end

    assert {tokens, calls, 4096} = NativeChunkRunner.run(0, [], 0, 4096, forward)
    assert tokens == Enum.to_list(1..4096)
    assert calls == [{0, 4096}]
    assert_receive {:native_call, 0, 0, 4096}
    refute_receive {:native_call, _, _, _}
  end

  test "splits 4097 tokens without changing order or KV offsets" do
    parent = self()

    forward = fn last_token, calls, offset, count ->
      send(parent, {:native_call, last_token, offset, count})
      tokens = Enum.to_list((offset + 1)..(offset + count))
      {tokens, calls ++ [{offset, count}]}
    end

    assert {tokens, calls, 4107} = NativeChunkRunner.run(10, [], 10, 4097, forward)
    assert tokens == Enum.to_list(11..4107)
    assert calls == [{10, 4096}, {4106, 1}]
    assert_receive {:native_call, 10, 10, 4096}
    assert_receive {:native_call, 4106, 4106, 1}
    refute_receive {:native_call, _, _, _}
  end

  test "applies the same subdivision before dense and generalized dispatch" do
    for kind <- [:dense, :generalized] do
      forward = fn last_token, calls, offset, count ->
        tokens = List.duplicate(last_token + 1, count)
        {tokens, calls ++ [{kind, offset, count}]}
      end

      assert {tokens, calls, 4097} = NativeChunkRunner.run(0, [], 0, 4097, forward)
      assert length(tokens) == 4097
      assert calls == [{kind, 0, 4096}, {kind, 4096, 1}]
    end
  end

  test "preserves small chunks and rejects invalid counts" do
    forward = fn last_token, kv_cache, offset, count ->
      {Enum.to_list((last_token + 1)..(last_token + count)), [{offset, count} | kv_cache]}
    end

    assert {[1, 2, 3], [{0, 3}], 3} = NativeChunkRunner.run(0, [], 0, 3, forward)

    for count <- [0, -1, 1.5, "1"] do
      assert_raise ArgumentError, ~r/positive integer/, fn ->
        NativeChunkRunner.run(0, [], 0, count, forward)
      end
    end
  end

  test "rejects malformed native results before advancing state" do
    assert_raise ArgumentError, ~r/returned 1 tokens, expected 2/, fn ->
      NativeChunkRunner.run(0, :kv, 0, 2, fn _last, kv, _offset, _count -> {[1], kv} end)
    end

    assert_raise ArgumentError, ~r/returned a non-list value/, fn ->
      NativeChunkRunner.run(0, :kv, 0, 1, fn _last, kv, _offset, _count -> {:bad, kv} end)
    end
  end
end
