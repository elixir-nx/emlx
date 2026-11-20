defmodule EMLX.ToDeviceTest do
  use EMLX.Case, async: false

  describe "CPU → CPU" do
    test "preserves values" do
      t = Nx.iota({3, 4}, type: :f32, backend: {EMLX.Backend, device: :cpu})
      result_ref = EMLX.to_device(EMLX.Backend.from_nx(t), :cpu)

      assert EMLX.to_blob(result_ref) ==
               Nx.to_binary(Nx.iota({3, 4}, type: :f32, backend: Nx.BinaryBackend))
    end

    test "result is tagged :cpu" do
      t = Nx.iota({4}, type: :f32, backend: {EMLX.Backend, device: :cpu})
      result_ref = EMLX.to_device(EMLX.Backend.from_nx(t), :cpu)
      assert elem(result_ref, 0) == :cpu
    end

    test "source tensor remains accessible after to_device" do
      t = Nx.iota({4}, type: :f32, backend: {EMLX.Backend, device: :cpu})
      ref = EMLX.Backend.from_nx(t)
      _result = EMLX.to_device(ref, :cpu)
      # Source must not be deallocated — to_blob would raise on deallocation
      assert is_binary(EMLX.to_blob(ref))
    end

    test "works on a non-contiguous (transposed) tensor" do
      t = Nx.iota({3, 4}, type: :f32, backend: {EMLX.Backend, device: :cpu}) |> Nx.transpose()
      ref = EMLX.Backend.from_nx(t)
      result_ref = EMLX.to_device(ref, :cpu)
      assert elem(result_ref, 0) == :cpu
      assert is_binary(EMLX.to_blob(result_ref))
    end
  end

  describe "CPU → GPU" do
    @tag :metal
    test "result is tagged :gpu" do
      t = Nx.iota({4}, type: :f32, backend: {EMLX.Backend, device: :cpu})
      result_ref = EMLX.to_device(EMLX.Backend.from_nx(t), :gpu)
      assert elem(result_ref, 0) == :gpu
    end

    @tag :metal
    test "values are preserved" do
      t = Nx.iota({3, 4}, type: :f32, backend: {EMLX.Backend, device: :cpu})
      result_ref = EMLX.to_device(EMLX.Backend.from_nx(t), :gpu)
      binary = EMLX.to_blob(result_ref)
      expected = Nx.to_binary(Nx.iota({3, 4}, type: :f32, backend: Nx.BinaryBackend))
      assert binary == expected
    end

    @tag :metal
    test "source tensor remains accessible after transfer" do
      t = Nx.iota({4}, type: :f32, backend: {EMLX.Backend, device: :cpu})
      ref = EMLX.Backend.from_nx(t)
      _result_ref = EMLX.to_device(ref, :gpu)
      assert is_binary(EMLX.to_blob(ref))
    end
  end

  describe "GPU → CPU" do
    @tag :metal
    test "result is tagged :cpu" do
      t = Nx.iota({4}, type: :f32, backend: {EMLX.Backend, device: :gpu})
      result_ref = EMLX.to_device(EMLX.Backend.from_nx(t), :cpu)
      assert elem(result_ref, 0) == :cpu
    end

    @tag :metal
    test "values are preserved" do
      t = Nx.iota({3, 4}, type: :f32, backend: {EMLX.Backend, device: :gpu})
      result_ref = EMLX.to_device(EMLX.Backend.from_nx(t), :cpu)
      binary = EMLX.to_blob(result_ref)
      expected = Nx.to_binary(Nx.iota({3, 4}, type: :f32, backend: Nx.BinaryBackend))
      assert binary == expected
    end

    @tag :metal
    test "source tensor remains accessible after transfer" do
      t = Nx.iota({4}, type: :f32, backend: {EMLX.Backend, device: :gpu})
      ref = EMLX.Backend.from_nx(t)
      _result_ref = EMLX.to_device(ref, :cpu)
      assert is_binary(EMLX.to_blob(ref))
    end
  end

  describe "GPU → GPU" do
    @tag :metal
    test "result is tagged :gpu" do
      t = Nx.iota({4}, type: :f32, backend: {EMLX.Backend, device: :gpu})
      result_ref = EMLX.to_device(EMLX.Backend.from_nx(t), :gpu)
      assert elem(result_ref, 0) == :gpu
    end

    @tag :metal
    test "values are preserved" do
      t = Nx.iota({3, 4}, type: :f32, backend: {EMLX.Backend, device: :gpu})
      result_ref = EMLX.to_device(EMLX.Backend.from_nx(t), :gpu)
      binary = EMLX.to_blob(result_ref)
      expected = Nx.to_binary(Nx.iota({3, 4}, type: :f32, backend: Nx.BinaryBackend))
      assert binary == expected
    end
  end

  # An op moves all its operands to one device first. A worker cannot evaluate a
  # lazy op on the stream of another worker, so an operand that stays on the
  # other device fails with "There is no Stream(...) in current thread".
  describe "ops with operands on both devices" do
    @describetag :metal

    test "gather with GPU indices into a CPU tensor" do
      t = Nx.iota({3, 3}, type: :f32, backend: {EMLX.Backend, device: :cpu})
      indices = Nx.tensor([[0, 0], [1, 1], [2, 2]], backend: {EMLX.Backend, device: :gpu})

      assert Nx.to_flat_list(Nx.gather(t, indices)) == [0.0, 4.0, 8.0]
    end

    test "indexed_add and indexed_put with GPU indices into a CPU tensor" do
      t = Nx.broadcast(Nx.tensor(1.0, backend: {EMLX.Backend, device: :cpu}), {3})
      indices = Nx.tensor([[0], [2]], backend: {EMLX.Backend, device: :gpu})
      updates = Nx.tensor([10.0, 20.0], backend: {EMLX.Backend, device: :cpu})

      assert Nx.to_flat_list(Nx.indexed_add(t, indices, updates)) == [11.0, 1.0, 21.0]
      assert Nx.to_flat_list(Nx.indexed_put(t, indices, updates)) == [10.0, 1.0, 20.0]
    end

    test "quantized_matmul with a GPU input, CPU weights and no biases" do
      weight = Nx.iota({64, 64}, type: :f32, backend: {EMLX.Backend, device: :cpu})

      {w, scales, nil} =
        EMLX.quantize(EMLX.Backend.from_nx(Nx.divide(weight, 100)), 32, 4, "mxfp4")

      x = Nx.iota({2, 64}, type: :f32) |> Nx.divide(64)

      matmul = fn device ->
        x
        |> Nx.backend_copy({EMLX.Backend, device: device})
        |> EMLX.Backend.from_nx()
        |> EMLX.quantized_matmul(w, scales, nil, true, 32, 4, "mxfp4")
        |> EMLX.to_blob()
      end

      assert matmul.(:gpu) == matmul.(:cpu)
    end
  end
end
