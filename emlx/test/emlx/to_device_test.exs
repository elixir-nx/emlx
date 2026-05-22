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
end
