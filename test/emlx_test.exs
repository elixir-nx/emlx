defmodule EMLXTest do
  use EMLX.Case
  doctest EMLX

  test "__jit__" do
    {left, right} =
      Nx.Defn.jit_apply(&{Nx.add(&1, &2), Nx.subtract(&1, &2)}, [Nx.tensor(1), 2], compiler: EMLX)

    assert_equal(left, Nx.tensor(3))
    assert_equal(right, Nx.tensor(-1))
  end

  test "__jit__ supports binary backend in arguments" do
    {left, right} =
      Nx.Defn.jit_apply(
        &{Nx.add(&1, &2), Nx.subtract(&1, &2)},
        [Nx.tensor(1, backend: Nx.BinaryBackend), 2],
        compiler: EMLX
      )

    assert_equal(left, Nx.tensor(3))
    assert_equal(right, Nx.tensor(-1))
  end

  test "__jit__ supports binary backend as the default backend" do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      {left, right} =
        Nx.Defn.jit_apply(
          &{Nx.add(&1, &2), Nx.subtract(&1, &2)},
          [Nx.tensor(1), 2],
          compiler: EMLX
        )

      assert_equal(left, Nx.tensor(3))
      assert_equal(right, Nx.tensor(-1))
    end)
  end

  describe "scalar item extraction (MLX layout bug fix)" do
    # Tests for the fix in emlx_nif.cpp:item()
    # The bug: MLX creates scalars with invalid memory layout after slice→squeeze
    # The fix: Call item<T>() with the correct dtype instead of always using int64/double

    # Helper to call EMLX.item() directly (bypasses any Elixir workarounds)
    defp item_direct(tensor) do
      {_device, ref} = EMLX.Backend.from_nx(tensor)
      EMLX.item({:cpu, ref})
    end

    test "extracts int32 scalar from slice→squeeze" do
      array = Nx.iota({1000}, type: :s32)

      # Test various indices that previously failed
      for idx <- [0, 1, 100, 500, 900, 951, 998, 999] do
        sliced = Nx.slice_along_axis(array, idx, 1, axis: 0)
        scalar = Nx.squeeze(sliced, axes: [0])
        value = item_direct(scalar)

        assert value == idx, "Expected #{idx}, got #{value} for int32 scalar"
      end
    end

    test "extracts int8 scalar correctly" do
      array = Nx.iota({128}, type: :s8)

      for idx <- [0, 1, 50, 100, 127] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == idx
      end
    end

    test "extracts int16 scalar correctly" do
      array = Nx.iota({1000}, type: :s16)

      for idx <- [0, 1, 500, 999] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == idx
      end
    end

    test "extracts int64 scalar correctly" do
      array = Nx.iota({100}, type: :s64)

      for idx <- [0, 1, 50, 99] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == idx
      end
    end

    test "extracts uint8 scalar correctly" do
      array = Nx.iota({200}, type: :u8)

      for idx <- [0, 1, 100, 199] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == idx
      end
    end

    test "extracts uint16 scalar correctly" do
      array = Nx.iota({1000}, type: :u16)

      for idx <- [0, 1, 500, 999] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == idx
      end
    end

    test "extracts uint32 scalar correctly" do
      array = Nx.iota({1000}, type: :u32)

      for idx <- [0, 1, 500, 951, 999] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == idx
      end
    end

    test "extracts uint64 scalar correctly" do
      array = Nx.iota({100}, type: :u64)

      for idx <- [0, 1, 50, 99] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == idx
      end
    end

    test "extracts float32 scalar correctly" do
      array = Nx.iota({100}, type: :f32)

      for idx <- [0, 1, 50, 99] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert_in_delta item_direct(scalar), idx * 1.0, 1.0e-6
      end
    end

    test "extracts boolean scalar correctly" do
      # Create array [0, 1, 0, 1, ...] as uint8
      array = Nx.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], type: :u8)

      for idx <- [0, 1, 2, 3] do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        expected = rem(idx, 2)
        assert item_direct(scalar) == expected
      end
    end

    test "direct scalar creation works (baseline)" do
      # Ensure direct scalar creation still works
      scalar = Nx.tensor(951, type: :s32)
      assert item_direct(scalar) == 951
    end

    test "negative values work correctly" do
      array = Nx.tensor([-100, -50, 0, 50, 100], type: :s32)

      for {expected, idx} <- Enum.with_index([-100, -50, 0, 50, 100]) do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == expected
      end
    end

    test "edge values for int32" do
      # Test boundary values
      max_val = 2_147_483_647
      min_val = -2_147_483_648
      array = Nx.tensor([min_val, -1, 0, 1, max_val], type: :s32)

      for {expected, idx} <- Enum.with_index([min_val, -1, 0, 1, max_val]) do
        scalar = array |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
        assert item_direct(scalar) == expected
      end
    end
  end
end
