defmodule EMLX.CTest do
  @moduledoc """
  Differential tests: every op in the mlx-c lane (`EMLX.C.NIF`) must return
  bit-identical results to the direct-C++ reference lane (`EMLX.C.RefNIF`)
  for the same inputs. Both lanes link their own libmlx and are built with
  `EMLX_MLXC=true mix compile`; the suite self-excludes otherwise.
  """
  use ExUnit.Case, async: true

  @moduletag :mlx_c

  defp f32(n) do
    for _ <- 1..n, into: <<>>, do: <<:rand.uniform() - 0.5::float-32-little>>
  end

  defp f16(n) do
    for _ <- 1..n, into: <<>> do
      sign = :rand.uniform(2) - 1
      exp = 15 + :rand.uniform(5) - 3
      mant = :rand.uniform(1024) - 1
      <<sign * 32768 + Bitwise.bsl(exp, 10) + mant::16-little>>
    end
  end

  defp devices do
    case EMLX.C.NIF.device_check() do
      {:ok, gpu?, _ver} -> if gpu? == :gpu, do: [:gpu, :cpu], else: [:cpu]
      _ -> [:cpu]
    end
  end

  describe "binary I/O" do
    test "copy round-trips identically on all devices" do
      for dev <- devices() do
        bin = f32(1024)
        assert {:ok, out_cpp} = EMLX.C.RefNIF.copy(bin, dev, 1024, :float32)
        assert {:ok, out_c} = EMLX.C.NIF.copy(bin, dev, 1024, :float32)
        assert out_cpp == bin
        assert out_c == bin
      end
    end
  end

  describe "elementwise" do
    test "add/subtract/multiply/divide match across lanes" do
      for dev <- devices() do
        {m, n} = {33, 100}

        for op <- [:add, :subtract, :multiply, :divide] do
          a = f32(m * n)
          b = f32(m * n)
          {:ok, out_cpp} = apply(EMLX.C.RefNIF, op, [a, b, :float32, dev, m, n])
          {:ok, out_c} = apply(EMLX.C.NIF, op, [a, b, :float32, dev, m, n])
          assert out_cpp == out_c, "#{op}/#{dev} differs between lanes"
        end
      end
    end

    test "exp matches across lanes" do
      for dev <- devices() do
        a = f32(1000)
        assert {:ok, out_cpp} = EMLX.C.RefNIF.exp(a, :float32, dev, 10, 100)
        assert {:ok, out_c} = EMLX.C.NIF.exp(a, :float32, dev, 10, 100)
        assert out_cpp == out_c
      end
    end
  end

  describe "reductions and transforms" do
    test "sum over all axes matches across lanes" do
      for dev <- devices() do
        a = f32(128 * 128)
        assert {:ok, out_cpp} = EMLX.C.RefNIF.sum(a, :float32, dev, 128, 128)
        assert {:ok, out_c} = EMLX.C.NIF.sum(a, :float32, dev, 128, 128)
        assert out_cpp == out_c
      end
    end

    test "softmax_axis matches across lanes (fp32 and fp16)" do
      for dev <- devices() do
        a = f32(16 * 64)
        assert {:ok, out_cpp} = EMLX.C.RefNIF.softmax_axis(a, :float32, dev, 16, 64, 1)
        assert {:ok, out_c} = EMLX.C.NIF.softmax_axis(a, :float32, dev, 16, 64, 1)
        assert out_cpp == out_c

        a16 = f16(16 * 64)
        assert {:ok, out_cpp} = EMLX.C.RefNIF.softmax_axis(a16, :float16, dev, 16, 64, 1)
        assert {:ok, out_c} = EMLX.C.NIF.softmax_axis(a16, :float16, dev, 16, 64, 1)
        assert out_cpp == out_c
      end
    end

    test "transpose matches across lanes" do
      for dev <- devices() do
        a = f32(16 * 64)
        assert {:ok, out_cpp} = EMLX.C.RefNIF.transpose(a, :float32, dev, 16, 64)
        assert {:ok, out_c} = EMLX.C.NIF.transpose(a, :float32, dev, 16, 64)
        assert out_cpp == out_c
      end
    end

    test "astype matches across lanes" do
      for dev <- devices() do
        a = f32(1000)
        assert {:ok, out_cpp} = EMLX.C.RefNIF.astype(a, :float32, :float16, dev, 1000)
        assert {:ok, out_c} = EMLX.C.NIF.astype(a, :float32, :float16, dev, 1000)
        assert out_cpp == out_c
      end
    end

    test "reshape is a metadata-only identity on row-major data" do
      for dev <- devices() do
        bin = f32(24)
        assert {:ok, out_cpp} = EMLX.C.RefNIF.reshape(bin, :float32, dev, [3, 8])
        assert {:ok, out_c} = EMLX.C.NIF.reshape(bin, :float32, dev, [3, 8])
        assert out_cpp == bin
        assert out_c == bin
      end
    end
  end

  describe "matmul family" do
    test "matmul fp32/fp16 matches across lanes" do
      for dev <- devices() do
        {m, k, n} = {64, 64, 64}
        a = f32(m * k)
        b = f32(k * n)
        {:ok, out_cpp} = EMLX.C.RefNIF.matmul(a, b, :float32, dev, m, k, n)
        {:ok, out_c} = EMLX.C.NIF.matmul(a, b, :float32, dev, m, k, n)
        assert out_cpp == out_c

        a16 = f16(m * k)
        b16 = f16(k * n)
        {:ok, out_cpp} = EMLX.C.RefNIF.matmul(a16, b16, :float16, dev, m, k, n)
        {:ok, out_c} = EMLX.C.NIF.matmul(a16, b16, :float16, dev, m, k, n)
        assert out_cpp == out_c
      end
    end

    test "bmm fp16 matches across lanes" do
      for dev <- devices() do
        a = f16(4 * 32 * 32)
        b = f16(4 * 32 * 32)
        {:ok, out_cpp} = EMLX.C.RefNIF.bmm(a, b, :float16, dev, 4, 32, 32, 32)
        {:ok, out_c} = EMLX.C.NIF.bmm(a, b, :float16, dev, 4, 32, 32, 32)
        assert out_cpp == out_c
      end
    end

    test "sdpa fp16 matches across lanes" do
      for dev <- devices() do
        n = 2 * 4 * 32 * 16
        q = f16(n)
        k = f16(n)
        v = f16(n)
        {:ok, out_cpp} = EMLX.C.RefNIF.attn(q, k, v, 0.15, dev, 2, 4, 32, 16)
        {:ok, out_c} = EMLX.C.NIF.attn(q, k, v, 0.15, dev, 2, 4, 32, 16)
        assert out_cpp == out_c
      end
    end
  end

  describe "linalg" do
    test "svd singular values match across lanes (CPU)" do
      a = f32(64 * 64)
      assert {:ok, out_cpp} = EMLX.C.RefNIF.svd(a, :cpu, 64)
      assert {:ok, out_c} = EMLX.C.NIF.svd(a, :cpu, 64)
      assert out_cpp == out_c
    end
  end

  test "tight matmul loop produces identical final results" do
    for dev <- devices() do
      a = f32(32 * 32)
      b = f32(32 * 32)
      {:ok, _ns_cpp, out_cpp} = EMLX.C.RefNIF.tiny_loop(a, b, dev, 1000, 32)
      {:ok, _ns_c, out_c} = EMLX.C.NIF.tiny_loop(a, b, dev, 1000, 32)
      assert out_cpp == out_c
    end
  end
end
