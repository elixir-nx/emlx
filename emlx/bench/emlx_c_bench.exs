# bench/emlx_c_bench.exs
#
# Two-lane benchmark: identical work through (A) direct mlx::core C++ calls
# (the style emlx's c_src uses today) and (B) the official mlx-c API.
# Both lanes run inside this one BEAM on their own statically linked libmlx,
# and lanes are sampled interleaved rep-by-rep so thermal/background load
# affects both equally.
#
# Run with (after `EMLX_MLXC=true mix compile`):
#   mix run bench/emlx_c_bench.exs
#
# Tunables (env vars):
#   DEVICES=gpu,cpu          devices to test (default: gpu if available)
#   REPS=15                  timed reps for large ops
#   REPS_SMALL=50            timed reps for small ops
#   TINY_ITERS=20000         inner iterations for the dispatch-overhead lane

defmodule B do
  def f32(n) do
    for _ <- 1..n, into: <<>>, do: <<:rand.uniform() - 0.5::float-32-little>>
  end

  def f16(n) do
    for _ <- 1..n, into: <<>> do
      sign = :rand.uniform(2) - 1
      exp = 15 + :rand.uniform(5) - 3
      mant = :rand.uniform(1024) - 1
      <<sign * 32768 + Bitwise.bsl(exp, 10) + mant::16-little>>
    end
  end

  def max_diff(b1, b2) do
    Enum.zip(to_floats(b1), to_floats(b2))
    |> Enum.reduce(0.0, fn {a, b}, acc -> max(acc, abs(a - b)) end)
  end

  def to_floats(bin), do: for(<<f::float-32-little <- bin>>, do: f)
end

defmodule Stats do
  def time_pair(fa, fb, reps, warmup) do
    for _ <- 1..warmup do
      fa.()
      fb.()
    end

    {ta, tb} =
      for _ <- 1..reps do
        {elem(:timer.tc(fa), 0), elem(:timer.tc(fb), 0)}
      end
      |> Enum.unzip()

    {sort_stats(ta), sort_stats(tb)}
  end

  defp sort_stats(ts) do
    ts = Enum.sort(ts)
    {List.first(ts), Enum.at(ts, div(length(ts), 2)), List.last(ts)}
  end

  def fmt(us), do: :io_lib.format("~.1f", [us / 1.0]) |> IO.iodata_to_binary()
end

defmodule Main do
  def run() do
    {:ok, gpu?, ver} = EMLX.C.NIF.device_check()
    IO.puts("mlx-c lane: libmlx #{ver}, gpu #{inspect(gpu?)}")
    IO.puts("reference lane: #{inspect(EMLX.C.RefNIF.device_check())}")

    devices =
      case System.get_env("DEVICES") do
        nil -> if gpu? == :gpu, do: [:gpu, :cpu], else: [:cpu]
        s -> s |> String.split(",") |> Enum.map(&String.to_atom/1)
      end

    reps = String.to_integer(System.get_env("REPS") || "15")
    reps_small = String.to_integer(System.get_env("REPS_SMALL") || "50")
    tiny_iters = String.to_integer(System.get_env("TINY_ITERS") || "20000")

    jobs = [
      fn d ->
        {"matmul fp32 4096x4096", :matmul,
         [B.f32(4096 * 4096), B.f32(4096 * 4096), :float32, d, 4096, 4096, 4096], reps}
      end,
      fn d ->
        {"matmul fp32 1024x1024", :matmul,
         [B.f32(1024 * 1024), B.f32(1024 * 1024), :float32, d, 1024, 1024, 1024], reps_small}
      end,
      fn d ->
        {"matmul fp16 4096x4096", :matmul,
         [B.f16(4096 * 4096), B.f16(4096 * 4096), :float16, d, 4096, 4096, 4096], reps}
      end,
      fn d ->
        {"bmm fp16 16x512x512", :bmm,
         [B.f16(16 * 512 * 512), B.f16(16 * 512 * 512), :float16, d, 16, 512, 512, 512],
         reps_small}
      end,
      fn d ->
        {"add fp32 4096x4096", :add,
         [B.f32(4096 * 4096), B.f32(4096 * 4096), :float32, d, 4096, 4096], reps}
      end,
      fn d ->
        {"softmax fp32 4096x4096", :softmax_axis,
         [B.f32(4096 * 4096), :float32, d, 4096, 4096, 1], reps}
      end,
      fn d ->
        {"sdpa fp16 2x8x1024x128", :attn,
         [
           B.f16(2 * 8 * 1024 * 128),
           B.f16(2 * 8 * 1024 * 128),
           B.f16(2 * 8 * 1024 * 128),
           0.0883883476,
           d,
           2,
           8,
           1024,
           128
         ], reps_small}
      end,
      fn d ->
        {"svd fp32 512 (s only)", :svd, [B.f32(512 * 512), :cpu, 512], reps_small}
      end
    ]

    for d <- devices do
      IO.puts("\n== device: #{d} ==")

      IO.puts(
        :io_lib.format("~34ts ~12ts ~12ts ~8ts~n", ["op (min us)", "cpp", "mlx-c", "c/cpp"])
      )

      for jf <- jobs do
        {name, op, args, reps_i} = jf.(d)

        fa = fn -> apply(EMLX.C.RefNIF, op, args) |> unwrap() end
        fb = fn -> apply(EMLX.C.NIF, op, args) |> unwrap() end

        {sa, sb} = Stats.time_pair(fa, fb, reps_i, 3)
        {amin, _, _} = sa
        {bmin, _, _} = sb

        IO.puts(
          :io_lib.format(
            "~34ts ~12ts ~12ts ~8ts~n",
            [name, Stats.fmt(amin), Stats.fmt(bmin), Stats.fmt(bmin / amin)]
          )
        )

        check_parity(op, args)
      end

      # dispatch-overhead lane (timing measured inside the NIF)
      a = B.f32(32 * 32)
      b = B.f32(32 * 32)
      {:ok, ns_cpp, out_cpp} = EMLX.C.RefNIF.tiny_loop(a, b, d, tiny_iters, 32)
      {:ok, ns_c, out_c} = EMLX.C.NIF.tiny_loop(a, b, d, tiny_iters, 32)

      IO.puts(
        :io_lib.format(
          "~34ts ~12ts ~12ts ~8ts~n",
          [
            "tiny 32x32 matmul x#{tiny_iters} (per-op)",
            Stats.fmt(ns_cpp / tiny_iters / 1000),
            Stats.fmt(ns_c / tiny_iters / 1000),
            Stats.fmt(ns_c * 1.0 / ns_cpp)
          ]
        )
      )

      IO.puts("  tiny outputs identical: #{out_cpp == out_c}")
    end

    IO.puts("\ndone")
  end

  defp unwrap({:ok, b}), do: b
  defp unwrap({:ok, _, b}), do: b
  defp unwrap({:error, r}), do: raise("NIF error: #{inspect(r)}")

  defp check_parity(op, args) do
    ra = apply(EMLX.C.RefNIF, op, args) |> unwrap()
    rb = apply(EMLX.C.NIF, op, args) |> unwrap()

    cond do
      ra == rb ->
        :ok

      byte_size(ra) == byte_size(rb) ->
        IO.puts("  PARITY #{op}: max|diff| = #{inspect(B.max_diff(ra, rb))}")

      true ->
        IO.puts("  PARITY #{op}: size mismatch")
    end
  end
end

Main.run()
