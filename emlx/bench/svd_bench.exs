# bench/svd_bench.exs
#
# Compares EMLX's SVD (a heavy, multi-output LAPACK-style kernel) across:
#
#   - device:    :cpu vs :gpu (GPU skipped automatically if Metal is unavailable)
#   - execution: :eager            — direct EMLX.Backend calls, one NIF round-trip per op
#                :defn_evaluator    — same computation wrapped in `defn`, run via the
#                                     default `Nx.Defn.Evaluator` (interprets the graph,
#                                     still one NIF round-trip per op)
#                :defn_compiled     — same `defn`, jitted with `compiler: EMLX`, which
#                                     lowers the whole graph to a single fused native
#                                     `compile_program`/`eval_program` NIF call
#
# This isolates two independent costs: device (CPU vs GPU/Metal) and dispatch overhead
# (per-op NIF round-trips vs one fused native call).
#
# Run with:
#   cd emlx
#   mix run bench/svd_bench.exs
#
# Tunables (env vars):
#   EMLX_SVD_SIZES=128,256,512,1024   (default: 128,256,512,1024)
#   EMLX_SVD_ITERS=5                  (timed iterations per combination)
#   EMLX_SVD_WARMUP=2                 (untimed warmup iterations; also primes the
#                                      defn_compiled JIT cache)
#   EMLX_SVD_TIMEOUT_MS=15000         (per-call timeout; :eager/:defn_evaluator fall
#                                      back to Nx.LinAlg.SVD's generic composite
#                                      while-loop algorithm — since EMLX.Backend has
#                                      no native eager `svd` op — which can be very
#                                      slow, or hang, especially on the GPU device)

# MLX's CPU backend JIT-compiles fused kernels via `popen`/`pclose`, which fails with
# `ECHILD` under the BEAM's default SIGCHLD=SIG_IGN disposition — see
# `EMLX.Application`'s moduledoc. `mix run` scripts own this trade-off just like our
# test suite does.
:os.set_signal(:sigchld, :default)

defmodule EMLX.Bench.SVD do
  import Nx.Defn

  # Sums every output so the whole decomposition is materialized (and the
  # comparison isn't skewed by a fused-but-unused output being elided).
  defn run(a) do
    {u, s, vt} = Nx.LinAlg.svd(a)
    Nx.sum(u) + Nx.sum(s) + Nx.sum(vt)
  end
end

sizes =
  case System.get_env("EMLX_SVD_SIZES") do
    nil -> [128, 256, 512, 1024]
    csv -> csv |> String.split(",") |> Enum.map(&String.to_integer(String.trim(&1)))
  end

iters = String.to_integer(System.get_env("EMLX_SVD_ITERS", "5"))
warmup = String.to_integer(System.get_env("EMLX_SVD_WARMUP", "2"))
timeout_ms = String.to_integer(System.get_env("EMLX_SVD_TIMEOUT_MS", "15000"))

devices =
  case EMLX.NIF.command_queue_new(:gpu) do
    {:ok, _} ->
      [:cpu, :gpu]

    {:error, reason} ->
      IO.puts("==> GPU unavailable (#{inspect(reason)}), benchmarking CPU only.\n")
      [:cpu]
  end

modes = [:eager, :defn_evaluator, :defn_compiled]

# Deterministic input, generated once on the host and transferred per-device below —
# keeps the matrix identical across every {device, mode} combination.
random_matrix = fn n ->
  key = Nx.Random.key(42)
  {t, _key} = Nx.Random.uniform(key, shape: {n, n}, type: :f32)
  t
end

sync! = fn %Nx.Tensor{} = scalar -> Nx.to_number(scalar) end

eager_fun = fn a ->
  {u, s, vt} = Nx.LinAlg.svd(a)
  Nx.sum(u) |> Nx.add(Nx.sum(s)) |> Nx.add(Nx.sum(vt))
end

build_runner = fn device, mode ->
  case mode do
    :eager ->
      eager_fun

    :defn_evaluator ->
      Nx.Defn.jit(&EMLX.Bench.SVD.run/1, compiler: Nx.Defn.Evaluator)

    :defn_compiled ->
      Nx.Defn.jit(&EMLX.Bench.SVD.run/1, compiler: EMLX, device: device)
  end
end

median = fn list ->
  sorted = Enum.sort(list)
  n = length(sorted)
  Enum.at(sorted, div(n, 2))
end

bench_one = fn a, runner ->
  task =
    Task.async(fn ->
      try do
        for _ <- 1..warmup, do: sync!.(runner.(a))

        times_ms =
          for _ <- 1..iters do
            t0 = System.monotonic_time(:microsecond)
            sync!.(runner.(a))
            t1 = System.monotonic_time(:microsecond)
            (t1 - t0) / 1000.0
          end

        Float.round(median.(times_ms), 3)
      rescue
        e -> {:error, Exception.format(:error, e, []) |> String.split("\n") |> hd()}
      catch
        kind, reason -> {:error, "#{kind}: #{inspect(reason)}"}
      end
    end)

  case Task.yield(task, timeout_ms) || Task.shutdown(task, :brutal_kill) do
    {:ok, result} -> result
    nil -> {:error, "timeout after #{timeout_ms}ms"}
  end
end

IO.puts("==> EMLX SVD benchmark")
IO.puts("    sizes:   #{Enum.join(sizes, ", ")}")
IO.puts("    iters:   #{iters} (warmup: #{warmup})")
IO.puts("    devices: #{Enum.join(devices, ", ")}\n")

results =
  for n <- sizes, device <- devices do
    a = Nx.backend_transfer(random_matrix.(n), {EMLX.Backend, device: device})

    row =
      for mode <- modes do
        runner = build_runner.(device, mode)
        result = bench_one.(a, runner)

        case result do
          {:error, msg} -> IO.puts("    n=#{n} device=#{device} mode=#{mode}: ERROR (#{msg})")
          ms -> IO.puts("    n=#{n} device=#{device} mode=#{mode}: #{ms} ms")
        end

        {mode, result}
      end

    {n, device, Map.new(row)}
  end

IO.puts("\n=== Summary (median ms over #{iters} runs) ===\n")

header =
  "| size | device | eager | defn_evaluator | defn_compiled | evaluator/eager | compiled/eager |"

sep = "|------|--------|-------|----------------|----------------|------------------|-----------------|"

IO.puts(header)
IO.puts(sep)

fmt = fn
  {:error, _} -> "error"
  ms -> ms
end

ratio = fn a, b -> if is_number(a) and is_number(b) and b > 0, do: Float.round(a / b, 2) end

Enum.each(results, fn {n, device, by_mode} ->
  eager = by_mode[:eager]
  evaluator = by_mode[:defn_evaluator]
  compiled = by_mode[:defn_compiled]

  IO.puts(
    "| #{n} | #{device} | #{fmt.(eager)} | #{fmt.(evaluator)} | #{fmt.(compiled)} | " <>
      "#{ratio.(evaluator, eager)} | #{ratio.(compiled, eager)} |"
  )
end)

if :cpu in devices and :gpu in devices do
  IO.puts("\n=== CPU vs GPU speedup (compiled/fused mode) ===\n")

  by_size =
    results
    |> Enum.group_by(fn {n, _device, _} -> n end)

  Enum.each(sizes, fn n ->
    case Map.get(by_size, n) do
      [{_, :cpu, cpu_row}, {_, :gpu, gpu_row}] ->
        cpu_ms = cpu_row[:defn_compiled]
        gpu_ms = gpu_row[:defn_compiled]
        speedup = ratio.(cpu_ms, gpu_ms)
        IO.puts("    n=#{n}: cpu=#{fmt.(cpu_ms)} ms, gpu=#{fmt.(gpu_ms)} ms, speedup=#{speedup}x")

      _ ->
        :ok
    end
  end)
end
