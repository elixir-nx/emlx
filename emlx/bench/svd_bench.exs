# bench/svd_bench.exs
#
# Benchmarks `Nx.LinAlg.svd/1` across two MLX-backed Nx stacks — EMLX and Emily
# (ausimian/emily) — and, within each, across:
#
#   - device:    :cpu vs :gpu (GPU skipped automatically if Metal is unavailable)
#   - compiler:  :defn_evaluator   — `compiler: Nx.Defn.Evaluator`, dispatching op-by-op
#                                    onto whichever backend the input already lives on
#                                    (EMLX.Backend or Emily.Backend)
#                :defn_compiled    — `compiler: EMLX`, which lowers the whole graph to
#                                    a single fused native call
#                :defn_native      — `compiler: Emily.Compiler, native: true`, Emily's
#                                    equivalent single-NIF fused replay
#
# Every lane is built *once* per `{n, device}` via `Nx.Defn.compile/3` and that
# one closure is replayed across all Benchee-measured iterations — the
# realistic "compile once, run many" pattern (see `EMLX`'s moduledoc), not a
# fresh retrace (and, pre-`native_lowerable_block?/2`, dispatch-key rehash) on
# every call, which is what repeated `Nx.Defn.jit_apply/3` would measure.
#
# This isolates two independent costs: library (EMLX vs Emily) and dispatch overhead
# (per-op NIF round-trips vs one fused native call), on both CPU and GPU/Metal.
#
# This script is standalone — it does not need to run inside the `emlx` Mix project.
# `Mix.install/2` pulls in EMLX from this checkout (via a `path:` dependency) and
# Emily from Hex (Apple Silicon only), plus Benchee for the actual measurement.
#
# Run with (from anywhere):
#   elixir bench/svd_bench.exs
#
# Tunables (env vars):
#   EMLX_SVD_SIZES=128,256,512,1024   (default: 128,256,512,1024)
#   EMLX_SVD_TIME=2                   (Benchee measurement time per lane, in seconds)
#   EMLX_SVD_WARMUP_TIME=1            (Benchee warmup time per lane, in seconds; also
#                                       primes the defn_compiled/defn_native JIT caches)

apple_silicon? =
  match?({:unix, :darwin}, :os.type()) and
    :erlang.system_info(:system_architecture)
    |> to_string()
    |> String.starts_with?("aarch64")

# Emily's precompiled NIFs are Apple Silicon macOS only (see its README's
# Prerequisites section), so it's only added to the install list there — on any
# other host this script benchmarks EMLX alone.
deps =
  [
    {:emlx, path: Path.expand("..", __DIR__)},
    # Pin the same `nx` source as `emlx`'s own mix.exs (git main), overriding
    # Emily's Hex-sourced `~> 0.12` requirement, which main satisfies.
    {:nx, github: "elixir-nx/nx", branch: "main", sparse: "nx", override: true},
    {:benchee, "~> 1.3"}
  ] ++ if(apple_silicon?, do: [{:emily, "~> 0.7"}], else: [])

Mix.install(deps)

# MLX's CPU backend JIT-compiles fused kernels via `popen`/`pclose`, which fails with
# `ECHILD` under the BEAM's default SIGCHLD=SIG_IGN disposition — see
# `EMLX.Application`'s moduledoc. `elixir` scripts own this trade-off just like our
# test suite does.
:os.set_signal(:sigchld, :default)

defmodule Bench.SVD do
  # The benchmarked computation is always exactly `Nx.LinAlg.svd/1`
  # (`full_matrices?: true`) — every lane only varies the `Nx.Defn.compile/3`
  # `opts` (`:compiler`, `:device`, `:native`). Each lane's runner is built
  # *once* via `compile/3` (see `build/2`) and replayed across every
  # Benchee-measured iteration, mirroring the "compile once, run many" pattern
  # `EMLX`'s moduledoc recommends — not a fresh retrace per iteration, which
  # is the worst case `Nx.Defn.jit_apply/3` would otherwise measure.
  #
  # The compiled closure is stashed in `:persistent_term` (keyed by `key`)
  # rather than returned directly, because Benchee runs each job in its own
  # process: closing over the closure itself would make the *spawn* copy it
  # into that process, and `Nx.Defn.compile/3`'s closure keeps a reference to
  # the full (un-lowered) `Nx.Block` computation graph — including any
  # `default_expr` fallback, e.g. `Nx.LinAlg.svd/1`'s ~100-iteration
  # Jacobi-rotation algorithm, a DAG whose nodes heavily share
  # sub-expressions across iterations. Erlang's term-copy path doesn't
  # preserve that sharing, so flattening the DAG into a tree to copy it
  # blows up combinatorially and the spawn effectively never finishes.
  # `:persistent_term` values are read by reference (no copy), so fetching
  # the closure from inside the already-spawned job process sidesteps this
  # entirely.
  def build(key, template, opts) do
    compiled = Nx.Defn.compile(&Nx.LinAlg.svd(&1, full_matrices?: true), [template], opts)
    :persistent_term.put(key, compiled)
    key
  end

  # The `Nx.sum`/`Nx.to_number` calls aren't part of the compiled closure;
  # they just force materialization of all three outputs so a lazily-elided,
  # fused-but-unused output can't skew the timing.
  def run(key, a) do
    {u, s, vt} = :persistent_term.get(key).(a)
    Nx.to_number(Nx.sum(u)) + Nx.to_number(Nx.sum(s)) + Nx.to_number(Nx.sum(vt))
  end
end

sizes =
  case System.get_env("EMLX_SVD_SIZES") do
    nil -> [128, 256, 512, 1024]
    csv -> csv |> String.split(",") |> Enum.map(&String.to_integer(String.trim(&1)))
  end

parse_seconds = fn env, default -> {f, ""} = Float.parse(System.get_env(env, default)); f end

bench_time = parse_seconds.("EMLX_SVD_TIME", "2")
warmup_time = parse_seconds.("EMLX_SVD_WARMUP_TIME", "1")

emily_available? = apple_silicon? and Code.ensure_loaded?(Emily.Backend)

devices =
  case EMLX.NIF.command_queue_new(:gpu) do
    {:ok, _} ->
      [:cpu, :gpu]

    {:error, reason} ->
      IO.puts("==> GPU unavailable (#{inspect(reason)}), benchmarking CPU only.\n")
      [:cpu]
  end

engines = [:emlx] ++ if(emily_available?, do: [:emily], else: [])

unless emily_available? do
  IO.puts("==> Emily unavailable on this host (Apple Silicon macOS only), skipping.\n")
end

# Deterministic input, generated once on the host and transferred per-{engine,device}
# below — keeps the matrix identical across every combination.
random_matrix = fn n ->
  key = Nx.Random.key(42)
  {t, _key} = Nx.Random.uniform(key, shape: {n, n}, type: :f32)
  t
end

backend_for = fn
  :emlx, device -> {EMLX.Backend, device: device}
  :emily, device -> {Emily.Backend, device: device}
end

compiled_opts_for = fn
  :emlx, device -> [compiler: EMLX, device: device]
  :emily, device -> [compiler: Emily.Compiler, native: true, device: device]
end

# `emlx defn_evaluator` on `:gpu` has no native `full_matrices?: true` kernel
# (see `EMLX.Backend.block/4`'s `:gpu` clause), so `Nx.Defn.Evaluator` runs the
# `default_expr` fallback (a ~100-iteration Jacobi-rotation algorithm) fully
# eagerly, one NIF round-trip per op. That's *correct* but tens of seconds
# slow even at `n=128` and grows with `n`, so it's skipped by default — a
# single stuck-looking lane otherwise dominates the whole suite's wall time.
# Set `EMLX_SVD_INCLUDE_SLOW_GPU_EVALUATOR=1` to include it anyway.
include_slow_gpu_evaluator? = System.get_env("EMLX_SVD_INCLUDE_SLOW_GPU_EVALUATOR") == "1"

# Each lane builds its `Nx.Defn.compile/3` closure once (for `n`'s template)
# and returns a runner that replays it — the benchmarked function itself
# always lives in `Bench.SVD`, never as an inline anonymous function.
# (`defn_evaluator` has no comparable per-call compile cost to amortize, but
# building it the same way keeps lane-construction code uniform.)
lanes_for = fn engine, device, n ->
  native_key = {:svd_bench, engine, device, n, :native}
  template = Nx.template({n, n}, {:f, 32})
  Bench.SVD.build(native_key, template, compiled_opts_for.(engine, device))

  native_lane = [
    {(if engine == :emlx, do: :defn_compiled, else: :defn_native),
     &Bench.SVD.run(native_key, &1)}
  ]

  if engine == :emlx and device == :gpu and not include_slow_gpu_evaluator? do
    native_lane
  else
    evaluator_key = {:svd_bench, engine, device, n, :defn_evaluator}
    Bench.SVD.build(evaluator_key, template, compiler: Nx.Defn.Evaluator)
    [{:defn_evaluator, &Bench.SVD.run(evaluator_key, &1)} | native_lane]
  end
end

IO.puts("==> SVD benchmark (EMLX vs Emily)")
IO.puts("    sizes:   #{Enum.join(sizes, ", ")}")
IO.puts("    time:    #{bench_time}s per lane (warmup: #{warmup_time}s)")
IO.puts("    devices: #{Enum.join(devices, ", ")}")
IO.puts("    engines: #{Enum.join(engines, ", ")}")

unless include_slow_gpu_evaluator? do
  IO.puts(
    "    (skipping emlx defn_evaluator on :gpu — eager Jacobi fallback, tens of " <>
      "seconds+ per call; set EMLX_SVD_INCLUDE_SLOW_GPU_EVALUATOR=1 to include it)"
  )
end

IO.puts("")

suites =
  for n <- sizes, device <- devices, into: %{} do
    IO.puts("--- n=#{n} device=#{device} " <> String.duplicate("-", 40))

    jobs =
      for engine <- engines,
          {mode, runner} <- lanes_for.(engine, device, n),
          reduce: %{} do
        jobs ->
          a = Nx.backend_transfer(random_matrix.(n), backend_for.(engine, device))
          label = "#{engine} #{mode}"
          Map.put(jobs, label, fn -> runner.(a) end)
      end

    suite =
      Benchee.run(jobs,
        time: bench_time,
        warmup: warmup_time,
        memory_time: 0,
        print: [benchmarking: false, fast_warning: false],
        formatters: [Benchee.Formatters.Console]
      )

    {{n, device}, suite}
  end

# Cross-device speedup for the fused/native lane of each engine — Benchee only
# compares jobs within a single run, so CPU vs GPU (two separate runs) is
# summarized here from each run's own statistics.
median_ms = fn suite, job_name ->
  case suite && Enum.find(suite.scenarios, &(&1.job_name == job_name)) do
    nil -> nil
    scenario -> scenario.run_time_data.statistics.median / 1_000_000
  end
end

if :cpu in devices and :gpu in devices do
  IO.puts("\n=== CPU vs GPU speedup (fused/native lane, median ms) ===\n")

  Enum.each(engines, fn engine ->
    mode = if engine == :emlx, do: "defn_compiled", else: "defn_native"
    job_name = "#{engine} #{mode}"

    Enum.each(sizes, fn n ->
      cpu_ms = median_ms.(suites[{n, :cpu}], job_name)
      gpu_ms = median_ms.(suites[{n, :gpu}], job_name)

      case {cpu_ms, gpu_ms} do
        {c, g} when is_number(c) and is_number(g) and g > 0 ->
          speedup = Float.round(c / g, 2)

          IO.puts(
            "    #{engine} n=#{n}: cpu=#{Float.round(c, 3)} ms, gpu=#{Float.round(g, 3)} ms, speedup=#{speedup}x"
          )

        _ ->
          :ok
      end
    end)
  end)
end
