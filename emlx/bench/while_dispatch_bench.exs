# bench/while_dispatch_bench.exs
#
# Stage 14 gate measurement — is a C++ in-worker `while` worth building?
#
# The Stage-08 host loop drives `while` from Elixir: per iteration it makes two
# jit-dispatched `eval_program` NIF calls (condition + body) plus one
# `Nx.to_number` scalar pull to decide continuation. A C++ `while` (held
# pred/body subprograms looped on the worker) would collapse the whole loop into
# ONE NIF call, removing those per-iteration crossings — but MLX 0.31.2 has no
# in-trace control flow, so it CANNOT add cross-iteration fusion: the worker eval
# of cond+body per iteration is irreducible either way.
#
# So the load-bearing number is the REMOVABLE-OVERHEAD FRACTION per iteration:
#     removable = 2 * jit_dispatch_floor + to_number_floor
#     fraction  = removable / per_iteration_host_cost
# High fraction (light body OR a counter-only cond) => C++ while is worth it.
# Low fraction  (heavy body with a carry-reading cond) => host loop is fine.
#
# Two cond regimes are measured because they behave very differently under MLX
# laziness:
#   * counted    — cond reads only the loop counter. The body's lazy graph
#                  accumulates across iterations and fuses; only the counter is
#                  forced each step. (Models fixed trip-count loops.)
#   * convergent — cond also reads the carry (sum(abs(x)) >= 0, always true, but
#                  forces materialization). This reproduces a real
#                  Newton/fixed-point eval barrier every iteration.
#
# Run (from the emlx/ directory):
#   cd emlx
#   mix run bench/while_dispatch_bench.exs
#   EMLX_WHILE_DEVICE=gpu EMLX_WHILE_N=2000 mix run bench/while_dispatch_bench.exs

device = System.get_env("EMLX_WHILE_DEVICE", "cpu") |> String.to_atom()
n_iters = System.get_env("EMLX_WHILE_N", "1000") |> String.to_integer()
reps = System.get_env("EMLX_WHILE_REPS", "5") |> String.to_integer()

Nx.default_backend({EMLX.Backend, device: device})

defmodule WhileBench do
  import Nx.Defn

  # Elementwise body: weight scales with numel. cos keeps values finite in
  # [-1, 1] so the carry is shape- and value-stable across iterations.
  defn counted_cos(x, i, n) do
    while {x, i, n}, Nx.less(i, n) do
      {Nx.cos(x), Nx.add(i, 1), n}
    end
  end

  defn convergent_cos(x, i, n) do
    while {x, i, n}, Nx.logical_and(Nx.less(i, n), Nx.greater_equal(Nx.sum(Nx.abs(x)), 0.0)) do
      {Nx.cos(x), Nx.add(i, 1), n}
    end
  end

  # Matmul body: weight scales with dim^3. tanh + scale keeps the square carry
  # bounded so it stays finite across many iterations.
  defn counted_mat(x, i, n) do
    while {x, i, n}, Nx.less(i, n) do
      {Nx.tanh(Nx.multiply(Nx.dot(x, x), 0.01)), Nx.add(i, 1), n}
    end
  end

  defn convergent_mat(x, i, n) do
    while {x, i, n}, Nx.logical_and(Nx.less(i, n), Nx.greater_equal(Nx.sum(Nx.abs(x)), 0.0)) do
      {Nx.tanh(Nx.multiply(Nx.dot(x, x), 0.01)), Nx.add(i, 1), n}
    end
  end

  # Trivial 1-op program — the dispatch-overhead probe (BEAM<->NIF crossing +
  # Nx.Defn dispatch + output reconstruct), with ~zero worker compute.
  defn bump(x), do: Nx.add(x, 1)

  # Single body applications, used by the manual lazy-vs-forced loop probe that
  # models the two host-loop regimes without hitting the counted-while bug.
  defn cos_step(x), do: Nx.cos(x)
  defn mat_step(x), do: Nx.tanh(Nx.multiply(Nx.dot(x, x), 0.01))
end

emlx = [compiler: EMLX, device: device]

median = fn list -> list |> Enum.sort() |> Enum.at(div(length(list), 2)) end

time_us = fn fun ->
  {us, res} = :timer.tc(fun)
  # touch result to avoid dead-code elimination of the closure
  _ = res
  us
end

t = fn shape, type ->
  case shape do
    {} -> Nx.tensor(0.5, type: type, backend: {EMLX.Backend, device: device})
    _ -> Nx.broadcast(Nx.tensor(0.5, type: type, backend: {EMLX.Backend, device: device}), shape)
  end
end

i0 = Nx.tensor(0, type: :s32, backend: {EMLX.Backend, device: device})
n0 = Nx.tensor(n_iters, type: :s32, backend: {EMLX.Backend, device: device})

# ── Probe 1: jit-dispatch floor (no host read) ────────────────────────────────
bump_fn = Nx.Defn.jit(&WhileBench.bump/1, emlx)
xb = t.({}, :f32)
_ = bump_fn.(xb)

probe_reps = 2000

dispatch_floor_us =
  for(_ <- 1..probe_reps, do: time_us.(fn -> bump_fn.(xb) end))
  |> then(&(Enum.sum(&1) / probe_reps))

# ── Probe 2: to_number floor (scalar device->host pull, forces eval) ──────────
# Force one eval first so the value is materialized; we want the steady-state
# host-read cost, which is what `run_while_loop` pays on the counter each step.
scalar = bump_fn.(xb)
_ = Nx.to_number(scalar)

to_number_floor_us =
  for(_ <- 1..probe_reps, do: time_us.(fn -> Nx.to_number(scalar) end))
  |> then(&(Enum.sum(&1) / probe_reps))

removable_us = 2 * dispatch_floor_us + to_number_floor_us

IO.puts("""
================================================================================
 Stage 14 gate — host-loop `while` per-iteration cost decomposition
 device=#{device}  iterations/run(N)=#{n_iters}  reps=#{reps}
--------------------------------------------------------------------------------
 jit-dispatch floor (1-op, no host read) : #{Float.round(dispatch_floor_us, 2)} us
 to_number floor (scalar host pull)       : #{Float.round(to_number_floor_us, 2)} us
 => removable per iter (2*dispatch + pull): #{Float.round(removable_us, 2)} us
    (this is the MAX a C++ in-worker while can save per iteration)
================================================================================
""")

cases = [
  {"cos scalar {}", &WhileBench.counted_cos/3, &WhileBench.convergent_cos/3, {}},
  {"cos vec {1024}", &WhileBench.counted_cos/3, &WhileBench.convergent_cos/3, {1024}},
  {"cos mat {256,256}", &WhileBench.counted_cos/3, &WhileBench.convergent_cos/3, {256, 256}},
  {"dot mat {64,64}", &WhileBench.counted_mat/3, &WhileBench.convergent_mat/3, {64, 64}},
  {"dot mat {256,256}", &WhileBench.counted_mat/3, &WhileBench.convergent_mat/3, {256, 256}},
  {"dot mat {512,512}", &WhileBench.counted_mat/3, &WhileBench.convergent_mat/3, {512, 512}}
]

run_case = fn fun, shape ->
  jit = Nx.Defn.jit(fun, emlx)
  x = t.(shape, :f32)
  # Warmup / compile, and force the final carry so the whole loop materializes.
  {wx, wi, _} = jit.(x, i0, n0)
  _ = Nx.to_number(Nx.sum(wx))
  final_i = Nx.to_number(wi)

  us_list =
    for _ <- 1..reps do
      time_us.(fn ->
        {rx, _, _} = jit.(x, i0, n0)
        # Force the final carry so lazy bodies actually evaluate (counted case
        # would otherwise defer all body work past the timer).
        Nx.to_number(Nx.sum(rx))
      end)
    end

  total = median.(us_list)
  {total, total / n_iters, final_i}
end

IO.puts(
  String.pad_trailing("body", 22) <>
    String.pad_trailing("regime", 12) <>
    String.pad_trailing("iters", 8) <>
    String.pad_trailing("us/iter", 12) <>
    String.pad_trailing("removable%", 12) <> "C++ ceiling (us/iter)"
)

IO.puts(String.duplicate("-", 84))

for {label, counted_fun, conv_fun, shape} <- cases do
  for {regime, fun} <- [{"counted", counted_fun}, {"convergent", conv_fun}] do
    try do
      {_total, per_iter, final_i} = run_case.(fun, shape)
      frac = min(removable_us / per_iter * 100.0, 100.0)
      ceiling = max(per_iter - removable_us, 0.0)

      IO.puts(
        String.pad_trailing(label, 22) <>
          String.pad_trailing(regime, 12) <>
          String.pad_trailing(Integer.to_string(round(final_i)), 8) <>
          String.pad_trailing(Float.to_string(Float.round(per_iter, 2)), 12) <>
          String.pad_trailing(Float.to_string(Float.round(frac, 1)) <> "%", 12) <>
          Float.to_string(Float.round(ceiling, 2))
      )
    rescue
      e ->
        IO.puts(
          String.pad_trailing(label, 22) <>
            String.pad_trailing(regime, 12) <> ("ERROR: " <> Exception.message(e)) |> String.slice(0, 60)
        )
    end
  end
end

# ── Probe 2b: lazy-vs-forced host loop (models the two `while` regimes) ───────
# A C++ in-worker `while` must eval the predicate each iteration (MLX 0.31.2 has
# no in-trace control flow), so it behaves like the FORCED loop minus the
# per-iteration BEAM<->NIF crossing. The COUNTED-while host loop instead lets the
# body accumulate lazily and fuse (only the counter is forced) -> it behaves like
# the LAZY loop. This probe measures both directly.
#   * lazy   per-iter: dispatch only; body graph fuses, evaluated once at end.
#   * forced per-iter: full eval barrier every iteration (worker sync).
# C++-while ceiling ~= forced - removable(crossing). If lazy << forced, a C++
# while (forced-style) is WORSE than the counted host loop (lazy-style).

step_cases = [
  {"cos  vec {1024}", &WhileBench.cos_step/1, {1024}},
  {"cos  mat {256,256}", &WhileBench.cos_step/1, {256, 256}},
  {"dot  mat {64,64}", &WhileBench.mat_step/1, {64, 64}},
  {"dot  mat {256,256}", &WhileBench.mat_step/1, {256, 256}},
  {"dot  mat {512,512}", &WhileBench.mat_step/1, {512, 512}}
]

IO.puts("\n" <> String.duplicate("=", 84))
IO.puts(" Lazy (counted-while, fuses) vs Forced (convergent-while / C++-while) per-iter")
IO.puts(String.duplicate("-", 84))

IO.puts(
  String.pad_trailing("body", 20) <>
    String.pad_trailing("lazy us/it", 14) <>
    String.pad_trailing("forced us/it", 14) <>
    String.pad_trailing("C++ ceil us/it", 16) <> "verdict"
)

IO.puts(String.duplicate("-", 84))

for {label, step_fun, shape} <- step_cases do
  step = Nx.Defn.jit(step_fun, emlx)
  x0 = t.(shape, :f32)
  _ = Nx.to_number(Nx.sum(step.(x0)))

  lazy_us =
    for _ <- 1..reps do
      time_us.(fn ->
        r = Enum.reduce(1..n_iters, x0, fn _, acc -> step.(acc) end)
        Nx.to_number(Nx.sum(r))
      end)
    end
    |> median.()
    |> then(&(&1 / n_iters))

  forced_us =
    for _ <- 1..reps do
      time_us.(fn ->
        Enum.reduce(1..n_iters, x0, fn _, acc ->
          r = step.(acc)
          _ = Nx.to_number(Nx.sum(r))
          r
        end)
      end)
    end
    |> median.()
    |> then(&(&1 / n_iters))

  # C++ while removes ONE crossing/iter vs forced (loop stays on worker); it still
  # pays the per-iter eval barrier. Ceiling = forced - dispatch_floor.
  cpp_ceiling = max(forced_us - dispatch_floor_us, 0.0)

  verdict =
    if lazy_us < cpp_ceiling,
      do: "host(lazy) BEATS C++ while",
      else: "C++ while saves #{Float.round(forced_us - cpp_ceiling, 1)}us/it vs forced"

  IO.puts(
    String.pad_trailing(label, 20) <>
      String.pad_trailing(Float.to_string(Float.round(lazy_us, 2)), 14) <>
      String.pad_trailing(Float.to_string(Float.round(forced_us, 2)), 14) <>
      String.pad_trailing(Float.to_string(Float.round(cpp_ceiling, 2)), 16) <> verdict
  )
end

# ── Probe 3: Graph.split fragmentation — a PER-INVOCATION fixed cost ──────────
# A `while` surrounded by other work splits into pre/while/post stages
# (build_while_chain_eval_fn). This cost is independent of trip count; the
# advisor flagged not to conflate it with per-iteration overhead. Measure the
# warm per-call overhead of a chained-while defn beyond the bare loop.

defmodule WhileChainBench do
  import Nx.Defn

  # Carry-dependent cond (sum(abs)>=0 is always true) avoids the counter-only
  # bare-while bug and forces a real per-iteration eval barrier.
  defn bare(x, i, n) do
    while {x, i, n}, Nx.logical_and(Nx.less(i, n), Nx.greater_equal(Nx.sum(Nx.abs(x)), 0.0)) do
      {Nx.cos(x), Nx.add(i, 1), n}
    end
  end

  # Same loop but wrapped in pre/post work -> Graph.split into 3 stages.
  defn chained(x, i, n) do
    pre = Nx.add(x, 0.1)

    {y, _, _} =
      while {acc = pre, j = i, m = n},
            Nx.logical_and(Nx.less(j, m), Nx.greater_equal(Nx.sum(Nx.abs(acc)), 0.0)) do
        {Nx.cos(acc), Nx.add(j, 1), m}
      end

    Nx.cos(y)
  end
end

IO.puts("\n" <> String.duplicate("=", 80))
IO.puts(" Graph.split fragmentation — per-invocation fixed cost (small N=20)")
IO.puts(String.duplicate("-", 80))

small_n = Nx.tensor(20, type: :s32, backend: {EMLX.Backend, device: device})
xc = t.({256}, :f32)

bare_jit = Nx.Defn.jit(&WhileChainBench.bare/3, emlx)
{bw, _, _} = bare_jit.(xc, i0, small_n)
_ = Nx.to_number(Nx.sum(bw))

bare_us =
  for(_ <- 1..reps, do: time_us.(fn ->
    {r, _, _} = bare_jit.(xc, i0, small_n)
    Nx.to_number(Nx.sum(r))
  end))
  |> median.()
  |> Kernel.*(1.0)

chain_jit = Nx.Defn.jit(&WhileChainBench.chained/3, emlx)
cw = chain_jit.(xc, i0, small_n)
_ = Nx.to_number(Nx.sum(cw))

chain_us =
  for(_ <- 1..reps, do: time_us.(fn ->
    r = chain_jit.(xc, i0, small_n)
    Nx.to_number(Nx.sum(r))
  end))
  |> median.()
  |> Kernel.*(1.0)

IO.puts("bare while   (20 iters)     : #{Float.round(bare_us, 1)} us total")
IO.puts("chained while (20 iters)    : #{Float.round(chain_us, 1)} us total")
IO.puts("Graph.split delta (fixed)   : #{Float.round(chain_us - bare_us, 1)} us / invocation")
IO.puts("dispatch floor (per call)   : #{Float.round(dispatch_floor_us, 2)} us")
IO.puts("removable per iter          : #{Float.round(removable_us, 2)} us")
IO.puts(String.duplicate("=", 80))
