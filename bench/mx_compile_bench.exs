#!/usr/bin/env elixir
# C03 — graph_capture / graph_replay benchmark
#
# Tests graph_capture/replay speedup across kernels of increasing graph depth
# and NIF-call count, to determine whether NIF dispatch or GPU execution
# dominates, and at what graph size compile gives ≥2×.
#
# Kernels (increasing NIF count, decreasing NIF/GPU ratio):
#   K1 — elementwise chain  : 100 sequential Nx ops, small GPU work per op
#   K2 — FFN block          : ~6 Nx ops, large GPU matmuls (Qwen3-0.6B proxy)
#   K3 — 28-layer FFN stack : 28× K2 on the same inputs (full decode proxy)
#   K4 — SVD 512×512        : 1 Nx op, complex internal graph in MLX
#   K5 — SVD 1024×1024      : 1 Nx op, larger SVD (more internal GPU kernels)
#
# Run:
#   mix run bench/mx_compile_bench.exs
#   EMLX_BENCH_ITERS=200 mix run bench/mx_compile_bench.exs

defmodule MxCompileBench do
  @iters String.to_integer(System.get_env("EMLX_BENCH_ITERS", "200"))
  @warmup 15

  # ── Helpers ─────────────────────────────────────────────────────────────

  defp raw_ref(t), do: elem(t.data.ref, 1)
  defp dev(t),     do: elem(t.data.ref, 0)

  defp tensor(val, shape, type \\ :f32),
    do: Nx.broadcast(Nx.tensor(val, type: type, backend: EMLX.Backend), shape)

  defp bench(label, n, fun) do
    for _ <- 1..@warmup, do: fun.()
    t0 = System.monotonic_time(:microsecond)
    for _ <- 1..n, do: fun.()
    t1 = System.monotonic_time(:microsecond)
    per = (t1 - t0) / n
    IO.puts("    #{String.pad_trailing(label, 22)}: #{Float.round(per, 1)} μs/iter")
    per
  end

  # Capture graph, measure capture latency, return compiled_ref
  defp capture(input_tensors, output_tensors) do
    inputs  = Enum.map(input_tensors,  &raw_ref/1)
    outputs = Enum.map(output_tensors, &raw_ref/1)
    {us, {:ok, cr}} = :timer.tc(fn -> EMLX.NIF.graph_capture(inputs, outputs, false) end)
    {cr, inputs, us}
  end

  defp replay_and_eval(cr, input_refs, dev, _n_outputs) do
    {:ok, out_raws} = EMLX.NIF.graph_replay(cr, input_refs)
    Enum.each(out_raws, fn r -> EMLX.eval({dev, r}) end)
    # Return first output ref for correctness checks
    {dev, hd(out_raws)}
  end

  # Check max-abs-diff between ref result and replayed result
  defp check_correctness(ref_tensor, cr, input_refs, dev, shape, type) do
    {:ok, [r | _]} = EMLX.NIF.graph_replay(cr, input_refs)
    rep = {dev, r} |> EMLX.Backend.to_nx(Nx.template(shape, type))
    Nx.subtract(ref_tensor, rep) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  defp print_result(baseline, compiled, capture_us, max_diff, gate) do
    speedup = baseline / compiled
    pass    = if speedup >= gate, do: "✓ PASS", else: "✗ FAIL"
    IO.puts("    baseline  #{Float.round(baseline, 1)} μs  │  " <>
            "compiled  #{Float.round(compiled, 1)} μs  │  " <>
            "speedup  #{Float.round(speedup, 2)}×  │  " <>
            "capture  #{capture_us} μs  │  " <>
            "max_diff  #{Float.round(max_diff * 1.0, 4)}  │  " <>
            "#{pass} (gate #{gate}×)")
    speedup
  end

  # ── Kernel K1: elementwise chain ─────────────────────────────────────────

  def bench_elementwise_chain do
    IO.puts("\n── K1  Elementwise chain (100 sequential ops, 1024-elem vector) ──")
    n = 100
    x = tensor(1.0, {1024})

    # Build the chain
    chain_fn = fn x ->
      Enum.reduce(1..n, x, fn i, acc ->
        scale = tensor(1.0 + i * 0.001, {1024})
        Nx.multiply(acc, scale)
      end)
    end

    out_trace = chain_fn.(x)
    # Inputs = x + all n scale tensors (captures them as constants in the tape)
    {cr, input_refs, cap_us} = capture([x], [out_trace])
    d = dev(x)

    t_base = bench("dispatch+eval", @iters, fn ->
      EMLX.eval(chain_fn.(x).data.ref)
    end)

    t_comp = bench("replay+eval",   @iters, fn ->
      replay_and_eval(cr, input_refs, d, 1)
    end)

    diff = check_correctness(chain_fn.(x), cr, input_refs, d, {1024}, :f32)
    print_result(t_base, t_comp, cap_us, diff, 1.5)
  end

  # ── Kernel K2: single FFN block ───────────────────────────────────────────

  def bench_ffn_block do
    IO.puts("\n── K2  Single FFN block (Qwen3-0.6B, decode seq_len=1) ──────────")
    # hidden=1024, intermediate=2816
    x  = tensor(0.1,  {1, 1024})
    w1 = tensor(0.01, {1024, 2816})
    w2 = tensor(0.01, {2816, 1024})
    b  = tensor(0.0,  {1, 1024})

    ffn = fn x, w1, w2, b ->
      x |> Nx.dot(w1) |> Nx.sigmoid() |> Nx.dot(w2) |> Nx.add(b)
    end

    out_trace = ffn.(x, w1, w2, b)
    {cr, input_refs, cap_us} = capture([x, w1, w2, b], [out_trace])
    d = dev(x)

    t_base = bench("dispatch+eval", @iters, fn ->
      EMLX.eval(ffn.(x, w1, w2, b).data.ref)
    end)

    t_comp = bench("replay+eval",   @iters, fn ->
      replay_and_eval(cr, input_refs, d, 1)
    end)

    diff = check_correctness(ffn.(x, w1, w2, b), cr, input_refs, d, {1, 1024}, :f32)
    print_result(t_base, t_comp, cap_us, diff, 2.0)
  end

  # ── Kernel K3: 28-layer FFN stack ─────────────────────────────────────────

  def bench_ffn_stack do
    IO.puts("\n── K3  28-layer FFN stack (28× K2, full decode proxy) ───────────")
    x  = tensor(0.1,  {1, 1024})
    w1 = tensor(0.01, {1024, 2816})
    w2 = tensor(0.01, {2816, 1024})
    b  = tensor(0.0,  {1, 1024})

    stack_fn = fn x, w1, w2, b ->
      Enum.reduce(1..28, x, fn _, acc ->
        acc |> Nx.dot(w1) |> Nx.sigmoid() |> Nx.dot(w2) |> Nx.add(b)
      end)
    end

    out_trace = stack_fn.(x, w1, w2, b)
    {cr, input_refs, cap_us} = capture([x, w1, w2, b], [out_trace])
    d = dev(x)

    t_base = bench("dispatch+eval", @iters, fn ->
      EMLX.eval(stack_fn.(x, w1, w2, b).data.ref)
    end)

    t_comp = bench("replay+eval",   @iters, fn ->
      replay_and_eval(cr, input_refs, d, 1)
    end)

    diff = check_correctness(stack_fn.(x, w1, w2, b), cr, input_refs, d, {1, 1024}, :f32)
    print_result(t_base, t_comp, cap_us, diff, 2.0)
  end

  # ── Kernel K4: SVD 512×512 ─────────────────────────────────────────────

  def bench_svd_512 do
    IO.puts("\n── K4  SVD 512×512 (single NIF, complex internal MLX graph) ────")
    m = tensor(0.1, {512, 512})
    # SVD returns {u, s, vt} — we capture all 3 outputs
    {u, s, vt} = Nx.LinAlg.svd(m, full_matrices?: false)
    {cr, input_refs, cap_us} = capture([m], [u, s, vt])
    d = dev(m)

    t_base = bench("dispatch+eval", @iters, fn ->
      {u2, s2, vt2} = Nx.LinAlg.svd(m, full_matrices?: false)
      EMLX.eval(u2.data.ref); EMLX.eval(s2.data.ref); EMLX.eval(vt2.data.ref)
    end)

    t_comp = bench("replay+eval",   @iters, fn ->
      {:ok, [ur, sr, vtr]} = EMLX.NIF.graph_replay(cr, input_refs)
      EMLX.eval({d, ur}); EMLX.eval({d, sr}); EMLX.eval({d, vtr})
    end)

    # Correctness: check U only
    {:ok, [ur | _]} = EMLX.NIF.graph_replay(cr, input_refs)
    {u_ref, _, _} = Nx.LinAlg.svd(m, full_matrices?: false)
    rep_u = {d, ur} |> EMLX.Backend.to_nx(Nx.template({512, 512}, :f32))
    diff = Nx.subtract(u_ref, rep_u) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    print_result(t_base, t_comp, cap_us, diff, 1.5)
  end

  # ── Kernel K5: SVD 1024×1024 ───────────────────────────────────────────

  def bench_svd_1024 do
    IO.puts("\n── K5  SVD 1024×1024 (larger SVD, more internal GPU kernels) ────")
    m = tensor(0.1, {1024, 1024})
    {u, s, vt} = Nx.LinAlg.svd(m, full_matrices?: false)
    {cr, input_refs, cap_us} = capture([m], [u, s, vt])
    d = dev(m)

    t_base = bench("dispatch+eval", @iters, fn ->
      {u2, s2, vt2} = Nx.LinAlg.svd(m, full_matrices?: false)
      EMLX.eval(u2.data.ref); EMLX.eval(s2.data.ref); EMLX.eval(vt2.data.ref)
    end)

    t_comp = bench("replay+eval",   @iters, fn ->
      {:ok, [ur, sr, vtr]} = EMLX.NIF.graph_replay(cr, input_refs)
      EMLX.eval({d, ur}); EMLX.eval({d, sr}); EMLX.eval({d, vtr})
    end)

    {:ok, [ur | _]} = EMLX.NIF.graph_replay(cr, input_refs)
    {u_ref, _, _} = Nx.LinAlg.svd(m, full_matrices?: false)
    rep_u = {d, ur} |> EMLX.Backend.to_nx(Nx.template({1024, 1024}, :f32))
    diff = Nx.subtract(u_ref, rep_u) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    print_result(t_base, t_comp, cap_us, diff, 1.5)
  end

  # ── Overhead breakdown ─────────────────────────────────────────────────

  def bench_overhead do
    IO.puts("\n── Overhead breakdown ────────────────────────────────────────────")
    n = 1000
    x  = tensor(0.1,  {1, 1024})
    w1 = tensor(0.01, {1024, 2816})
    ffn1 = fn x, w1 -> x |> Nx.dot(w1) |> Nx.sigmoid() end
    out = ffn1.(x, w1)
    {cr, input_refs, _} = capture([x, w1], [out])
    nif_dispatch_only =
      (for _ <- 1..n, do: ffn1.(x, w1)) |> then(fn _ ->
        t0 = System.monotonic_time(:microsecond)
        for _ <- 1..n, do: ffn1.(x, w1)
        t1 = System.monotonic_time(:microsecond)
        (t1 - t0) / n
      end)

    graph_replay_only =
      (for _ <- 1..n, do: EMLX.NIF.graph_replay(cr, input_refs)) |> then(fn _ ->
        t0 = System.monotonic_time(:microsecond)
        for _ <- 1..n, do: EMLX.NIF.graph_replay(cr, input_refs)
        t1 = System.monotonic_time(:microsecond)
        (t1 - t0) / n
      end)

    eval_only =
      # Eval same pre-built array (GPU time lower bound — cached result, no-op)
      (for _ <- 1..n, do: EMLX.eval(out.data.ref)) |> then(fn _ ->
        t0 = System.monotonic_time(:microsecond)
        for _ <- 1..n, do: EMLX.eval(out.data.ref)
        t1 = System.monotonic_time(:microsecond)
        (t1 - t0) / n
      end)

    # Fresh eval (actually computes on GPU)
    fresh_eval =
      (for _ <- 1..@warmup, do: ffn1.(x, w1) |> then(& EMLX.eval(&1.data.ref))) |> then(fn _ ->
        t0 = System.monotonic_time(:microsecond)
        for _ <- 1..@iters, do: ffn1.(x, w1) |> then(& EMLX.eval(&1.data.ref))
        t1 = System.monotonic_time(:microsecond)
        (t1 - t0) / @iters
      end)

    IO.puts("    NIF dispatch only (no eval) : #{Float.round(nif_dispatch_only, 1)} μs")
    IO.puts("    graph_replay only (no eval) : #{Float.round(graph_replay_only, 1)} μs")
    IO.puts("    eval (cached/no-op)         : #{Float.round(eval_only, 1)} μs")
    IO.puts("    dispatch + fresh eval       : #{Float.round(fresh_eval, 1)} μs")
    IO.puts("    GPU time (approx)           : #{Float.round(fresh_eval - nif_dispatch_only, 1)} μs")
    IO.puts("    NIF share of total          : #{Float.round(100 * nif_dispatch_only / fresh_eval, 1)}%")
  end

  # ── Main ──────────────────────────────────────────────────────────────────

  def run do
    IO.puts("""
    ╔══════════════════════════════════════════════════════════════════╗
      C03  graph_capture / graph_replay  benchmark  (#{@iters} iters, #{@warmup} warmup)
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    bench_overhead()
    bench_elementwise_chain()
    bench_ffn_block()
    bench_ffn_stack()
    bench_svd_512()
    bench_svd_1024()

    IO.puts("\nDone.")
  end
end

MxCompileBench.run()
