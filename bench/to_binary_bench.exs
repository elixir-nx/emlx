# Benchmark: Nx.to_binary for EMLX tensors
#
# Measures GPU→CPU transfer time for contiguous tensors of various sizes.
# Run: mix run bench/to_binary_bench.exs

IO.puts("Benchmarking Nx.to_binary (EMLX GPU → CPU transfer)\n")

for {rows, cols, label} <- [
      {1, 1536, "1×1536 (single embedding)"},
      {32, 1536, "32×1536 (small batch)"},
      {64, 1536, "64×1536 (medium batch)"},
      {512, 1536, "512×1536 (large batch)"},
      {1024, 1024, "1024×1024 (1M elements)"}
    ] do
  t = Nx.iota({rows, cols}, type: :f32, backend: EMLX.Backend)

  # Force evaluation before benchmarking transfer
  EMLX.eval(EMLX.Backend.from_nx(t))

  # Warm up
  _ = Nx.to_binary(t)

  runs = 10
  times = for _ <- 1..runs do
    t0 = System.monotonic_time(:microsecond)
    _ = Nx.to_binary(t)
    System.monotonic_time(:microsecond) - t0
  end

  avg = Enum.sum(times) / runs
  bytes = rows * cols * 4
  throughput_gbs = bytes / (avg / 1_000_000) / 1_073_741_824

  IO.puts(
    "  #{label}: #{Float.round(avg, 0)} µs avg " <>
      "(#{Float.round(throughput_gbs, 1)} GB/s, #{div(bytes, 1024)} KB)"
  )
end
