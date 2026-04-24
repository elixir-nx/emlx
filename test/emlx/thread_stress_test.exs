defmodule EMLX.ThreadStressTest do
  use EMLX.Case, async: true

  # 16 concurrent BEAM processes × 100 forced-eval matmuls each.
  # Green-light check before task 01 builds the worker model on MLX 0.31.2.
  # Pass criteria: zero crashes, bit-identical outputs across all 16 processes.
  @processes 16
  @iterations 100
  @shape {256, 256}

  @tag :stress_soak
  @tag timeout: 120_000
  test "16-process concurrent matmul stress soak (ThreadLocalStream)" do
    # Fixed input — identical for every process so results must be bit-identical.
    a = Nx.iota(@shape, type: :f32) |> Nx.divide(65_536.0)

    # Each call forces mx::eval via Nx.to_binary/1 (goes through to_blob NIF).
    run = fn ->
      Enum.reduce(1..@iterations, nil, fn _, _ ->
        Nx.dot(a, a) |> Nx.to_binary()
      end)
    end

    # Sequential reference computed before spawning concurrent tasks.
    reference = run.()

    results =
      1..@processes
      |> Task.async_stream(fn _ -> run.() end,
        max_concurrency: @processes,
        timeout: 90_000
      )
      |> Enum.map(fn {:ok, bin} -> bin end)

    assert length(results) == @processes,
           "Expected #{@processes} results, got #{length(results)}"

    for {bin, idx} <- Enum.with_index(results, 1) do
      assert bin == reference,
             "Process #{idx} output differs from reference — possible ThreadLocalStream race"
    end
  end
end
