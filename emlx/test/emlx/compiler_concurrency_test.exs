defmodule EMLX.CompilerConcurrencyTest do
  use ExUnit.Case, async: false
  import Nx.Defn

  @iterations 100

  # The compiler wraps the graph it builds in an `mlx::core::StreamContext`, which
  # sets MLX's default device. A :while rebuilds its body once per iteration, on its
  # own worker thread. If a :cpu and a :gpu worker build at the same moment and the
  # default device is shared, one program's ops land on the other's device.
  #
  # The barrier makes both workers finish iteration k, wait for each other, and start
  # building iteration k + 1 together. The CPU and GPU `Nx.dot` kernels round
  # differently here, so equal bits against an eager run prove each program stayed
  # on its own device.
  def barrier(x, opts) do
    coord = Keyword.fetch!(opts, :coord)
    send(coord, {:arrive, self()})

    receive do
      :go -> x
    end
  end

  defn loop(acc, w, opts \\ []) do
    opts = keyword!(opts, [:coord])

    {acc, _, _} =
      while {acc, w, i = 0}, Nx.less(i, @iterations) do
        acc =
          Nx.runtime_call(
            Nx.template(acc.shape, acc.type),
            acc,
            [coord: opts[:coord]],
            &barrier/2
          )

        {Nx.dot(acc, w), w, i + 1}
      end

    acc
  end

  defp coordinator(parties) do
    spawn_link(fn -> coordinate(parties, []) end)
  end

  defp coordinate(parties, waiting) do
    receive do
      {:arrive, pid} ->
        waiting = [pid | waiting]

        if length(waiting) == parties do
          Enum.each(waiting, &send(&1, :go))
          coordinate(parties, [])
        else
          coordinate(parties, waiting)
        end
    end
  end

  defp inputs(device) do
    key = Nx.Random.key(7)
    {acc, key} = Nx.Random.normal(key, shape: {256, 512}, type: :f32)
    {w, _key} = Nx.Random.uniform(key, -0.05, 0.05, shape: {512, 512}, type: :f32)

    Enum.map([acc, w], &Nx.backend_transfer(&1, {EMLX.Backend, device: device}))
  end

  # Eager step-by-step reference on one device, no compiler involved.
  defp eager(device) do
    [acc, w] = inputs(device)

    Enum.reduce(1..@iterations, acc, fn _, acc -> Nx.dot(acc, w) end)
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_binary()
  end

  defp compiled(device, coord) do
    [acc, w] = inputs(device)

    Nx.Defn.jit(&loop(&1, &2, coord: coord), compiler: EMLX, device: device).(acc, w)
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_binary()
  end

  @tag :metal
  test "a :cpu and a :gpu program building their loop bodies together stay on their devices" do
    cpu = eager(:cpu)
    gpu = eager(:gpu)
    refute cpu == gpu, "CPU and GPU results must differ for this test to prove anything"

    # Control: each program alone must already match its eager run.
    assert compiled(:cpu, coordinator(1)) == cpu
    assert compiled(:gpu, coordinator(1)) == gpu

    for _round <- 1..3 do
      coord = coordinator(2)

      tasks = [
        {:cpu, Task.async(fn -> compiled(:cpu, coord) end)},
        {:gpu, Task.async(fn -> compiled(:gpu, coord) end)}
      ]

      results = Map.new(tasks, fn {device, task} -> {device, Task.await(task, 60_000)} end)

      assert results.cpu == cpu
      assert results.gpu == gpu
    end
  end
end
