defmodule EMLX.TelemetryTest do
  use EMLX.Case, async: false
  doctest EMLX.Telemetry

  alias EMLX.Telemetry

  setup do
    on_exit(fn ->
      :telemetry.list_handlers([])
      |> Enum.each(&:telemetry.detach(&1.id))
    end)

    :ok
  end

  # Module-captured handler to avoid telemetry's local-fn perf warning.
  def forward_to_test(event, measurements, metadata, %{pid: pid}) do
    send(pid, {:event, event, measurements, metadata})
  end

  defp attach(event, id) do
    :telemetry.attach(id, event, &__MODULE__.forward_to_test/4, %{pid: self()})
  end

  describe "[:emlx, :eval, *]" do
    test "fires start + stop around EMLX.eval/1" do
      attach([:emlx, :eval, :start], "eval-start-#{inspect(self())}")
      attach([:emlx, :eval, :stop], "eval-stop-#{inspect(self())}")

      t = Nx.tensor([1, 2, 3], backend: EMLX.Backend)
      EMLX.eval(EMLX.Backend.from_nx(t))

      assert_receive {:event, [:emlx, :eval, :start], _, %{}}
      assert_receive {:event, [:emlx, :eval, :stop], %{duration: dur}, %{}}
      assert is_integer(dur) and dur >= 0
    end
  end

  describe "[:emlx, :to_binary, *]" do
    test "fires on EMLX.Backend.to_binary/2 (the Nx.to_binary path)" do
      attach([:emlx, :to_binary, :stop], "to-bin-stop-#{inspect(self())}")

      t = Nx.tensor([1.0, 2.0, 3.0], backend: EMLX.Backend)
      _ = Nx.to_binary(t)

      assert_receive {:event, [:emlx, :to_binary, :stop], %{duration: dur},
                      %{shape: shape, dtype: dtype, byte_size: bs}}

      assert shape == {3}
      assert dtype == {:f, 32}
      assert bs == 12
      assert is_integer(dur) and dur >= 0
    end
  end

  describe "[:emlx, :memory, :stats]" do
    test "memory_stats/0 emits active/peak/cache measurements" do
      attach([:emlx, :memory, :stats], "mem-stats-#{inspect(self())}")

      result = Telemetry.memory_stats()

      assert %{active_memory: a, peak_memory: p, cache_memory: c} = result
      assert is_integer(a) and a >= 0
      assert is_integer(p) and p >= 0
      assert is_integer(c) and c >= 0

      assert_receive {:event, [:emlx, :memory, :stats], ^result, _}
    end
  end
end
