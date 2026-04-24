defmodule EMLX.CommandQueueTest do
  # Tests for the EMLX.CommandQueue struct + with_queue/2 wrapper and the
  # resolve_worker/1 dispatch logic (process-dict binding, cross-device
  # promotion, and optional Logger.warning).
  #
  # async: false because several tests temporarily mutate Application env or
  # the process dictionary in ways that could interfere if run concurrently.
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

  alias EMLX.CommandQueue

  # ── CommandQueue.new/1 ─────────────────────────────────────────────────────

  describe "CommandQueue.new/1" do
    test "returns a struct with a reference and the requested device for :gpu" do
      assert {:ok, %CommandQueue{ref: ref, device: :gpu}} = CommandQueue.new(:gpu)
      assert is_reference(ref)
    end

    test "returns a struct with a reference and the requested device for :cpu" do
      assert {:ok, %CommandQueue{ref: ref, device: :cpu}} = CommandQueue.new(:cpu)
      assert is_reference(ref)
    end

    test "returns {:error, _} for unknown device" do
      assert {:error, _} = CommandQueue.new(:tpu)
    end

    test "two queues for the same device yield distinct refs" do
      {:ok, a} = CommandQueue.new(:gpu)
      {:ok, b} = CommandQueue.new(:gpu)
      assert a.ref != b.ref
    end
  end

  # ── CommandQueue.new!/1 ────────────────────────────────────────────────────

  describe "CommandQueue.new!/1" do
    test "returns the struct directly for a valid device" do
      assert %CommandQueue{ref: ref, device: :gpu} = CommandQueue.new!(:gpu)
      assert is_reference(ref)
    end

    test "raises EMLX.NIFError for an unknown device" do
      assert_raise EMLX.NIFError, fn -> CommandQueue.new!(:tpu) end
    end
  end

  # ── CommandQueue.synchronize/1 ─────────────────────────────────────────────

  describe "CommandQueue.synchronize/1" do
    test "returns :ok on a freshly created CPU queue" do
      q = CommandQueue.new!(:cpu)
      assert :ok = CommandQueue.synchronize(q)
    end

    test "returns :ok on a freshly created GPU queue" do
      q = CommandQueue.new!(:gpu)
      assert :ok = CommandQueue.synchronize(q)
    end

    test "returns :ok after enqueuing work via with_queue" do
      q = CommandQueue.new!(:gpu)

      CommandQueue.with_queue(q, fn ->
        Nx.add(Nx.tensor([1, 2, 3], backend: EMLX.Backend), Nx.tensor([4, 5, 6], backend: EMLX.Backend))
      end)

      assert :ok = CommandQueue.synchronize(q)
    end
  end

  # ── CommandQueue.with_queue/2 ──────────────────────────────────────────────

  describe "CommandQueue.with_queue/2" do
    test "sets :emlx_command_queue to {ref, device} for the duration" do
      {:ok, q} = CommandQueue.new(:gpu)
      assert Process.get(:emlx_command_queue) == nil

      CommandQueue.with_queue(q, fn ->
        assert Process.get(:emlx_command_queue) == {q.ref, :gpu}
      end)
    end

    test "restores nil after the block" do
      {:ok, q} = CommandQueue.new(:gpu)
      CommandQueue.with_queue(q, fn -> :ok end)
      assert Process.get(:emlx_command_queue) == nil
    end

    test "restores the previous value when nested" do
      {:ok, outer} = CommandQueue.new(:gpu)
      {:ok, inner} = CommandQueue.new(:cpu)

      CommandQueue.with_queue(outer, fn ->
        assert Process.get(:emlx_command_queue) == {outer.ref, :gpu}

        CommandQueue.with_queue(inner, fn ->
          assert Process.get(:emlx_command_queue) == {inner.ref, :cpu}
        end)

        assert Process.get(:emlx_command_queue) == {outer.ref, :gpu}
      end)

      assert Process.get(:emlx_command_queue) == nil
    end

    test "restores previous value even when the block raises" do
      {:ok, q} = CommandQueue.new(:gpu)

      assert_raise RuntimeError, fn ->
        CommandQueue.with_queue(q, fn -> raise "boom" end)
      end

      assert Process.get(:emlx_command_queue) == nil
    end
  end

  # ── EMLX.resolve_worker/1 ──────────────────────────────────────────────────

  describe "resolve_worker/1 with no bound queue" do
    test "returns the application-default worker for :gpu" do
      Process.delete(:emlx_command_queue)
      {worker, device} = EMLX.resolve_worker(:gpu)
      assert is_reference(worker)
      assert device == :gpu
    end

    test "returns the application-default worker for :cpu" do
      Process.delete(:emlx_command_queue)
      {worker, device} = EMLX.resolve_worker(:cpu)
      assert is_reference(worker)
      assert device == :cpu
    end
  end

  describe "resolve_worker/1 with a matching bound queue" do
    test "returns the bound queue worker and the requested device" do
      {:ok, q} = CommandQueue.new(:gpu)

      CommandQueue.with_queue(q, fn ->
        {worker, device} = EMLX.resolve_worker(:gpu)
        assert worker == q.ref
        assert device == :gpu
      end)
    end

    test "CPU queue is returned for a CPU tensor" do
      {:ok, q} = CommandQueue.new(:cpu)

      CommandQueue.with_queue(q, fn ->
        {worker, device} = EMLX.resolve_worker(:cpu)
        assert worker == q.ref
        assert device == :cpu
      end)
    end
  end

  # ── Cross-device promotion ─────────────────────────────────────────────────

  describe "resolve_worker/1 cross-device promotion disabled (default)" do
    setup do
      Application.delete_env(:emlx, :cross_device_promotion)
      Application.delete_env(:emlx, :warn_cross_device)
      on_exit(fn ->
        Application.delete_env(:emlx, :cross_device_promotion)
        Application.delete_env(:emlx, :warn_cross_device)
      end)
    end

    test "returns the app-default CPU worker when GPU queue is bound but :cpu is requested" do
      {:ok, gpu_q} = CommandQueue.new(:gpu)
      app_cpu_worker = EMLX.Application.default_worker(:cpu)

      CommandQueue.with_queue(gpu_q, fn ->
        {worker, device} = EMLX.resolve_worker(:cpu)
        assert worker == app_cpu_worker
        assert device == :cpu
      end)
    end
  end

  describe "resolve_worker/1 cross-device promotion enabled" do
    setup do
      Application.put_env(:emlx, :cross_device_promotion, true)
      Application.delete_env(:emlx, :warn_cross_device)
      on_exit(fn ->
        Application.delete_env(:emlx, :cross_device_promotion)
        Application.delete_env(:emlx, :warn_cross_device)
      end)
    end

    test "routes a :cpu request through the bound GPU queue" do
      {:ok, gpu_q} = CommandQueue.new(:gpu)

      CommandQueue.with_queue(gpu_q, fn ->
        {worker, device} = EMLX.resolve_worker(:cpu)
        assert worker == gpu_q.ref
        assert device == :gpu
      end)
    end

    test "does not emit a warning when warn_cross_device is false" do
      {:ok, gpu_q} = CommandQueue.new(:gpu)

      log =
        capture_log(fn ->
          CommandQueue.with_queue(gpu_q, fn ->
            EMLX.resolve_worker(:cpu)
          end)
        end)

      refute log =~ "[EMLX] cross-device promotion"
    end

    test "emits a warning when warn_cross_device is true" do
      Application.put_env(:emlx, :warn_cross_device, true)
      {:ok, gpu_q} = CommandQueue.new(:gpu)

      log =
        capture_log(fn ->
          CommandQueue.with_queue(gpu_q, fn ->
            EMLX.resolve_worker(:cpu)
          end)
        end)

      assert log =~ "[EMLX] cross-device promotion"
      assert log =~ "cpu"
      assert log =~ "gpu"
    end
  end
end
