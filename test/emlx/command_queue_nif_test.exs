defmodule EMLX.CommandQueueNIFTest do
  # Unit tests for the worker / command_queue NIFs introduced in
  # clean-room-import/01-worker-thread-dispatch.md. These exercise the
  # C++ primitives directly via EMLX.NIF before the Elixir-side
  # EMLX.CommandQueue / EMLX.Application wrappers are exposed.
  #
  # async: true is fine — every test allocates its own worker (no
  # shared global state).
  use ExUnit.Case, async: true

  alias EMLX.NIF

  defp ok!({:ok, value}), do: value
  defp ok!(:ok), do: :ok

  describe "command_queue_new/1" do
    test "returns a reference for :gpu" do
      ref = NIF.command_queue_new(:gpu) |> ok!()
      assert is_reference(ref)
    end

    test "returns a reference for :cpu" do
      ref = NIF.command_queue_new(:cpu) |> ok!()
      assert is_reference(ref)
    end

    test "rejects unknown device atoms" do
      assert {:error, _} = NIF.command_queue_new(:tpu)
    end

    test "two queues yield distinct refs" do
      a = NIF.command_queue_new(:gpu) |> ok!()
      b = NIF.command_queue_new(:gpu) |> ok!()
      assert a != b
    end
  end

  describe "command_queue_synchronize/1" do
    test "round-trip on an idle queue is prompt" do
      queue = NIF.command_queue_new(:gpu) |> ok!()
      job_ref = NIF.command_queue_synchronize(queue) |> ok!()
      assert is_reference(job_ref)

      assert_receive {^job_ref, {:ok, :ok}}, 1_000
    end

    test "sequential synchronize calls each get their own reply" do
      queue = NIF.command_queue_new(:gpu) |> ok!()

      refs =
        for _ <- 1..5 do
          NIF.command_queue_synchronize(queue) |> ok!()
        end

      for ref <- refs do
        assert_receive {^ref, {:ok, :ok}}, 1_000
      end
    end

    test "rejects an invalid queue ref" do
      bogus = make_ref()
      assert {:error, _} = NIF.command_queue_synchronize(bogus)
    end
  end

  # All graph-touching NIFs are now async-routed via emlx::async_dispatch.
  # The smoke tests below validate the routing for a representative few:
  # eval, to_blob, ones (creation), add (binary), reshape (manipulation).
  # The full Elixir API is exercised by the rest of the suite.
  describe "graph-touching NIFs route through worker (smoke)" do
    test "ones/4 + eval/2 on a private queue" do
      queue = NIF.command_queue_new(:gpu) |> ok!()

      tensor_ref = NIF.ones(queue, {4, 4}, :float32, :gpu) |> ok!()
      assert is_reference(tensor_ref)

      assert_receive {^tensor_ref, {:ok, ref}}, 5_000
      assert is_reference(ref)

      eval_ref = NIF.eval(queue, ref) |> ok!()
      assert_receive {^eval_ref, :ok}, 5_000
    end

    test "add/4 produces a tensor whose to_blob/2 matches Nx" do
      queue = NIF.command_queue_new(:gpu) |> ok!()

      a_job = NIF.ones(queue, {4}, :float32, :gpu) |> ok!()
      assert_receive {^a_job, {:ok, a}}
      b_job = NIF.ones(queue, {4}, :float32, :gpu) |> ok!()
      assert_receive {^b_job, {:ok, b}}

      sum_job = NIF.add(queue, a, b, :gpu) |> ok!()
      assert_receive {^sum_job, {:ok, sum_ref}}, 5_000

      eval_job = NIF.eval(queue, sum_ref) |> ok!()
      assert_receive {^eval_job, :ok}, 5_000

      blob_job = NIF.to_blob(queue, sum_ref) |> ok!()
      assert_receive {^blob_job, {:ok, blob}}, 5_000

      reference = Nx.broadcast(2.0, {4}) |> Nx.as_type(:f32) |> Nx.to_binary()
      assert blob == reference
    end

    test "rejects an invalid worker ref" do
      bogus = make_ref()
      assert {:error, _} = NIF.eval(bogus, make_ref())
    end
  end

  describe "ordering and concurrency" do
    test "jobs posted in order to one queue reply in FIFO order" do
      queue = NIF.command_queue_new(:gpu) |> ok!()

      refs =
        for _ <- 1..20 do
          NIF.command_queue_synchronize(queue) |> ok!()
        end

      # The worker is single-threaded; replies are emitted in the
      # order jobs ran, which equals the order they were posted.
      received =
        for _ <- refs do
          receive do
            {ref, {:ok, :ok}} -> ref
          after
            2_000 -> flunk("Timed out awaiting worker reply")
          end
        end

      assert received == refs
    end

    test "two queues handle interleaved load without crashing" do
      # Each task owns its own queue and uses ONLY that queue for both
      # graph construction and eval — this is the critical invariant in
      # MLX 0.31.2 (encoders are thread-local). Sharing a lazy tensor
      # across queues would cross threads and crash.
      run_on_own_queue = fn ->
        q = NIF.command_queue_new(:gpu) |> ok!()

        a_job = NIF.ones(q, {64, 64}, :float32, :gpu) |> ok!()
        assert_receive {^a_job, {:ok, a}}, 5_000
        b_job = NIF.ones(q, {64, 64}, :float32, :gpu) |> ok!()
        assert_receive {^b_job, {:ok, b}}, 5_000

        prod_job = NIF.add(q, a, b, :gpu) |> ok!()
        assert_receive {^prod_job, {:ok, prod}}, 5_000

        eval_ref = NIF.eval(q, prod) |> ok!()

        receive do
          {^eval_ref, :ok} -> :ok
        after
          5_000 -> flunk("Timed out awaiting eval")
        end
      end

      tasks = for _ <- 1..6, do: Task.async(run_on_own_queue)

      for task <- tasks do
        assert :ok = Task.await(task, 10_000)
      end
    end
  end

  describe "garbage collection" do
    test "dropping the queue ref triggers thread join without crashing" do
      _queue = NIF.command_queue_new(:gpu) |> ok!()
      :erlang.garbage_collect(self())
      # If the destructor double-frees or fails to join, BEAM crashes
      # before this assertion runs.
      assert true
    end
  end
end
