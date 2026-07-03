Application.ensure_all_started(:emlx)

# Stage 32a spike — load-bearing-unknown probe (Procedure #1 in
# workdir/native-compiler/32a-inline-runtime-call.md). Not a permanent test;
# run manually with `mix run bench/spike32a_host_callback.exs`.
#
# Confirms/denies: can the worker OS thread executing a compiled program's
# replay safely call back into Erlang and block on a reply, without
# deadlocking EMLX's ASYNC_NIF/enif_send worker-queue dispatch or corrupting
# in-flight Metal state, on both :cpu and :gpu, and does the compile cache
# still invoke the callback on every real eval (not just the first trace)?
#
# The compiled program returns TWO outputs: `z = 2 * HostCallback(x) + x`
# (the callback's output consumed by a real downstream op on the graph's
# own stream -- :gpu when requested -- in the SAME compiled program, not
# just returned raw) and `w = x + 100` (a second, independent graph_stream
# op sharing the program but NOT depending on the callback -- reading it
# back correctly after the callback's blocking round trip is the
# encoder/stream-survival check). With the default reply_fn (&(&1*2.0)):
# reply = 2*x, z = 2*reply + x = 5*x.

defmodule Spike32a do
  # Drives one `spike32a_run` call end-to-end, single-process: dispatches the
  # worker-routed NIF (non-blocking, returns a job_ref immediately), then a
  # single selective `receive` loop fields whichever arrives first out of
  # (a) the mid-eval {:spike32a_callback, call_id, value} message (replied to
  # via spike32a_resume/2, bypassing the worker's own queue on purpose) or
  # (b) the job's own {job_ref, payload} completion reply.
  def run(worker, device, input_value, compile_id, reply_fn \\ &(&1 * 2.0)) do
    self_pid = self()

    job_ref =
      case EMLX.NIF.spike32a_run(worker, self_pid, device, input_value, compile_id) do
        {:ok, ref} -> ref
        {:error, reason} -> raise("spike32a_run failed to dispatch: #{inspect(reason)}")
      end

    await_loop(job_ref, reply_fn)
  end

  defp await_loop(job_ref, reply_fn) do
    receive do
      {:spike32a_callback, call_id, value} ->
        reply = reply_fn.(value)
        :ok = EMLX.NIF.spike32a_resume(call_id, reply)
        await_loop(job_ref, reply_fn)

      {^job_ref, {:ok, result}} ->
        {:ok, result}

      {^job_ref, {:error, reason}} ->
        {:error, List.to_string(reason)}
    after
      20_000 -> {:error, :job_timeout}
    end
  end
end

worker_cpu = EMLX.Application.default_worker(:cpu)

IO.puts("== Test 1: standalone round trip, :cpu ==")
{:ok, {value, same_thread, trace_count, eval_count, independent}} =
  Spike32a.run(worker_cpu, :cpu, 3.0, 1)

IO.inspect(
  %{
    value: value,
    same_thread: same_thread,
    trace_count: trace_count,
    eval_count: eval_count,
    independent: independent
  },
  label: "cpu call 1 (compile_id=1)"
)

IO.puts(
  "value correct (5*3=15, callback output consumed by a downstream op in the same program): #{value == 15.0}"
)

IO.puts(
  "independent op correct (3+100=103 -- unrelated graph_stream op in the same program, read back after the callback's round trip): #{independent == 103.0}"
)

IO.puts("== Test 1b: same compile_id, second call (does callback still fire? does it re-trace?) ==")
{:ok, {value2, same_thread2, trace_count2, eval_count2, _independent2}} =
  Spike32a.run(worker_cpu, :cpu, 5.0, 1)

IO.inspect(
  %{value: value2, same_thread: same_thread2, trace_count: trace_count2, eval_count: eval_count2},
  label: "cpu call 2 (compile_id=1, same as call 1)"
)

IO.puts("value correct (5*5=25): #{value2 == 25.0}")

if trace_count2 > trace_count do
  IO.puts("!! UNEXPECTED: trace_count grew on the second call with the same compile_id -- compile() re-traced instead of replaying.")
else
  IO.puts("OK: trace_count stayed at #{trace_count2} across 2 calls -- compile() replayed, did not re-trace.")
end

if eval_count2 != eval_count + 1 do
  IO.puts("!! UNEXPECTED: eval_count did not increase by exactly 1 on the second call -- callback did not fire on replay as expected.")
else
  IO.puts("OK: eval_count increased by 1 on the second call -- the host callback re-fires on every real eval, not just the first trace.")
end

IO.puts("\n== Test 2: standalone round trip, :gpu ==")
try do
  worker_gpu = EMLX.Application.default_worker(:gpu)

  {:ok, {value_gpu, same_thread_gpu, trace_count_gpu, eval_count_gpu, independent_gpu}} =
    Spike32a.run(worker_gpu, :gpu, 7.0, 2)

  IO.inspect(
    %{
      value: value_gpu,
      same_thread: same_thread_gpu,
      trace_count: trace_count_gpu,
      eval_count: eval_count_gpu,
      independent: independent_gpu
    },
    label: "gpu call (compile_id=2)"
  )

  IO.puts(
    "value correct (5*7=35 -- 'z = 2*callback_out+x' ran on the real :gpu stream downstream of the callback, inside the one compiled program): #{value_gpu == 35.0}"
  )

  IO.puts(
    "independent GPU-stream op correct (7+100=107 -- confirms the :gpu command-encoder/stream survives the callback's blocking round trip within the same compiled program, not just in a separate later call): #{independent_gpu == 107.0}"
  )

  # Sanity: an ordinary op on the same :gpu worker right after the spike
  # call, to check for corrupted Metal command-encoder/stream state.
  a = Nx.tensor([1, 2, 3], type: :f32, backend: {EMLX.Backend, device: :gpu})
  b = Nx.tensor([4, 5, 6], type: :f32, backend: {EMLX.Backend, device: :gpu})
  sanity = Nx.add(a, b)
  IO.inspect(sanity, label: "post-spike GPU sanity op ([1,2,3]+[4,5,6])")
rescue
  e -> IO.puts("GPU test skipped/failed: #{Exception.message(e)}")
end

IO.puts("\n== Test 3: repeated calls in a row (simulates N structurally-identical layers / decode steps) ==")
results =
  for i <- 1..5 do
    {:ok, {v, _st, tc, ec, _ind}} = Spike32a.run(worker_cpu, :cpu, i * 1.0, 3)
    {v, tc, ec}
  end

IO.inspect(results, label: "5 sequential calls, compile_id=3")

values_ok? = Enum.with_index(results, 1) |> Enum.all?(fn {{v, _tc, _ec}, i} -> v == 5.0 * i end)
trace_ok? = results |> Enum.map(&elem(&1, 1)) |> Enum.uniq() == [1]
eval_ok? = results |> Enum.map(&elem(&1, 2)) == [1, 2, 3, 4, 5]

IO.puts("values correct (reply_fn = &(&1*2)): #{values_ok?}")
IO.puts("trace_count stayed at 1 across all 5 calls: #{trace_ok?}")
IO.puts("eval_count incremented 1..5 (fires every call): #{eval_ok?}")

IO.puts("\nSpike complete.")
