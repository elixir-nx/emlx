Application.ensure_all_started(:emlx)

# Stage 32a Procedure #3 — regression probe for the bug the thread-local
# caller-pid redesign fixes: a compiled program is traced ONCE (building
# the HostCallback Primitive object once) but replayed many times, possibly
# by DIFFERENT Erlang processes (e.g. a pooled decode loop). Confirms the
# mid-eval {:emlx_host_callback, ...} message routes to whichever process
# ACTUALLY called eval_program for a given replay, not whichever process
# happened to trigger the compiled program's first evaluation.
import Bitwise

defmodule MultiCallerProbe do
  def await(job_ref, callback_queue) do
    receive do
      {:emlx_host_callback, call_id, _callback_slot, operands} ->
        [{ref, shape, :float32}] = operands
        template = Nx.template(List.to_tuple(shape), {:f, 32})

        reply_tensor =
          EMLX.CommandQueue.with_queue(callback_queue, fn ->
            operand_tensor = EMLX.Backend.to_nx({:cpu, ref}, template)
            Nx.multiply(operand_tensor, 2)
          end)

        %Nx.Tensor{data: %EMLX.Backend{ref: {:cpu, reply_ref}}} = reply_tensor

        # host_callback_resume/2 does NOT evaluate the reply itself (dirty
        # NIF, arbitrary OS thread) -- force it on callback_queue's own
        # thread first (mirrors EMLX.dispatch_host_callback/5).
        {:ok, callback_eval_ref} = EMLX.NIF.eval(callback_queue.ref, reply_ref)
        :ok = await(callback_eval_ref, callback_queue)
        :ok = EMLX.NIF.host_callback_resume(call_id, reply_ref)
        await(job_ref, callback_queue)

      {^job_ref, :ok} ->
        :ok

      {^job_ref, {:ok, result}} ->
        result
    after
      20_000 -> raise("job timed out -- callback likely routed to the wrong pid")
    end
  end
end

worker = EMLX.Application.default_worker(:cpu)
{:ok, callback_queue} = EMLX.CommandQueue.new(:cpu)

kind_input = 0
kind_instr = 3
kind_shift = 60
dtype_float32 = 11

instr_output_ref = kind_instr <<< kind_shift ||| 0
op_names = [:host_callback]
operands = [[kind_input <<< kind_shift ||| 0]]
iattrs = [[0, dtype_float32, 1, 3]]
output_refs = [instr_output_ref]

{:ok, compile_job_ref} =
  EMLX.NIF.compile_program(worker, 1, [], [], [], op_names, operands, iattrs, output_refs)

program_ref = MultiCallerProbe.await(compile_job_ref, callback_queue)

run_eval = fn tag ->
  input_tensor =
    Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: {EMLX.Backend, device: :cpu})

  %Nx.Tensor{data: %EMLX.Backend{ref: {:cpu, input_ref}}} = input_tensor

  {:ok, eval_job_ref} = EMLX.NIF.eval_program(worker, program_ref, [input_ref])
  [out_ref] = MultiCallerProbe.await(eval_job_ref, callback_queue)

  {:ok, eval_job_ref2} = EMLX.NIF.eval(worker, out_ref)
  :ok = MultiCallerProbe.await(eval_job_ref2, callback_queue)
  {:ok, blob_job_ref} = EMLX.NIF.to_blob(worker, out_ref)
  binary = MultiCallerProbe.await(blob_job_ref, callback_queue)
  result = Nx.from_binary(binary, {:f, 32})
  IO.inspect({tag, result}, label: "result")
  result
end

# First real evaluation traces + runs the compiled program from THIS
# process (self()). Before the fix, this would have baked self() into the
# HostCallback primitive forever.
result1 = run_eval.("caller A (this process)")

# Second evaluation of the SAME compiled program, but dispatched from a
# freshly spawned, unrelated process. Before the fix, the C++ side would
# have tried to enif_send the callback message to caller A's pid (which
# never sent host_callback_resume for THIS call_id), and this would hang
# until host_round_trip's 30s timeout. After the fix it must route to the
# new process and complete promptly.
parent = self()

spawn(fn ->
  result2 = run_eval.("caller B (spawned process)")
  send(parent, {:done, result2})
end)

result2 =
  receive do
    {:done, r} -> r
  after
    25_000 -> raise("caller B's eval_program never completed -- callback routing bug")
  end

expected = Nx.tensor([2.0, 4.0, 6.0], type: :f32)
ok1? = Nx.equal(result1, expected) |> Nx.all() |> Nx.to_number() == 1
ok2? = Nx.equal(result2, expected) |> Nx.all() |> Nx.to_number() == 1

IO.puts("caller A correct: #{ok1?}, caller B correct: #{ok2?}")
if !(ok1? and ok2?), do: raise("multi-caller host_callback routing test FAILED")
IO.puts("\nProcedure #3 multi-caller routing probe complete: PASS")
