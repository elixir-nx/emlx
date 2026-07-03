Application.ensure_all_started(:emlx)

# Stage 32a Procedure #2 — smoke test for the production ":host_callback"
# opcode (c_src/emlx_compiler.cpp's `host_callback` namespace / op_registry
# entry), driven through the REAL compile_program/eval_program NIF path
# (not spike32a's standalone run_program helper). Not a permanent test; run
# manually with `mix run bench/host_callback_opcode.exs`.
#
# Confirms the opcode's actual wire-format contract end-to-end:
#   - compile_program accepts a hand-built "host_callback" instruction
#   - eval_program's replay sends {:emlx_host_callback, call_id,
#     callback_slot, [{ref, shape, dtype}]} to the CURRENT calling process
#     (emlx::current_caller_pid(), not a registered pid -- see Stage 32a
#     Procedure #3's redesign in the stage doc) for every operand
#     (self-describing -- no worker-routed NIF call needed to interpret it,
#     since the worker thread that would service such a call is the one
#     blocked)
#   - host_callback_resume/2 unblocks it with a real Nx-computed reply
#     tensor, and eval_program returns that reply as the program's output
#
# Callback body here: reply = 2 * operand (elementwise), to prove a
# multi-element (not just scalar) tensor round-trips correctly -- the
# realistic shape for e.g. an attention callback's tensor operands, unlike
# spike32a's single-scalar probe.
import Bitwise

defmodule HostCallbackOpcodeTest do
  # Mirrors emlx.ex's private await_worker/1: worker NIFs reply
  # {job_ref, {:ok, result}} / {job_ref, {:error, reason}}. Also services
  # the mid-eval {:emlx_host_callback, ...} message when it arrives instead.
  def await(job_ref, callback_queue) do
    receive do
      {:emlx_host_callback, call_id, callback_slot, operands} ->
        IO.inspect({call_id, callback_slot, length(operands)}, label: "got callback")

        [{ref, shape, :float32}] = operands
        template = Nx.template(List.to_tuple(shape), {:f, 32})

        reply_tensor =
          EMLX.CommandQueue.with_queue(callback_queue, fn ->
            operand_tensor = EMLX.Backend.to_nx({:cpu, ref}, template)
            Nx.multiply(operand_tensor, 2)
          end)

        %Nx.Tensor{data: %EMLX.Backend{ref: {:cpu, reply_ref}}} = reply_tensor

        # host_callback_resume/2 is a dirty NIF (arbitrary OS thread, no MLX
        # stream of its own) and does NOT evaluate the reply itself -- force
        # it here, on callback_queue's own thread, first (see the C++
        # resume()'s comment; mirrors EMLX.dispatch_host_callback/5).
        {:ok, callback_eval_ref} = EMLX.NIF.eval(callback_queue.ref, reply_ref)
        :ok = await(callback_eval_ref, callback_queue)
        :ok = EMLX.NIF.host_callback_resume(call_id, reply_ref)
        await(job_ref, callback_queue)

      {^job_ref, :ok} ->
        :ok

      {^job_ref, {:ok, result}} ->
        result

      {^job_ref, {:error, reason}} ->
        raise("job failed: #{List.to_string(reason)}")
    after
      20_000 -> raise("job timed out")
    end
  end
end

worker = EMLX.Application.default_worker(:cpu)

# The callback handler below computes with real Nx/EMLX ops (2 * operand),
# the realistic shape for a real runtime_call callback (e.g.
# native_kv_attn_callback). It deliberately does NOT use the default :cpu
# worker: that worker is the one BLOCKED inside host_round_trip while this
# message is in flight, so any Nx op that routed to it would queue behind
# the block and self-deadlock (recoverable only via host_round_trip's 30s
# timeout, observed empirically while writing this test -- see the stage
# doc's Results). A dedicated CommandQueue (its own worker OS thread) lets
# the callback's own Nx computation proceed independently. Procedure #3/#6
# must give the real callback dispatcher the same property.
{:ok, callback_queue} = EMLX.CommandQueue.new(:cpu)

# No registration step: eval_program's C++ side routes the mid-eval
# message to emlx::current_caller_pid() -- whichever process actually
# dispatched THIS eval_program call (this process, since we call it
# directly below) -- see Stage 32a Procedure #3.
callback_slot = 0

# ── Hand-build the wire format (normally EMLX.Native.Expr.to_wire/1's job) ──
kind_input = 0
kind_instr = 3
kind_shift = 60
dtype_float32 = 11

input_tensor = Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: {EMLX.Backend, device: :cpu})
%Nx.Tensor{data: %EMLX.Backend{ref: {:cpu, input_ref}}} = input_tensor

instr_output_ref = kind_instr <<< kind_shift ||| 0
op_names = [:host_callback]
operands = [[kind_input <<< kind_shift ||| 0]]
# attrs = [callback_slot, dtype_int, n_dims, d0]
iattrs = [[callback_slot, dtype_float32, 1, 3]]
output_refs = [instr_output_ref]

{:ok, compile_job_ref} =
  EMLX.NIF.compile_program(worker, 1, [], [], [], op_names, operands, iattrs, output_refs)

program_ref = HostCallbackOpcodeTest.await(compile_job_ref, callback_queue)
IO.inspect(program_ref, label: "compiled program")

{:ok, eval_job_ref} = EMLX.NIF.eval_program(worker, program_ref, [input_ref])
IO.puts("dispatched eval_program")
[out_ref] = HostCallbackOpcodeTest.await(eval_job_ref, callback_queue)
IO.puts("got eval_program result: #{inspect(out_ref)}")

# eval_program defers materialization to the caller (its output refs are
# still lazy graph nodes) -- force it via eval + to_blob ourselves,
# serviced by the SAME await/1 loop, so we're still listening for the
# mid-eval {:emlx_host_callback, ...} message when materialization
# actually drives the compiled graph's Primitive::eval_cpu (unlike
# EMLX.to_blob/1, which internally awaits eval's and to_blob's replies on
# its own private receive, leaving our callback message unhandled in the
# mailbox until timeout).
{:ok, eval_job_ref2} = EMLX.NIF.eval(worker, out_ref)
:ok = HostCallbackOpcodeTest.await(eval_job_ref2, callback_queue)
{:ok, blob_job_ref} = EMLX.NIF.to_blob(worker, out_ref)
binary = HostCallbackOpcodeTest.await(blob_job_ref, callback_queue)

result = Nx.from_binary(binary, {:f, 32})
IO.inspect(result, label: "host_callback opcode result (expect [2.0, 4.0, 6.0])")

expected = Nx.tensor([2.0, 4.0, 6.0], type: :f32)
ok? = Nx.equal(result, expected) |> Nx.all() |> Nx.to_number() == 1
IO.puts("result correct: #{ok?}")

if !ok?, do: raise("host_callback opcode smoke test FAILED")
IO.puts("\nProcedure #2 smoke test complete.")
