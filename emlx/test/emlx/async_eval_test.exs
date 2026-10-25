defmodule EMLX.AsyncEvalTest do
  use EMLX.Case, async: true

  defp step(x, w), do: x |> Nx.dot(w) |> Nx.divide(16) |> Nx.add(1)

  defp ref(tensor), do: EMLX.Backend.from_nx(tensor)

  for device <- [:cpu, :gpu] do
    describe "#{device}" do
      if device == :gpu, do: @describetag(:metal)

      setup do
        backend = {EMLX.Backend, device: unquote(device)}
        x = Nx.iota({64, 64}, type: :f32, backend: backend) |> Nx.divide(4096)
        w = Nx.iota({64, 64}, type: :f32, backend: backend) |> Nx.remainder(7) |> Nx.divide(7)

        expected =
          x |> Nx.backend_copy(Nx.BinaryBackend) |> step(Nx.backend_copy(w, Nx.BinaryBackend))

        %{backend: backend, x: x, w: w, expected: expected}
      end

      test "a compiled call takes lazy inputs and the lazy outputs of another call", ctx do
        compiled =
          Nx.Defn.compile(&step/2, [ctx.x, ctx.w], compiler: EMLX, device: unquote(device))

        lazy_x = ctx.x |> Nx.multiply(2) |> Nx.divide(2)
        first = compiled.(lazy_x, ctx.w)
        second = compiled.(first, ctx.w)

        expected_second = step(ctx.expected, Nx.backend_copy(ctx.w, Nx.BinaryBackend))
        assert_all_close(Nx.backend_copy(first, Nx.BinaryBackend), ctx.expected)
        assert_all_close(Nx.backend_copy(second, Nx.BinaryBackend), expected_second)
      end

      test "async_eval and then a read give the values of the tensors", ctx do
        a = step(ctx.x, ctx.w)
        b = Nx.multiply(a, 3)

        assert EMLX.async_eval([ref(a), ref(b)]) == :ok
        assert_all_close(Nx.backend_copy(a, Nx.BinaryBackend), ctx.expected)
        assert_all_close(Nx.backend_copy(b, Nx.BinaryBackend), Nx.multiply(ctx.expected, 3))
      end

      test "a read right after eval has the values while later work runs", ctx do
        a = step(ctx.x, ctx.w)
        EMLX.async_eval(ref(a))

        later = Enum.reduce(1..20, a, fn _, acc -> step(acc, ctx.w) end)
        EMLX.async_eval(ref(later))

        EMLX.eval(ref(a))
        assert_all_close(Nx.backend_copy(a, Nx.BinaryBackend), ctx.expected)
        assert later |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      end

      test "a pipelined loop reads each step after it schedules the next one", ctx do
        compiled =
          Nx.Defn.compile(&step/2, [ctx.x, ctx.w], compiler: EMLX, device: unquote(device))

        host_w = Nx.backend_copy(ctx.w, Nx.BinaryBackend)

        expected =
          Enum.map_reduce(1..8, Nx.backend_copy(ctx.x, Nx.BinaryBackend), fn _, acc ->
            next = step(acc, host_w)
            {Nx.sum(next) |> Nx.to_number(), next}
          end)
          |> elem(0)

        {sums, last} =
          Enum.map_reduce(1..8, ctx.x, fn i, previous ->
            next = compiled.(previous, ctx.w)
            EMLX.async_eval(ref(next))
            sum = if i > 1, do: [Nx.sum(previous) |> Nx.to_number()], else: []
            {sum, next}
          end)

        sums = List.flatten(sums) ++ [Nx.sum(last) |> Nx.to_number()]

        for {got, want} <- Enum.zip(sums, expected) do
          assert_in_delta got, want, abs(want) * 1.0e-3
        end
      end
    end
  end
end
