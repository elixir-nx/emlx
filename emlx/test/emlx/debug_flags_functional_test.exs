defmodule EMLX.DebugFlagsFunctionalTest do
  use EMLX.Case, async: false

  @moduledoc """
  Functional (raise-on-violation) counterpart to `debug_flags_test.exs`'s
  zero-cost-when-off opcode checks.

  `:enable_bounds_check`, `:detect_non_finites`, and `:compiler_debug` are all
  `Application.compile_env/3`-gated, so none can be toggled at runtime —
  these tests only pass against a build compiled with all three flags on. Run:

      EMLX_DEBUG_FLAGS=1 mix test --force --include debug_flags_functional

  Excluded by default (see `test/test_helper.exs`) so a normal `mix test`
  run — compiled with every debug flag off, per production defaults —
  doesn't fail here.
  """

  @moduletag :debug_flags_functional

  alias EMLX.Native.Expr

  setup_all do
    # `Application.get_env/3` here (not `compile_env/3`): the flags below are
    # genuinely compile-time-baked into the *lib* code under test, but this
    # guard is just a friendly diagnostic — reading via `get_env` avoids the
    # type-checker constant-folding the whole `unless` away (both flags are
    # literal `false` in a default build) into a "warnings-as-errors" hit.
    flags_on? =
      Application.get_env(:emlx, :enable_bounds_check, false) and
        Application.get_env(:emlx, :detect_non_finites, false) and
        Application.get_env(:emlx, :compiler_debug, false)

    unless flags_on? do
      flunk("""
      compiled with a debug flag off — these tests always fail here.
      Run: EMLX_DEBUG_FLAGS=1 mix test --force --include debug_flags_functional
      """)
    end

    :ok
  end

  defp gpu(tensor), do: Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

  test "take raises on an out-of-bounds index (:enable_bounds_check, pre-existing coverage, verified not regressed)" do
    t = Nx.tensor([1, 2, 3], backend: EMLX.Backend)
    idx = Nx.tensor([5], type: :s64, backend: EMLX.Backend)

    assert_raise ArgumentError, ~r/index out of bounds/, fn ->
      Nx.take(t, idx) |> Nx.backend_transfer()
    end
  end

  test "dot raises on a NaN-producing matmul (pre-existing coverage, unmoved by the EMLX.Debug extraction)" do
    nan = Nx.tensor([:nan, 1.0, 1.0], type: :f32)
    ones = Nx.tensor([1.0, 1.0, 1.0], type: :f32)

    assert_raise ArgumentError, ~r/dot produced NaN or Inf/, fn ->
      Nx.dot(nan, ones) |> Nx.backend_transfer()
    end
  end

  test "conv raises on a NaN-producing convolution" do
    input = Nx.tensor([[[[:nan, 1.0, 1.0, 1.0]]]], type: :f32)
    kernel = Nx.tensor([[[[1.0, 1.0]]]], type: :f32)

    assert_raise ArgumentError, ~r/conv produced NaN or Inf/, fn ->
      Nx.conv(input, kernel, padding: :valid) |> Nx.backend_transfer()
    end
  end

  @tag :metal
  test "EMLX.Fast.rms_norm raises on a NaN-producing input" do
    x = Nx.tensor([[:nan, 1.0, 1.0, 1.0]], type: :f32) |> gpu()
    w = Nx.tensor([1.0, 1.0, 1.0, 1.0], type: :f32) |> gpu()

    assert_raise ArgumentError, ~r/rms_norm produced NaN or Inf/, fn ->
      EMLX.Fast.rms_norm(x, w, 1.0e-5)
    end
  end

  test "EMLX.Native.Expr.to_wire raises on a ref id collision across categories (:compiler_debug)" do
    ref = make_ref()

    prog = %Expr{
      inputs: [ref],
      captures: [{ref, Nx.tensor(1.0)}],
      constants: [],
      instructions: [],
      outputs: [ref]
    }

    assert_raise ArgumentError, ~r/ref id collision across inputs\/captures\/constants/, fn ->
      Expr.to_wire(prog)
    end
  end

  test "EMLX.Native.Expr.to_wire raises when an instruction's result ref is already bound (:compiler_debug)" do
    ref = make_ref()

    prog = %Expr{
      inputs: [ref],
      captures: [],
      constants: [],
      instructions: [{ref, :add, [], []}],
      outputs: [ref]
    }

    assert_raise ArgumentError, ~r/that ref is already bound/, fn ->
      Expr.to_wire(prog)
    end
  end
end
