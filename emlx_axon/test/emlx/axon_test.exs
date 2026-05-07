defmodule EMLX.AxonTest do
  @moduledoc """
  Tests for `EMLX.Axon.rewrite/1`.

  ## Tag guide
  - no tag   — pure graph-structure tests; no GPU, no model load needed
  - `:metal` — runs the rewritten model and checks numerical output on GPU
  """
  use ExUnit.Case, async: true

  alias EMLX.Axon, as: EAxon

  # ── Helper: build a standalone rms_norm node matching Bumblebee's layout ──

  # Bumblebee builds:  Axon.layer(impl, [input, weight], op_name: :rms_norm, epsilon: eps, shift: shift)
  defp rms_norm_axon(eps \\ 1.0e-6, shift \\ 0.0) do
    x = Axon.input("x", shape: {nil, 4, 8})

    # Simulate Bumblebee.Layers.rms_norm — weight is an Axon.param (not a parent
    # node), stored in node.parameters, with op_name: :rms_norm in node opts.
    weight = Axon.param("weight", fn input_shape ->
      {elem(input_shape, Nx.rank(input_shape) - 1)}
    end, initializer: :ones)

    Axon.layer(
      fn x, w, opts ->
        eps_val = opts[:epsilon]
        shift_val = opts[:shift]
        variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
        x |> Nx.multiply(Nx.rsqrt(Nx.add(variance, eps_val))) |> Nx.multiply(Nx.add(shift_val, w))
      end,
      [x, weight],
      op_name: :rms_norm,
      epsilon: eps,
      shift: shift
    )
  end

  # ── Structure tests (no GPU required) ────────────────────────────────────────

  describe "rewrite/1 – graph structure" do
    test "rewrites rms_norm node to fast_rms_norm" do
      model = rms_norm_axon()
      rewritten = EAxon.rewrite(model)

      # Traverse nodes and confirm at least one :fast_rms_norm op_name exists
      %Axon{nodes: nodes} = rewritten
      op_names = nodes |> Map.values() |> Enum.map(& &1.op_name)

      assert :fast_rms_norm in op_names
      refute :rms_norm in op_names
    end

    test "does NOT rewrite rms_norm with non-zero shift" do
      model = rms_norm_axon(1.0e-6, 0.5)
      rewritten = EAxon.rewrite(model)

      %Axon{nodes: nodes} = rewritten
      op_names = nodes |> Map.values() |> Enum.map(& &1.op_name)

      # Shift ≠ 0 must be left alone
      assert :rms_norm in op_names
      refute :fast_rms_norm in op_names
    end

    test "rewrite/2 with only: [] skips all rewrites" do
      model = rms_norm_axon()
      rewritten = EAxon.rewrite(model, only: [])

      %Axon{nodes: nodes} = rewritten
      op_names = nodes |> Map.values() |> Enum.map(& &1.op_name)

      assert :rms_norm in op_names
      refute :fast_rms_norm in op_names
    end

    test "rewrite/2 with only: [:rms_norm] is equivalent to rewrite/1" do
      model = rms_norm_axon()
      default_result = EAxon.rewrite(model)
      explicit_result = EAxon.rewrite(model, only: [:rms_norm])

      %Axon{nodes: n1} = default_result
      %Axon{nodes: n2} = explicit_result

      assert map_size(n1) == map_size(n2)

      op_names1 = n1 |> Map.values() |> Enum.map(& &1.op_name) |> Enum.sort()
      op_names2 = n2 |> Map.values() |> Enum.map(& &1.op_name) |> Enum.sort()
      assert op_names1 == op_names2
    end

    test "model with no rms_norm nodes is returned unchanged" do
      model = Axon.input("x", shape: {nil, 8}) |> Axon.dense(4)
      rewritten = EAxon.rewrite(model)

      %Axon{nodes: orig_nodes} = model
      %Axon{nodes: new_nodes} = rewritten

      orig_op_names = orig_nodes |> Map.values() |> Enum.map(& &1.op_name) |> Enum.sort()
      new_op_names = new_nodes |> Map.values() |> Enum.map(& &1.op_name) |> Enum.sort()

      assert orig_op_names == new_op_names
    end
  end

  # ── Numerical tests (GPU required) ───────────────────────────────────────────

  describe "rewrite/1 – numerical output" do
    @moduletag :metal

    # Forces EMLX GPU backend for the duration of the test.
    setup do
      prev = Application.get_env(:nx, :default_backend, Nx.BinaryBackend)
      Application.put_env(:nx, :default_backend, {EMLX.Backend, device: :gpu})
      on_exit(fn -> Application.put_env(:nx, :default_backend, prev) end)
      :ok
    end

    defp build_and_init(model, input_template) do
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(input_template, %{})
      {predict_fn, params}
    end

    test "rewritten rms_norm produces numerically close output to the primitive" do
      eps = 1.0e-5
      x_val = Nx.iota({1, 4, 16}) |> Nx.divide(100) |> Nx.as_type(:f32)

      input_template = %{"x" => Nx.template({1, 4, 16}, :f32)}

      # Original model (primitive rms_norm) — weight initialised to :ones
      {orig_predict, orig_params} = build_and_init(rms_norm_axon(eps), input_template)
      orig_out = orig_predict.(orig_params, %{"x" => x_val})

      # Rewritten model (EMLX.Fast.rms_norm) — same :ones initialiser
      {fast_predict, fast_params} = build_and_init(EAxon.rewrite(rms_norm_axon(eps)), input_template)
      fast_out = fast_predict.(fast_params, %{"x" => x_val})

      orig_f32 = Nx.backend_transfer(orig_out) |> Nx.as_type(:f32)
      fast_f32 = Nx.backend_transfer(fast_out) |> Nx.as_type(:f32)

      # Allow 1e-3 absolute tolerance: EMLX.Fast.rms_norm accumulates in f32
      # internally; the primitive op sequence may differ in rounding order.
      assert Nx.all_close(orig_f32, fast_f32, atol: 1.0e-3) |> Nx.to_number() == 1
    end

    test "rewritten model output shape matches original" do
      input_template = %{"x" => Nx.template({2, 6, 16}, :f32)}
      x_val = Nx.iota({2, 6, 16}) |> Nx.divide(100) |> Nx.as_type(:f32)

      {fast_predict, fast_params} =
        build_and_init(EAxon.rewrite(rms_norm_axon(1.0e-6)), input_template)

      fast_out = fast_predict.(fast_params, %{"x" => x_val})

      assert Nx.shape(fast_out) == {2, 6, 16}
    end
  end
end
