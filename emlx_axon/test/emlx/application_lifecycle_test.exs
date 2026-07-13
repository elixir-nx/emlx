defmodule EMLXAxon.ApplicationLifecycleTest do
  use ExUnit.Case, async: true

  test "application restart keeps the registered Qwen3 plugin usable" do
    root = Path.expand("../..", __DIR__)
    code_paths = Path.wildcard(Path.join(root, "_build/test/lib/*/ebin"))

    script = """
    {:ok, _} = Application.ensure_all_started(:emlx_axon)
    :ok = EMLXAxon.Application.ensure_qwen3_plugin_loaded!()
    :ok = Application.stop(:emlx_axon)
    {:ok, _} = Application.ensure_all_started(:emlx_axon)
    :ok = EMLXAxon.Application.ensure_qwen3_plugin_loaded!()

    hidden = Nx.tensor([[[1.0]]], backend: EMLX.Backend)
    weight = Nx.tensor([1.0], backend: EMLX.Backend)
    projection = Nx.tensor([[1.0]], backend: EMLX.Backend)

    result =
      EMLXAxon.Qwen3.Native.mlp(
        hidden,
        weight,
        projection,
        projection,
        projection,
        1.0e-5
      )

    IO.inspect({:restart_result, Nx.shape(result), Nx.to_flat_list(result)})
    """

    args = Enum.flat_map(code_paths, &["-pa", &1]) ++ ["-e", script]

    {output, status} =
      System.cmd(
        System.find_executable("elixir") || raise("missing Elixir executable"),
        args,
        stderr_to_stdout: true,
        cd: root
      )

    assert status == 0, output
    assert output =~ "{:restart_result, {1, 1, 1}, ["
  end
end
