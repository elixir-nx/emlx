defmodule EMLXTest do
  use EMLX.Case
  doctest EMLX

  test "__jit__" do
    {left, right} =
      Nx.Defn.jit_apply(&{Nx.add(&1, &2), Nx.subtract(&1, &2)}, [Nx.tensor(1), 2], compiler: EMLX)

    assert_equal(left, Nx.tensor(3))
    assert_equal(right, Nx.tensor(-1))
  end

  test "__jit__ supports binary backend in arguments" do
    {left, right} =
      Nx.Defn.jit_apply(
        &{Nx.add(&1, &2), Nx.subtract(&1, &2)},
        [Nx.tensor(1, backend: Nx.BinaryBackend), 2],
        compiler: EMLX
      )

    assert_equal(left, Nx.tensor(3))
    assert_equal(right, Nx.tensor(-1))
  end

  test "__jit__ supports binary backend as the default backend" do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      {left, right} =
        Nx.Defn.jit_apply(
          &{Nx.add(&1, &2), Nx.subtract(&1, &2)},
          [Nx.tensor(1), 2],
          compiler: EMLX
        )

      assert_equal(left, Nx.tensor(3))
      assert_equal(right, Nx.tensor(-1))
    end)
  end
end
