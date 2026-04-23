defmodule EMLX.Nx.LinalgTest do
  use EMLX.Case, async: true

  # Note: most of these are depending on gather
  @not_implemented_yet [
    determinant: 1
  ]

  @rounding_error [
    lu: 1,
    norm: 2,
    triangular_solve: 3,
    solve: 2,
    matrix_power: 2,
    eigh: 2,
    svd: 2,
    cholesky: 1,
    least_squares: 3,
    invert: 1,
    pinv: 2
  ]

  doctest Nx.LinAlg, except: @not_implemented_yet ++ @rounding_error

  describe "rounding error approximations" do
    test "lu/1 batched PLU reconstruction" do
      {p, l, u} =
        Nx.LinAlg.lu(
          Nx.tensor([[[9, 8, 7], [6, 5, 4], [3, 2, 1]], [[-1, 0, -1], [1, 0, 1], [1, 1, 1]]])
        )

      result = p |> Nx.dot([2], [0], l, [1], [0]) |> Nx.dot([2], [0], u, [1], [0])

      assert_all_close(
        result,
        Nx.tensor([
          [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
          [[-1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        ])
      )
    end
  end
end
