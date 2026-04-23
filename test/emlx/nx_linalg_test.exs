defmodule EMLX.Nx.LinalgTest do
  use EMLX.Case, async: true

  # determinant: no native MLX implementation, stays on defn fallback
  @not_implemented_yet [
    determinant: 1
  ]

  # Ops that use defn fallbacks and may differ from LAPACK doctests:
  # - matrix_power: pure defn
  # - least_squares: no MLX native
  # - triangular_solve: small float differences from solve_triangular vs LAPACK reference
  # - cholesky: small float differences
  # - lu: MLX raises on singular matrices (doctest uses [[1,2,3],[4,5,6],[7,8,9]])
  # - qr: sign convention differences (-0.0 vs 0.0) and float precision
  # - svd: sign convention differences in U/Vt vectors
  @rounding_error [
    norm: 2,
    matrix_power: 2,
    least_squares: 3,
    triangular_solve: 3,
    cholesky: 1,
    lu: 1,
    qr: 2,
    svd: 2,
    solve: 2,
    eigh: 2,
    invert: 1,
    pinv: 2
  ]

  doctest Nx.LinAlg, except: @not_implemented_yet ++ @rounding_error

  describe "native linalg: lu/1" do
    test "PLU reconstruction (batched, non-singular)" do
      # Both matrices must be non-singular for MLX native LU
      {p, l, u} =
        Nx.LinAlg.lu(
          Nx.tensor([[[2.0, 1.0, 3.0], [1.0, 4.0, 2.0], [5.0, 1.0, 1.0]],
                     [[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 2.0]]])
        )

      # Verify P @ L @ U ≈ original for each batch element
      result = p |> Nx.dot([2], [0], l, [1], [0]) |> Nx.dot([2], [0], u, [1], [0])

      assert_all_close(
        result,
        Nx.tensor([[[2.0, 1.0, 3.0], [1.0, 4.0, 2.0], [5.0, 1.0, 1.0]],
                   [[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 2.0]]]),
        rtol: 1.0e-5
      )
    end

    test "PLU reconstruction roundtrip matches original" do
      a = Nx.tensor([[4.0, 3.0], [6.0, 3.0]])
      {p, l, u} = Nx.LinAlg.lu(a)
      reconstructed = p |> Nx.dot(l) |> Nx.dot(u)
      assert_all_close(reconstructed, a, rtol: 1.0e-5)
    end
  end

  describe "native linalg: qr/1" do
    test "QR reconstruction" do
      a = Nx.tensor([[3.0, 2.0], [2.0, 3.0]])
      {q, r} = Nx.LinAlg.qr(a)
      assert_all_close(Nx.dot(q, r), a, rtol: 1.0e-5)
    end

    test "Q is orthonormal" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
      {q, _r} = Nx.LinAlg.qr(a)
      qt_q = q |> Nx.transpose() |> Nx.dot(q)
      assert_all_close(qt_q, Nx.eye(3, type: :f32), rtol: 1.0e-5)
    end
  end

  describe "native linalg: svd/1" do
    test "SVD reconstruction via U @ diag(S) @ Vt ≈ A" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      {u, s, vt} = Nx.LinAlg.svd(a)
      # u is {3,3}, s is {2}, vt is {2,2}
      # Reconstruct: u[:, :2] @ diag(s) @ vt
      u_reduced = u[[0..-1//1, 0..1//1]]
      reconstructed = u_reduced |> Nx.multiply(s) |> Nx.dot(vt)
      assert_all_close(reconstructed, a, rtol: 1.0e-4)
    end
  end

  describe "native linalg: cholesky/1" do
    test "L @ L^T reconstruction" do
      a = Nx.tensor([[6.0, 3.0, 4.0, 8.0],
                     [3.0, 6.0, 5.0, 1.0],
                     [4.0, 5.0, 10.0, 7.0],
                     [8.0, 1.0, 7.0, 25.0]])
      l = Nx.LinAlg.cholesky(a)
      assert_all_close(Nx.dot(l, Nx.transpose(l)), a, rtol: 1.0e-5)
    end
  end

  describe "native linalg: eigh/1" do
    test "eigenvalue decomposition: A ≈ V @ diag(λ) @ V^T" do
      # symmetric positive-definite matrix
      a = Nx.tensor([[4.0, 2.0], [2.0, 3.0]])
      {eigenvalues, eigenvectors} = Nx.LinAlg.eigh(a)
      reconstructed =
        eigenvectors
        |> Nx.dot(Nx.make_diagonal(eigenvalues))
        |> Nx.dot(Nx.transpose(eigenvectors))
      assert_all_close(reconstructed, a, rtol: 1.0e-4)
    end
  end

  describe "native linalg: triangular_solve" do
    test "left-side lower solve: AX = B" do
      a = Nx.tensor([[3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
      b = Nx.tensor([[4.0, 7.0], [2.0, 5.0], [1.0, 3.0]])
      x = Nx.LinAlg.triangular_solve(a, b, lower: true)
      assert_all_close(Nx.dot(a, x), b, rtol: 1.0e-5)
    end

    test "left-side upper solve: AX = B" do
      a = Nx.tensor([[3.0, 2.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
      b = Nx.tensor([4.0, 2.0, 1.0])
      x = Nx.LinAlg.triangular_solve(a, b, lower: false)
      assert_all_close(Nx.dot(a, x), b, rtol: 1.0e-5)
    end
  end
end
