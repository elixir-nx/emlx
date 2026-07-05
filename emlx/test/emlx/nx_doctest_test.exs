defmodule EMLX.Nx.DoctestTest do
  use EMLX.Case, async: true

  setup do
    Nx.default_backend(EMLX.Backend)
    :ok
  end

  @os_specific_rounding_error (case(:os.type()) do
                                 {:unix, :darwin} ->
                                   []

                                 {:unix, _} ->
                                   [
                                     # aarch64
                                     ifft: 2,
                                     irfft: 2
                                   ]

                                 _ ->
                                   []
                               end)

  @rounding_error [
    exp: 1,
    erf: 1,
    erfc: 1,
    expm1: 1,
    atan: 1,
    sigmoid: 1,
    round: 1,
    asinh: 1,
    asin: 1,
    tan: 1,
    cos: 1,
    standard_deviation: 2,
    cosh: 1,
    log10: 1,
    acos: 1,
    covariance: 3,
    # These fail because we're using different representation types
    atan2: 2,
    as_type: 2,
    from_binary: 3,
    # f64 precision: MLX GPU math differs slightly from CPU reference values
    acosh: 1,
    atanh: 1,
    cbrt: 1,
    erf_inv: 1,
    log: 1,
    log1p: 1,
    rsqrt: 1,
    sin: 1,
    sinh: 1,
    sqrt: 1,
    tanh: 1
  ]

  @to_be_fixed [
    :moduledoc
  ]

  @not_supported [
    reduce: 4,
    window_reduce: 5,
    population_count: 1,
    count_leading_zeros: 1,
    # f8_e4m3fn is not supported by MLX; skip all tensor/2 doctests
    tensor: 2,
    # reflect deprecated for pad_outer
    reflect: 2
  ]

  doctest Nx,
    except: @rounding_error ++ @os_specific_rounding_error ++ @not_supported ++ @to_be_fixed

  describe "f64 rounding error approximations" do
    test "acosh/1 scalar" do
      assert_all_close(Nx.acosh(Nx.tensor(1, type: :f64)), Nx.tensor(0.0, type: :f64))
    end

    test "acosh/1 vector" do
      assert_all_close(
        Nx.acosh(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([0.0, 1.3169578969248166, 1.762747174039086], names: [:x], type: :f64)
      )
    end

    test "atanh/1 scalar" do
      assert_all_close(
        Nx.atanh(Nx.tensor(0.1, type: :f64)),
        Nx.tensor(0.10033534773107558, type: :f64)
      )
    end

    test "atanh/1 vector" do
      assert_all_close(
        Nx.atanh(Nx.tensor([0.1, 0.5, 0.9], names: [:x], type: :f64)),
        Nx.tensor([0.10033534773107558, 0.5493061443340549, 1.4722194895832204],
          names: [:x],
          type: :f64
        )
      )
    end

    test "cbrt/1 scalar" do
      assert_all_close(Nx.cbrt(Nx.tensor(1, type: :f64)), Nx.tensor(1.0, type: :f64))
    end

    test "cbrt/1 vector" do
      assert_all_close(
        Nx.cbrt(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([1.0, 1.2599210498948732, 1.4422495703074083], names: [:x], type: :f64)
      )
    end

    test "erf_inv/1 scalar" do
      assert_all_close(
        Nx.erf_inv(Nx.tensor(0.1, type: :f64)),
        Nx.tensor(0.08885598780483887, type: :f64)
      )
    end

    test "erf_inv/1 vector" do
      assert_all_close(
        Nx.erf_inv(Nx.tensor([0.1, 0.5, 0.9], names: [:x], type: :f64)),
        Nx.tensor([0.08885598780483887, 0.47693629334671295, 1.1630871013750377],
          names: [:x],
          type: :f64
        )
      )
    end

    test "log/1 scalar" do
      assert_all_close(Nx.log(Nx.tensor(1, type: :f64)), Nx.tensor(0.0, type: :f64))
    end

    test "log/1 vector" do
      assert_all_close(
        Nx.log(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([0.0, 0.6931471805599453, 1.0986122886681098], names: [:x], type: :f64)
      )
    end

    test "log1p/1 scalar" do
      assert_all_close(
        Nx.log1p(Nx.tensor(1, type: :f64)),
        Nx.tensor(0.6931471805599453, type: :f64)
      )
    end

    test "log1p/1 vector" do
      assert_all_close(
        Nx.log1p(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([0.6931471805599453, 1.0986122886681098, 1.3862943611198906],
          names: [:x],
          type: :f64
        )
      )
    end

    test "rsqrt/1 scalar" do
      assert_all_close(Nx.rsqrt(Nx.tensor(1, type: :f64)), Nx.tensor(1.0, type: :f64))
    end

    test "rsqrt/1 vector" do
      assert_all_close(
        Nx.rsqrt(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([1.0, 0.7071067811865475, 0.5773502691896258], names: [:x], type: :f64)
      )
    end

    test "sin/1 scalar" do
      assert_all_close(
        Nx.sin(Nx.tensor(1, type: :f64)),
        Nx.tensor(0.8414709848078965, type: :f64)
      )
    end

    test "sin/1 vector" do
      assert_all_close(
        Nx.sin(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([0.8414709848078965, 0.9092974268256817, 0.1411200080598672],
          names: [:x],
          type: :f64
        )
      )
    end

    test "sinh/1 scalar" do
      assert_all_close(
        Nx.sinh(Nx.tensor(1, type: :f64)),
        Nx.tensor(1.1752011936438014, type: :f64)
      )
    end

    test "sinh/1 vector" do
      assert_all_close(
        Nx.sinh(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([1.1752011936438014, 3.6268604078470186, 10.017874927409903],
          names: [:x],
          type: :f64
        )
      )
    end

    test "sqrt/1 scalar" do
      assert_all_close(Nx.sqrt(Nx.tensor(1, type: :f64)), Nx.tensor(1.0, type: :f64))
    end

    test "sqrt/1 vector" do
      assert_all_close(
        Nx.sqrt(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([1.0, 1.4142135623730951, 1.7320508075688772], names: [:x], type: :f64)
      )
    end

    test "tanh/1 scalar" do
      assert_all_close(
        Nx.tanh(Nx.tensor(1, type: :f64)),
        Nx.tensor(0.7615941559557649, type: :f64)
      )
    end

    test "tanh/1 vector" do
      assert_all_close(
        Nx.tanh(Nx.tensor([1, 2, 3], names: [:x], type: :f64)),
        Nx.tensor([0.7615941559557649, 0.9640275800758169, 0.9950547536867305],
          names: [:x],
          type: :f64
        )
      )
    end
  end

  describe "Linux rounding error approximations" do
    test "irfft/2 real-valued input (4th element rounds to ~0 on aarch64)" do
      assert_all_close(
        Nx.irfft(Nx.tensor([5.0, 1.0, -1.0, 1.0])),
        Nx.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], type: :f32)
      )
    end
  end
end
