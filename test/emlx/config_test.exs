defmodule EMLX.ConfigTest do
  use EMLX.Case, async: false

  import ExUnit.CaptureLog

  setup do
    # Store original config value to restore after each test
    original_value = Application.get_env(:emlx, :warn_unsupported_option, true)

    on_exit(fn ->
      Application.put_env(:emlx, :warn_unsupported_option, original_value)
    end)

    {:ok, original_value: original_value}
  end

  # Test both argmax and argmin with config enabled/disabled
  for op <- [:argmax, :argmin] do
    describe "#{op} with warn_unsupported_option" do
      test "logs warning when config is enabled (default)" do
        Application.put_env(:emlx, :warn_unsupported_option, true)

        tensor = Nx.tensor([[1, 3, 2], [6, 4, 5]], backend: EMLX.Backend)

        log_output =
          capture_log(fn ->
            Nx.unquote(op)(tensor, axis: 0, tie_break: :high)
          end)

        assert log_output =~
                 "Nx.Backend.#{unquote(op)}/3 with tie_break: :high is not supported in EMLX"
      end

      test "does not log warning when config is disabled" do
        Application.put_env(:emlx, :warn_unsupported_option, false)

        tensor = Nx.tensor([[1, 3, 2], [6, 4, 5]], backend: EMLX.Backend)

        log_output =
          capture_log(fn ->
            Nx.unquote(op)(tensor, axis: 0, tie_break: :high)
          end)

        assert log_output == ""
      end
    end
  end
end
