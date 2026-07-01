import Config

if config_env() == :test do
  config :emlx, :add_backend_on_inspect, false

  # Opt-in: recompile with both debug-assertion flags on so
  # debug_flags_functional_test.exs can exercise their actual raise behavior.
  # compile_env is baked in at compile time, so this can't be toggled at
  # runtime — see that file's moduledoc for the invocation.
  if System.get_env("EMLX_DEBUG_FLAGS") == "1" do
    config :emlx, detect_non_finites: true, enable_bounds_check: true
  end
end
