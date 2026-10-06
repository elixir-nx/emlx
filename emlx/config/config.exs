import Config

# Selects the MLX-facing dispatch layer used by `EMLX.C`:
# `:c` routes through the official mlx-c API, `:cpp` (default) through
# direct mlx::core C++ calls (the reference implementation).
# Build the lanes with `EMLX_MLXC=true mix compile`.
config :emlx, :mlx_api, :cpp

if config_env() == :test do
  config :emlx, :add_backend_on_inspect, false

  # Opt-in: recompile with both debug-assertion flags on so
  # debug_flags_functional_test.exs can exercise their actual raise behavior.
  # compile_env is baked in at compile time, so this can't be toggled at
  # runtime — see that file's moduledoc for the invocation.
  if System.get_env("EMLX_DEBUG_FLAGS") == "1" do
    config :emlx, detect_non_finites: true, enable_bounds_check: true, compiler_debug: true
  end
end
