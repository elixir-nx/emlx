import Config

# Compile-time debug flags for EMLX. Both default to false (zero runtime cost).
# Enable these only during development or CI runs where you want extra assertions.
#
# After changing either flag you MUST force-recompile the affected modules:
#   mix compile --force
#
# WARNING: Both flags break MLX lazy-graph fusion by forcing eager mx::eval syncs.
# On non-unified-memory targets (Linux GPU), :enable_bounds_check incurs an extra
# GPU→CPU copy per indexed op. Never enable in production.

# Raises ArgumentError on out-of-bounds indices in gather, take, take_along_axis,
# indexed_add, and indexed_put before the NIF call.
# config :emlx, enable_bounds_check: true

# Raises ArgumentError if dot (matmul / einsum) produces NaN or Inf.
# When EMLX.Fast is implemented (task 05), rms_norm, layer_norm, and
# scaled_dot_product_attention will also be checked.
# config :emlx, detect_non_finites: true
