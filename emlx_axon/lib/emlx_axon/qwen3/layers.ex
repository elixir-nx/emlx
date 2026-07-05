defmodule EMLXAxon.Qwen3.Layers do
  @moduledoc """
  Stateless layer primitives: RMSNorm, SwiGLU.

  `rms_norm/3` delegates to `EMLX.Fast.rms_norm` (single fused Metal shader).
  RoPE is no longer computed here — `Attention.forward/12` delegates to the
  native `EMLX.Native.Qwen3.kv_cache_attention/attention_block` NIFs, which
  transpose to `{B, N, T, D}` and apply RoPE internally via
  `mlx::core::fast::rope` (see `qwen3_plugin.cpp`).
  """

  import Nx.Defn

  # ── RMSNorm ─────────────────────────────────────────────────────────────────

  @doc """
  Root-mean-square layer normalisation via `mlx::fast::rms_norm`.

  `x`: any shape; normalises over the last axis.
  `weight`: `{hidden}` scale vector.
  """
  def rms_norm(x, weight, eps), do: EMLX.Fast.rms_norm(x, weight, eps)

  # ── SwiGLU ──────────────────────────────────────────────────────────────────

  @doc "SwiGLU activation: `silu(gate) * up`.  `gate` and `up` must have the same shape."
  defn swiglu(gate, up), do: gate * Nx.sigmoid(gate) * up
end
