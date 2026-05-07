defmodule EMLX.Validation.Qwen3Quantized.Layers do
  @moduledoc """
  Stateless layer primitives: RMSNorm, SwiGLU.

  `rms_norm/3` delegates to `EMLX.Fast.rms_norm` (single fused Metal shader).
  RoPE is no longer computed here — `EMLX.Fast.rope/6` is called directly in
  `Attention.forward/10` after projecting and transposing to `{B, N, T, D}`.
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
