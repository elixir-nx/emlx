defmodule EMLX.Validation.LlamaTest do
  @moduledoc """
  End-to-end validation tests for Llama on EMLX.Backend.

  Mirrors Bumblebee's Llama test suite — same tiny-random checkpoints,
  same inputs, same expected output slices. Reference values are from the
  HuggingFace Transformers (PyTorch) implementation; any divergence is an
  EMLX bug on the decoder-only critical path: RMSNorm, RoPE, causal masked
  self-attention, SwiGLU FFN.

  This is the primary conformance gate for EMLX.Fast.rms_norm and
  EMLX.Fast.rope when fast kernels are present.

  Models are fetched from HuggingFace on first run and cached under
  `~/.cache/bumblebee`. Run with:

      mix test test/llama_test.exs --only validation
  """

  use EMLX.ValidationCase, async: true

  @moduletag :validation
  @moduletag capture_log: true
  @moduletag timeout: 120_000

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-LlamaModel"})

    assert %Bumblebee.Text.Llama{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[1.4799, -2.0333, 0.4759], [2.3749, -0.8369, -0.0206], [0.5767, -0.0515, -1.1795]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-LlamaForCausalLM"})

    assert %Bumblebee.Text.Llama{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0469, -0.0751, 0.0349], [0.0617, -0.1357, -0.0204], [-0.1495, 0.0557, -0.0737]]
      ])
    )
  end
end
