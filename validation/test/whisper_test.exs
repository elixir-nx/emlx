defmodule EMLX.Validation.WhisperTest do
  @moduledoc """
  End-to-end validation tests for Whisper on EMLX.Backend.

  Mirrors Bumblebee's Whisper test suite — same tiny-random HuggingFace
  checkpoints, same input mel features, same expected output slices.
  Reference values are from the HuggingFace Transformers (PyTorch)
  implementation; any divergence is an EMLX bug on Whisper's critical path:
  two 1-D conv encoder stages over 80×60 mel features, encoder self-attention,
  decoder self-attention and encoder-decoder cross-attention, sinusoidal PEs.

  Models are fetched from HuggingFace on first run and cached under
  `~/.cache/bumblebee`. Run with:

      mix test test/whisper_test.exs --only validation
  """

  use EMLX.ValidationCase, async: true

  @moduletag :validation
  @moduletag capture_log: true
  @moduletag timeout: 300_000

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-WhisperModel"})

    assert %Bumblebee.Audio.Whisper{architecture: :base} = spec

    inputs = %{
      "input_features" => Nx.sin(Nx.iota({1, 60, 80}, type: :f32)),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 8, 16}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.3791, -1.6131, -0.6913], [0.1247, -1.3631, 0.0034], [-0.0097, 0.2039, 1.9897]]
      ])
    )
  end

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"}
             )

    assert %Bumblebee.Audio.Whisper{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_features" => Nx.sin(Nx.iota({1, 60, 80}, type: :f32)),
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 50_257}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0942, 0.1288, 0.0243], [-0.1667, -0.1401, 0.1191], [0.0398, -0.0449, -0.0574]]
      ])
    )
  end
end
