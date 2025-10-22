Mix.install([
   {:bumblebee, github: "elixir-nx/bumblebee", override: true},
   {:emlx, path: __DIR__}
], system_env: %{"LIBMLX_ENABLE_DEBUG" => "true"}, force: true)

 Nx.global_default_backend({EMLX.Backend, device: :gpu})

 Nx.Defn.default_options(compiler: EMLX)

 {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
 {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})

 {:ok, generation_config} =
   Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

 generation_config = Bumblebee.configure(generation_config, max_new_tokens: 20)

 serving =
   Bumblebee.Text.generation(model_info, tokenizer, generation_config,
     compile: [batch_size: 1, sequence_length: 100],
     stream: true
   )

Nx.Serving.run(serving, "What is the capital of queensland?")
|> Enum.to_list()
|> IO.puts()
