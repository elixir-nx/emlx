Mix.install([
   {:bumblebee, github: "elixir-nx/bumblebee", override: true},
   {:emlx, path: __DIR__}
], system_env: %{"LIBMLX_ENABLE_DEBUG" => "true"})

IO.puts("1. Setting backend...")
Nx.global_default_backend({EMLX.Backend, device: :gpu})
Nx.Defn.default_options(compiler: EMLX)

IO.puts("2. Loading model...")
{:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})

IO.puts("3. Loading tokenizer...")
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})

IO.puts("4. Loading generation config...")
{:ok, generation_config} =
  Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

generation_config = Bumblebee.configure(generation_config, max_new_tokens: 20)

IO.puts("5. Creating serving...")
serving =
  Bumblebee.Text.generation(model_info, tokenizer, generation_config,
    compile: [batch_size: 1, sequence_length: 100],
    stream: true
  )

IO.puts("6. Running serving...")
result = Nx.Serving.run(serving, "What is the capital of queensland?")

dbg(result)

IO.puts("7. Collecting results...")
result
|> Enum.to_list()
|> IO.inspect(label: "Final result")
