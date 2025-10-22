IO.puts("1. Setting backend...")
Nx.global_default_backend({EMLX.Backend, device: :cpu})
Nx.Defn.default_options(compiler: EMLX)

IO.puts("2. Loading model...")
{:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})

IO.puts("3. Loading tokenizer...")
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})

IO.puts("4. Loading generation config...")
{:ok, generation_config} =
  Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

generation_config = Bumblebee.configure(generation_config, max_new_tokens: 20)

IO.puts("5. Creating serving (NO STREAM)...")
serving =
  Bumblebee.Text.generation(model_info, tokenizer, generation_config,
    compile: [batch_size: 1, sequence_length: 100],
    stream: false
  )

IO.puts("6. Running serving...")
result = Nx.Serving.run(serving, "What is the capital of queensland?")

IO.puts("7. Got result!")
IO.inspect(result, label: "Final result")
