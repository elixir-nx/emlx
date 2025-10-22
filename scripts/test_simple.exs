IO.puts("Setting backend...")
Nx.global_default_backend({EMLX.Backend, device: :cpu})
Nx.Defn.default_options(compiler: EMLX)

IO.puts("Defining simple defn...")
defmodule SimpleTest do
  import Nx.Defn

  defn add_one(x) do
    Nx.add(x, 1)
  end
end

IO.puts("Creating tensor...")
x = Nx.tensor([1, 2, 3])

IO.puts("Calling defn (this will trigger compilation)...")
result = SimpleTest.add_one(x)

IO.puts("Success! Result: #{inspect(result)}")
