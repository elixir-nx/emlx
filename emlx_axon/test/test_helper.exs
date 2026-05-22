use_gpu? =
  String.downcase(System.get_env("EMLX_TEST_DEFAULT_GPU", "false")) in [
    "1",
    "true",
    "yes",
    "t",
    "y"
  ]

if use_gpu? do
  Application.put_env(:nx, :default_backend, {EMLX.Backend, device: :gpu})
end

gpu_exclude =
  if use_gpu? do
    case EMLX.NIF.command_queue_new(:gpu) do
      {:ok, _} -> []
      {:error, _} -> [:metal]
    end
  else
    [:metal]
  end

ExUnit.start(exclude: [:bumblebee] ++ gpu_exclude)
