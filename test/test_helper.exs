backend =
  if String.downcase(System.get_env("EMLX_TEST_DEFAULT_GPU", "false")) in [
       "1",
       "true",
       "yes",
       "t",
       "y"
     ] do
    {EMLX.Backend, device: :gpu}
  else
    EMLX.Backend
  end

Application.put_env(:nx, :default_backend, backend)

ExUnit.start()
