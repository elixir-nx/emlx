import Config

if config_env() == :test do
  config :emlx, :add_backend_on_inspect, false
end
