import Config

config :emlx, :profile_eval, System.get_env("EMLX_PROFILE_EVAL") == "1"
