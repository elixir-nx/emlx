defmodule EMLX.MixProject do
  use Mix.Project

  @app :emlx
  @version "0.1.1-dev"
  @mlx_version "0.22.1"

  def project do
    libmlx_config = libmlx_config()

    [
      app: @app,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # elixir_make
      make_env: %{
        "MLX_DIR" => libmlx_config.dir,
        "MLX_VERSION" => libmlx_config.version,
        "MLX_BUILD" => to_string(libmlx_config.features.build?),
        "MLX_INCLUDE_DIR" =>
          Path.join(
            libmlx_config.dir,
            if(libmlx_config.features.build?, do: "usr/include", else: "include")
          ),
        "MLX_LIB_DIR" =>
          Path.join(
            libmlx_config.dir,
            if(libmlx_config.features.build?, do: "usr/lib", else: "lib")
          ),
        "MLX_VARIANT" => libmlx_config.variant,
        "EMLX_CACHE_DIR" => libmlx_config.cache_dir,
        "EMLX_VERSION" => @version,
        "MIX_BUILD_EMBEDDED" => "#{Mix.Project.config()[:build_embedded]}",
        "LIBMLX_ENABLE_DEBUG" => to_string(libmlx_config.features.debug?)
      },

      # Compilers
      compilers: compilers(libmlx_config),
      aliases: aliases()
    ]
  end

  def application do
    [
      mod: {EMLX.Application, []},
      extra_applications: [:logger, :inets, :ssl, :public_key, :crypto]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:elixir_make, "~> 0.6"},
      {:nx, "~> 0.9.2"},
      {:nif_call, "~> 0.1.2"}
    ]
  end

  defp aliases do
    [
      "compile.mlx": &download_and_unarchive/1
    ]
  end

  defp compilers(libmlx_config) do
    compilers = [:elixir_make] ++ Mix.compilers()

    unless libmlx_config.features.build? do
      [:mlx] ++ compilers
    else
      compilers
    end
  end

  defp libmlx_config() do
    version = System.get_env("LIBMLX_VERSION", @mlx_version)

    features = %{
      jit?: to_boolean(System.get_env("LIBMLX_ENABLE_JIT")),
      debug?: to_boolean(System.get_env("LIBMLX_ENABLE_DEBUG")),
      build?: to_boolean(System.get_env("LIBMLX_BUILD"))
    }

    variant = to_variant(features)

    cache_dir =
      if dir = System.get_env("LIBMLX_CACHE") do
        Path.expand(dir)
      else
        :filename.basedir(:user_cache, "libmlx")
      end

    %{
      version: version,
      dir: Path.join(cache_dir, "libmlx-#{version}#{variant}"),
      features: features,
      variant: variant,
      cache_dir: cache_dir
    }
  end

  defp to_boolean(nil), do: false

  defp to_boolean(var) when is_boolean(var) do
    var
  end

  defp to_boolean(var) do
    String.downcase(to_string(var)) in ["1", "true", "on", "yes", "y"]
  end

  defp to_variant(features) do
    [
      if(features.build?, do: "build", else: nil),
      if(features.debug?, do: "debug", else: nil),
      if(features.jit?, do: "jit", else: nil)
    ]
    |> Enum.filter(&(&1 != nil))
    |> Enum.sort()
    |> Enum.map(&"-#{&1}")
    |> Enum.join("")
  end

  defp download_and_unarchive(args) do
    libmlx_config = libmlx_config()

    if "--force" in args do
      File.rm_rf(libmlx_config.dir)
      File.rm_rf(libmlx_config.cache_dir)
    end

    if File.dir?(libmlx_config.dir) do
      {:ok, []}
    else
      download_and_unarchive(libmlx_config.cache_dir, libmlx_config)
    end
  end

  defp download_and_unarchive(cache_dir, libmlx_config) do
    File.mkdir_p!(cache_dir)

    libmlx_archive =
      Path.join(cache_dir, "libmlx-#{libmlx_config.version}#{libmlx_config.variant}.tar.gz")

    libmlx_archive = System.get_env("MLX_ARCHIVE_PATH", libmlx_archive)

    url =
      "https://github.com/cocoa-xu/mlx-build/releases/download/v#{libmlx_config.version}/mlx-arm64-apple-darwin#{libmlx_config.variant}.tar.gz"

    sha256_url = "#{url}.sha256"

    verify_integrity = "sha256=url:#{sha256_url}"

    {url, verify_integrity} =
      if customized_url = System.get_env("MLX_ARCHIVE_URL") do
        verify_integrity = System.get_env("MLX_ARCHIVE_INTEGRITY")
        {customized_url, verify_integrity}
      else
        {url, verify_integrity}
      end

    unless File.exists?(libmlx_archive) do
      # Download libmlx

      if {:unix, :darwin} != :os.type() do
        Mix.raise("EMLX only supports macOS for now")
      end

      download!(url, libmlx_archive)
      :ok = maybe_verify_integrity!(verify_integrity, libmlx_archive)
    end

    # Unpack libmlx and move to the target cache dir
    parent_libmlx_dir = Path.join(Path.dirname(libmlx_config.dir), "libmlx")
    File.mkdir_p!(parent_libmlx_dir)

    # Extract to the parent directory (it will be inside the libmlx directory)
    :ok = :erl_tar.extract(libmlx_archive, [:compressed, {:cwd, parent_libmlx_dir}])

    # And then rename
    File.rename!(parent_libmlx_dir, libmlx_config.dir)

    :ok
  end

  defp maybe_verify_integrity!(nil, _libmlx_archive), do: :ok

  defp maybe_verify_integrity!(verify_integrity, libmlx_archive) do
    {checksum_algo, expected_checksum} = get_checksum_info!(verify_integrity)
    libmlx_archive_checksum = checksum!(libmlx_archive, checksum_algo)

    if expected_checksum != libmlx_archive_checksum do
      Mix.raise(
        "Checksum (#{checksum_algo}) mismatch for #{libmlx_archive}. Expected: #{expected_checksum}, got: #{libmlx_archive_checksum}"
      )
    else
      :ok
    end
  end

  @known_checksum_algos [
    "sha",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "blake2b",
    "blake2s",
    "ripemd160",
    "md4",
    "md5"
  ]

  defp get_checksum_info!(verify_integrity) do
    case String.split(verify_integrity, "=", parts: 2, trim: true) do
      [algo, checksum] when algo in @known_checksum_algos ->
        {String.to_existing_atom(algo), get_checksum_value!(checksum)}

      _ ->
        Mix.raise("Invalid checksum: #{verify_integrity}")
    end
  end

  defp get_checksum_value!("url:" <> url) do
    checksum_from_url!(url)
  end

  defp get_checksum_value!(checksum) do
    checksum
  end

  defp checksum_from_url!(url) do
    data = download!(url)
    checksum = String.split(data, " ", parts: 2, trim: true)

    if length(checksum) == 0 do
      Mix.raise("Invalid checksum file: #{url}")
    end

    hd(checksum)
  end

  def download!(url, save_as \\ nil) do
    url_charlist = String.to_charlist(url)

    if proxy = System.get_env("HTTP_PROXY") || System.get_env("http_proxy") do
      Mix.shell().info("Using HTTP_PROXY: #{proxy}")
      %{host: host, port: port} = URI.parse(proxy)

      :httpc.set_options([{:proxy, {{String.to_charlist(host), port}, []}}])
    end

    if proxy = System.get_env("HTTPS_PROXY") || System.get_env("https_proxy") do
      Mix.shell().info("Using HTTPS_PROXY: #{proxy}")
      %{host: host, port: port} = URI.parse(proxy)
      :httpc.set_options([{:https_proxy, {{String.to_charlist(host), port}, []}}])
    end

    # https://erlef.github.io/security-wg/secure_coding_and_deployment_hardening/inets
    # TODO: This may no longer be necessary from Erlang/OTP 25.0 or later.
    https_options = [
      ssl:
        [
          verify: :verify_peer,
          customize_hostname_check: [
            match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
          ]
        ] ++ cacerts_options()
    ]

    options = [body_format: :binary]

    case :httpc.request(:get, {url_charlist, []}, https_options, options) do
      {:ok, {{_, 200, _}, _headers, body}} ->
        if save_as do
          File.write!(save_as, body)
        end

        body

      other ->
        Mix.raise("Failed to download #{url}, reason: #{inspect(other)}")
    end
  end

  defp cacerts_options do
    cond do
      path = System.get_env("HEX_CACERTS_PATH") ->
        [cacertfile: path]

      certs = otp_cacerts() ->
        [cacerts: certs]

      true ->
        warn_no_cacerts()
        []
    end
  end

  defp otp_cacerts do
    if System.otp_release() >= "25" do
      # cacerts_get/0 raises if no certs found
      try do
        :public_key.cacerts_get()
      rescue
        _ -> nil
      end
    end
  end

  defp warn_no_cacerts do
    Mix.shell().error("""
    No certificate trust store was found.

    A certificate trust store is required in
    order to download locales for your configuration.
    Since elixir_make could not detect a system
    installed certificate trust store one of the
    following actions may be taken:

    1. Specify the location of a certificate trust store
       by configuring it in environment variable:

         export HEX_CACERTS_PATH="/path/to/cacerts.pem"

    2. Use OTP 25+ on an OS that has built-in certificate
       trust store.
    """)
  end

  defp checksum!(file_path, algo) do
    case File.read(file_path) do
      {:ok, content} ->
        Base.encode16(:crypto.hash(algo, content), case: :lower)

      {:error, reason} ->
        Mix.raise(
          "Cannot read the file for checksum comparison: #{file_path}. Reason: #{inspect(reason)}"
        )
    end
  end
end
