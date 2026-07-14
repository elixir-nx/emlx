defmodule EMLX.NativeIdentityTest do
  use ExUnit.Case, async: true

  @empty_sha256 "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  @abc_sha256 "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
  @multi_block_sha256 "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
  @multi_block "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"

  setup_all do
    root = Path.expand("../..", __DIR__)

    temporary =
      Path.join(System.tmp_dir!(), "emlx-native-identity-#{System.unique_integer([:positive])}")

    File.mkdir_p!(temporary)
    tool = Path.join(temporary, "emlx_build_identity")
    compiler = System.find_executable("c++") || raise "c++ compiler is required"

    {output, status} =
      System.cmd(
        compiler,
        [
          "-std=c++20",
          "-O2",
          "-I",
          Path.join(root, "c_src"),
          Path.join(root, "c_src/emlx/plugin/build_identity.cpp"),
          Path.join(root, "c_src/emlx/sha256.cpp"),
          Path.join(root, "c_src/emlx/plugin/depfile.cpp"),
          "-o",
          tool
        ],
        stderr_to_stdout: true
      )

    if status != 0, do: raise("failed to compile identity tool:\n#{output}")

    abi_tool = Path.join(temporary, "plugin_abi_layout")
    mlx_include = Application.app_dir(:emlx, "priv/mlx/include")

    {output, status} =
      System.cmd(
        compiler,
        [
          "-std=c++20",
          "-O2",
          "-I",
          Path.join(root, "c_src"),
          "-I",
          mlx_include,
          Path.join(root, "test/support/plugin_abi_layout.cpp"),
          "-o",
          abi_tool
        ],
        stderr_to_stdout: true
      )

    if status != 0, do: raise("failed to compile plugin ABI fixture:\n#{output}")

    on_exit(fn -> File.rm_rf!(temporary) end)
    %{tool: tool, abi_tool: abi_tool, temporary: temporary}
  end

  test "shared SHA-256 implementation matches published vectors", context do
    vectors = [
      {"empty", "", @empty_sha256},
      {"abc", "abc", @abc_sha256},
      {"multi", @multi_block, @multi_block_sha256}
    ]

    Enum.each(vectors, fn {name, content, expected} ->
      path = Path.join(context.temporary, name)
      File.write!(path, content)
      assert run!(context.tool, ["sha256", path]) == expected
    end)
  end

  test "plugin ABI layout serialization matches the independent golden vector", context do
    assert run!(context.abi_tool, []) == "72ddc26bff8c5c1 2308e9846f0852a"
  end

  test "MLX public header identity is path independent and content sensitive", context do
    roots = Enum.map(["one", "two"], &Path.join(context.temporary, &1))

    Enum.each(roots, fn root ->
      File.mkdir_p!(Path.join(root, "mlx"))
      File.write!(Path.join(root, "mlx/a.h"), "alpha")
      File.write!(Path.join(root, "mlx/extensionless"), "beta")
    end)

    [first, second] = Enum.map(roots, &run!(context.tool, ["mlx-headers", &1]))
    assert first == second

    File.write!(Path.join(List.last(roots), "mlx/a.h"), "changed")
    refute run!(context.tool, ["mlx-headers", List.last(roots)]) == first
  end

  test "generated identity artifacts preserve their inode when content is unchanged", context do
    output = Path.join(context.temporary, "content-aware/generated.hpp")
    first_id = String.duplicate("1", 64)
    second_id = String.duplicate("2", 64)

    assert run!(context.tool, ["write-id-header", output, "GENERATED_ID", first_id]) == ""
    first_inode = File.stat!(output).inode

    assert run!(context.tool, ["write-id-header", output, "GENERATED_ID", first_id]) == ""
    assert File.stat!(output).inode == first_inode

    assert run!(context.tool, ["write-id-header", output, "GENERATED_ID", second_id]) == ""
    refute File.stat!(output).inode == first_inode
  end

  test "host depfile gate parses Make escaping and publishes atomically", context do
    root = Path.join(context.temporary, "dep root")
    mlx_root = Path.join(root, "selected include")
    source = Path.join(root, "source file.cpp")
    staged = Path.join(root, "stage/compat header.hpp")
    published = Path.join(root, "published/compat header.hpp")
    mlx_header = Path.join(mlx_root, "mlx/array.h")
    escaped_header = Path.join(root, "slash\\header.hpp")
    depfile = Path.join(root, "source.d")

    Enum.each([source, staged, mlx_header, escaped_header], fn path ->
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, Path.basename(path))
    end)

    File.write!(
      depfile,
      "#{make_escape(source)}: #{make_escape(source)} #{make_escape(staged)} \\\n       #{make_escape(mlx_header)} #{make_escape(escaped_header)}\n" <>
        "#{make_escape(mlx_header)}:\n"
    )

    assert run!(context.tool, [
             "verify-host-deps",
             mlx_root,
             staged,
             published,
             source,
             depfile
           ]) == ""

    assert File.read!(published) == File.read!(staged)
  end

  test "host depfile gate rejects MLX headers resolved from a shadow root", context do
    root = Path.join(context.temporary, "shadow-test")
    mlx_root = Path.join(root, "selected")
    shadow_header = Path.join(root, "shadow/mlx/array.h")
    source = Path.join(root, "source.cpp")
    staged = Path.join(root, "staged.hpp")
    depfile = Path.join(root, "source.d")

    Enum.each([Path.join(mlx_root, "mlx/array.h"), shadow_header, source, staged], fn path ->
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, "content")
    end)

    File.write!(
      depfile,
      "#{make_escape(source)}: #{make_escape(source)} #{make_escape(staged)} " <>
        "#{make_escape(shadow_header)}\n"
    )

    assert {output, 1} =
             System.cmd(
               context.tool,
               [
                 "verify-host-deps",
                 mlx_root,
                 staged,
                 Path.join(root, "published.hpp"),
                 source,
                 depfile
               ],
               stderr_to_stdout: true
             )

    assert output =~ "outside the selected include root"
  end

  test "plugin build identity is root independent and content sensitive", context do
    first =
      plugin_manifest_fixture(context, "manifest-one", "int operation = 1;\n", "dev")

    second =
      plugin_manifest_fixture(context, "manifest-two", "int operation = 1;\n", "test")

    changed = plugin_manifest_fixture(context, "manifest-three", "int operation = 2;\n")

    assert first == second
    refute first == changed
  end

  defp run!(tool, args) do
    case System.cmd(tool, args, stderr_to_stdout: true) do
      {output, 0} -> String.trim(output)
      {output, status} -> raise "identity tool exited with #{status}: #{output}"
    end
  end

  defp make_escape(path) do
    path
    |> String.replace("\\", "\\\\")
    |> String.replace(" ", "\\ ")
    |> String.replace(":", "\\:")
  end

  defp plugin_manifest_fixture(context, name, source_content, build_env \\ "dev") do
    root = Path.join(context.temporary, name)
    axon_root = Path.join(root, "emlx_axon")
    emlx_root = Path.join(root, "emlx")
    mlx_root = Path.join(root, "mlx_include")
    source = Path.join(axon_root, "c_src/qwen3_plugin.cpp")
    policy = Path.join(axon_root, "c_src/policy.txt")
    makefile = Path.join(axon_root, "Makefile")
    emlx_public = Path.join(axon_root, "_build/#{build_env}/lib/emlx/priv/include")
    abi = Path.join(emlx_public, "emlx/plugin/abi.hpp")
    packaged_toolchain = Path.join(emlx_public, "emlx/plugin/toolchain.hpp")
    wrapper = Path.join(emlx_root, "c_src/tools/wrapper.cpp")
    source_toolchain = Path.join(emlx_root, "c_src/emlx/plugin/toolchain.hpp")
    mlx_header = Path.join(mlx_root, "mlx/array.h")
    generated = Path.join(root, "generated")
    compat = Path.join(emlx_public, "emlx/plugin/build_compat.hpp")
    actual_mlx = Path.join(generated, "emlx_qwen3_mlx_headers_build_id.hpp")
    scan_header = Path.join(generated, "scan/emlx_qwen3_plugin_build_id.hpp")
    output_header = Path.join(generated, "final/emlx_qwen3_plugin_build_id.hpp")
    output_text = Path.join(generated, "final/emlx_qwen3.build_id")
    depfile = Path.join(generated, "scan/qwen3_plugin.d")

    files = [
      {source, source_content},
      {policy, "policy-v1\n"},
      {makefile, "flags = c++20 O3\n"},
      {abi, "abi-v1\n"},
      {packaged_toolchain, "toolchain-v1\n"},
      {wrapper, "wrapper-v1\n"},
      {source_toolchain, "toolchain-v1\n"},
      {mlx_header, "mlx-header-v1\n"}
    ]

    Enum.each(files, fn {path, content} ->
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, content)
    end)

    mlx_id = run!(context.tool, ["mlx-headers", mlx_root])

    File.mkdir_p!(Path.dirname(compat))

    File.write!(
      compat,
      "inline constexpr char EMLX_EXPECTED_MLX_HEADERS_BUILD_ID[] = \"#{mlx_id}\";\n"
    )

    run!(context.tool, ["write-mlx-header-id", mlx_root, actual_mlx, compat])

    run!(context.tool, [
      "write-id-header",
      scan_header,
      "EMLX_QWEN3_PLUGIN_BUILD_ID",
      String.duplicate("0", 64)
    ])

    dependencies = [
      source,
      abi,
      packaged_toolchain,
      compat,
      actual_mlx,
      scan_header,
      mlx_header
    ]

    File.mkdir_p!(Path.dirname(depfile))

    File.write!(
      depfile,
      "c_src/qwen3_plugin.cpp: " <>
        Enum.map_join(dependencies, " ", &make_escape/1) <> "\n"
    )

    run!(context.tool, [
      "plugin-build-id",
      axon_root,
      emlx_root,
      mlx_root,
      scan_header,
      actual_mlx,
      compat,
      output_header,
      output_text,
      "--policy",
      wrapper,
      "--policy",
      source_toolchain,
      "--policy",
      policy,
      "--policy",
      makefile,
      "--flag",
      "compiler-family=clang",
      "--flag",
      "language=c++20",
      "--dep",
      "c_src/qwen3_plugin.cpp",
      depfile
    ])
  end
end
