defmodule EMLX.NativeBuildCacheTest do
  use ExUnit.Case, async: true

  test "source cache identity is stable per checkout and distinct across worktrees" do
    root = Path.join(System.tmp_dir!(), "emlx source root")
    other = Path.join(System.tmp_dir!(), "emlx source root other")

    id = EMLX.MixProject.source_cache_id(root)

    assert id == EMLX.MixProject.source_cache_id(root)
    refute id == EMLX.MixProject.source_cache_id(other)
    assert id =~ ~r/^[0-9a-f]{64}$/
  end

  test "Make build directories isolate source roots and support spaces" do
    root = Path.expand("../..", __DIR__)
    temporary = Path.join(System.tmp_dir!(), "emlx cache #{System.unique_integer([:positive])}")
    first_id = EMLX.MixProject.source_cache_id(Path.join(temporary, "worktree one"))
    second_id = EMLX.MixProject.source_cache_id(Path.join(temporary, "worktree two"))

    first = build_dir(root, temporary, first_id)
    second = build_dir(root, temporary, second_id)

    assert first == build_dir(root, temporary, first_id)
    refute first == second
    assert String.starts_with?(first, temporary)
    assert first =~ "-source-#{first_id}/objs"
    assert second =~ "-source-#{second_id}/objs"
  end

  test "Make rejects an empty source cache identity" do
    root = Path.expand("../..", __DIR__)

    {output, status} =
      System.cmd("make", ["--no-print-directory", "print-build-dir"],
        cd: root,
        env: [
          {"EMLX_CACHE_DIR", System.tmp_dir!()},
          {"EMLX_SOURCE_CACHE_ID", ""},
          {"EMLX_VERSION", "0.4.0"},
          {"MLX_VERSION", "0.31.2"}
        ],
        stderr_to_stdout: true
      )

    assert status != 0
    assert output =~ "EMLX_SOURCE_CACHE_ID is required"
  end

  test "Make clean removes only the selected source cache" do
    root = Path.expand("../..", __DIR__)
    temporary = Path.join(System.tmp_dir!(), "emlx clean #{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf!(temporary) end)

    first_id = EMLX.MixProject.source_cache_id(Path.join(temporary, "worktree one"))
    second_id = EMLX.MixProject.source_cache_id(Path.join(temporary, "worktree two"))
    first = build_dir(root, temporary, first_id)
    second = build_dir(root, temporary, second_id)
    File.mkdir_p!(first)
    File.mkdir_p!(second)

    {_output, 0} =
      System.cmd("make", ["--no-print-directory", "clean"],
        cd: root,
        env: make_env(temporary, first_id),
        stderr_to_stdout: true
      )

    refute File.exists?(first)
    assert File.dir?(second)
  end

  defp build_dir(root, cache, source_id) do
    {output, 0} =
      System.cmd("make", ["--no-print-directory", "print-build-dir"],
        cd: root,
        env: make_env(cache, source_id),
        stderr_to_stdout: true
      )

    output
    |> String.split("\n", trim: true)
    |> List.last()
  end

  defp make_env(cache, source_id) do
    [
      {"EMLX_CACHE_DIR", cache},
      {"EMLX_SOURCE_CACHE_ID", source_id},
      {"EMLX_VERSION", "0.4.0"},
      {"MLX_VERSION", "0.31.2"},
      {"MLX_VARIANT", ""},
      {"LIBMLX_ENABLE_DEBUG", "false"},
      {"MIX_APP_PATH", Path.join(cache, "mix app")}
    ]
  end
end
