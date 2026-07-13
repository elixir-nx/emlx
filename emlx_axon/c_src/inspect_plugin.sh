#!/bin/sh
set -eu

mode=${1-}
artifact=${2-}
policy=${3-}
temporary=$(mktemp "${TMPDIR:-/tmp}/emlx-plugin-inspect.XXXXXX")
trap 'rm -f "$temporary"' EXIT HUP INT TERM

fail() {
  printf '%s\n' "plugin inspection failed: $1" >&2
  exit 1
}

policy_has() {
  grep -F -x "$1=$2" "$policy" >/dev/null
}

validate_policy() {
  [ -f "$policy" ] || fail "dependency policy is missing"
  [ "$(sed -n '1p' "$policy")" = "EMLX_PLUGIN_DEPENDENCY_POLICY_V1" ] ||
    fail "dependency policy has the wrong version"
}

inspect_object() {
  if nm "$artifact" | grep -E '(__cxx_global_var_init|_GLOBAL__sub_I|__cxa_atexit|__cxa_finalize)' >/dev/null; then
    fail "object contains dynamic initialization or exit-time destruction"
  fi

  if [ "$(uname -s)" = "Darwin" ] &&
     otool -l "$artifact" | grep -E '(__mod_init_func|__mod_term_func)' >/dev/null; then
    fail "object contains a Mach-O initializer or terminator section"
  fi
}

inspect_darwin_image() {
  exports=$(nm -gU "$artifact" | awk '{print $NF}' | sed 's/^_//' | LC_ALL=C sort -u)
  [ "$exports" = "emlx_plugin_descriptor_v1" ] ||
    fail "image must export only emlx_plugin_descriptor_v1"

  otool -l "$artifact" | awk '$1 == "cmd" { print $2 }' | LC_ALL=C sort -u >"$temporary"
  while IFS= read -r command; do
    policy_has darwin.command "$command" ||
      fail "unexpected Mach-O load command: $command"
  done <"$temporary"

  mlx_count=0
  otool -L "$artifact" | tail -n +3 | awk '{print $1}' >"$temporary"
  while IFS= read -r dependency; do
    if policy_has darwin.mlx "$dependency"; then
      mlx_count=$((mlx_count + 1))
    elif policy_has darwin.system "$dependency"; then
      :
    else
      framework_root=$(printf '%s\n' "$dependency" | sed -n 's#^\(.*/[^/]*\.framework\)/.*#\1#p')
      [ -n "$framework_root" ] && policy_has darwin.framework "$framework_root" ||
        fail "unexpected Mach-O dependency: $dependency"
    fi
  done <"$temporary"
  [ "$mlx_count" = 1 ] || fail "image must select exactly one libmlx dependency"

  otool -l "$artifact" | awk '
    $1 == "cmd" && $2 == "LC_RPATH" { wanted = 1; next }
    wanted && $1 == "path" { print $2; wanted = 0 }
  ' >"$temporary"
  rpath_count=0
  while IFS= read -r rpath; do
    [ -n "$rpath" ] || continue
    policy_has darwin.rpath "$rpath" || fail "unexpected Mach-O RPATH: $rpath"
    rpath_count=$((rpath_count + 1))
  done <"$temporary"
  [ "$rpath_count" = 1 ] || fail "image must contain exactly one approved LC_RPATH"
}

inspect_linux_image() {
  exports=$(nm -D --defined-only "$artifact" | awk '{print $NF}' | LC_ALL=C sort -u)
  [ "$exports" = "emlx_plugin_descriptor_v1" ] ||
    fail "image must export only emlx_plugin_descriptor_v1"

  readelf -d "$artifact" | grep -E '\((FILTER|AUXILIARY)\)' >/dev/null &&
    fail "ELF filter and auxiliary dependencies are forbidden"

  mlx_count=0
  readelf -d "$artifact" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' >"$temporary"
  while IFS= read -r dependency; do
    if policy_has linux.mlx "$dependency"; then
      mlx_count=$((mlx_count + 1))
    elif policy_has linux.system "$dependency"; then
      :
    else
      fail "unexpected ELF dependency: $dependency"
    fi
  done <"$temporary"
  [ "$mlx_count" = 1 ] || fail "image must select exactly one libmlx dependency"

  readelf -d "$artifact" | sed -n 's/.*\(RUNPATH\|RPATH\).*\[\(.*\)\]/\2/p' >"$temporary"
  runpath_count=0
  while IFS= read -r runpath; do
    [ -n "$runpath" ] || continue
    policy_has linux.rpath "$runpath" || fail "unexpected ELF search path: $runpath"
    runpath_count=$((runpath_count + 1))
  done <"$temporary"
  [ "$runpath_count" = 1 ] || fail "image must contain exactly one approved RPATH or RUNPATH"
}

case "$mode" in
  object)
    [ "$#" -eq 2 ] || fail "usage: inspect_plugin.sh object <artifact>"
    inspect_object
    ;;
  image)
    [ "$#" -eq 3 ] || fail "usage: inspect_plugin.sh image <artifact> <policy>"
    validate_policy
    case "$(uname -s)" in
      Darwin) inspect_darwin_image ;;
      Linux) inspect_linux_image ;;
      *) fail "unsupported image inspection platform" ;;
    esac
    ;;
  *) fail "expected object or image mode" ;;
esac
