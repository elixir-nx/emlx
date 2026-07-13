#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr size_t kMaxArguments = 65536;
constexpr size_t kMaxArgumentBytes = 16 * 1024 * 1024;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "emlx_plugin_tool_wrapper: " << message << "\n";
  std::exit(2);
}

std::string canonical_executable(const std::string &path) {
  if (path.empty() || path.front() != '/')
    fail("real tool path must be absolute");
  char *resolved = realpath(path.c_str(), nullptr);
  if (!resolved)
    fail("real tool path cannot be resolved");
  std::string result(resolved);
  std::free(resolved);
  struct stat info {};
  if (stat(result.c_str(), &info) != 0 || !S_ISREG(info.st_mode) ||
      access(result.c_str(), X_OK) != 0)
    fail("real tool must be a regular executable file");
  return result;
}

std::string canonical_self() {
#if defined(__APPLE__)
  uint32_t size = PATH_MAX;
  std::vector<char> path(size);
  if (_NSGetExecutablePath(path.data(), &size) != 0) {
    path.resize(size);
    if (_NSGetExecutablePath(path.data(), &size) != 0)
      fail("cannot resolve wrapper executable path");
  }
  return canonical_executable(path.data());
#elif defined(__linux__)
  std::array<char, PATH_MAX> path{};
  const ssize_t size = readlink("/proc/self/exe", path.data(), path.size() - 1);
  if (size <= 0)
    fail("cannot resolve wrapper executable path");
  path[static_cast<size_t>(size)] = '\0';
  return canonical_executable(path.data());
#else
  fail("unsupported wrapper platform");
#endif
}

void validate_argument(const std::string &argument, size_t index,
                       const std::vector<std::string> &arguments) {
  const bool loader_path_value =
      index > 0 && arguments[index - 1] == "-rpath" &&
      (argument.rfind("@loader_path/", 0) == 0 ||
       argument.rfind("@rpath/", 0) == 0 ||
       argument.rfind("@executable_path/", 0) == 0);
  if (!argument.empty() && argument.front() == '@' && !loader_path_value)
    fail("response-file argument at index " + std::to_string(index));
  if (argument.rfind("-Wl,@", 0) == 0)
    fail("forwarded response-file argument at index " + std::to_string(index));
  if ((argument == "-flto" || argument.rfind("-flto=", 0) == 0 ||
       argument == "-fuse-linker-plugin") ||
      argument.rfind("-Wl,-plugin", 0) == 0)
    fail("LTO is unsupported by plugin ABI v1");
  if (argument == "-Xlinker" && index + 1 < arguments.size() &&
      !arguments[index + 1].empty() && arguments[index + 1].front() == '@')
    fail("forwarded response-file argument at index " +
         std::to_string(index + 1));
}

std::string fresh_directory(const char *prefix) {
  std::string pattern = std::string("/tmp/") + prefix + ".XXXXXX";
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char *result = mkdtemp(writable.data());
  if (!result)
    fail("cannot create isolated tool directory");
  return result;
}

bool safe_value(const char *value) {
  if (!value || !*value)
    return false;
  return std::strchr(value, '\n') == nullptr && std::strchr(value, '\r') == nullptr;
}

void add_path_environment(std::vector<std::string> &environment,
                          const char *name) {
  const char *value = std::getenv(name);
  if (!safe_value(value))
    return;
  std::error_code error;
  const fs::path canonical = fs::canonical(value, error);
  if (error || !canonical.is_absolute())
    fail(std::string(name) + " must identify an existing absolute path");
  environment.emplace_back(std::string(name) + "=" + canonical.string());
}

std::vector<std::string> build_environment() {
  std::vector<std::string> environment = {
      "LC_ALL=C", "LANG=C", "TMPDIR=" + fresh_directory("emlx-plugin-tmp"),
      "HOME=" + fresh_directory("emlx-plugin-home")};
#if defined(__APPLE__)
  add_path_environment(environment, "SDKROOT");
  add_path_environment(environment, "DEVELOPER_DIR");
  if (const char *target = std::getenv("MACOSX_DEPLOYMENT_TARGET");
      safe_value(target)) {
    if (!std::all_of(target, target + std::strlen(target), [](unsigned char c) {
          return (c >= '0' && c <= '9') || c == '.';
        }))
      fail("MACOSX_DEPLOYMENT_TARGET is malformed");
    environment.emplace_back(std::string("MACOSX_DEPLOYMENT_TARGET=") + target);
  }
#elif defined(__linux__)
  if (const char *epoch = std::getenv("SOURCE_DATE_EPOCH"); safe_value(epoch)) {
    if (!std::all_of(epoch, epoch + std::strlen(epoch), [](unsigned char c) {
          return c >= '0' && c <= '9';
        }))
      fail("SOURCE_DATE_EPOCH is malformed");
    environment.emplace_back(std::string("SOURCE_DATE_EPOCH=") + epoch);
  }
#endif
  return environment;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 9)
    fail("expected --role, --mode, --real, and -- arguments");
  std::string role;
  std::string mode;
  std::string real;
  int separator = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--") {
      separator = i;
      break;
    }
    if (i + 1 >= argc)
      fail("missing option value");
    if (argument == "--role")
      role = argv[++i];
    else if (argument == "--mode")
      mode = argv[++i];
    else if (argument == "--real")
      real = argv[++i];
    else
      fail("unknown wrapper option");
  }
  if ((role != "compiler" && role != "linker") ||
      (mode != "scan" && mode != "compile" && mode != "link"))
    fail("unknown role or mode");
  if ((mode == "link") != (role == "linker"))
    fail("tool role does not match invocation mode");
  if (separator < 0 || separator + 1 >= argc)
    fail("missing real tool arguments");

  const std::string canonical_real = canonical_executable(real);
  if (canonical_real == canonical_self())
    fail("wrapper recursion is forbidden");

  std::vector<std::string> arguments;
  arguments.reserve(static_cast<size_t>(argc - separator));
  arguments.push_back(canonical_real);
  size_t total_bytes = canonical_real.size() + 1;
  for (int i = separator + 1; i < argc; ++i) {
    arguments.emplace_back(argv[i]);
    total_bytes += arguments.back().size() + 1;
    if (arguments.size() > kMaxArguments || total_bytes > kMaxArgumentBytes)
      fail("tool argument limits exceeded");
  }
  for (size_t i = 1; i < arguments.size(); ++i)
    validate_argument(arguments[i], i, arguments);

  auto environment = build_environment();
  std::vector<char *> argument_pointers;
  std::vector<char *> environment_pointers;
  argument_pointers.reserve(arguments.size() + 1);
  environment_pointers.reserve(environment.size() + 1);
  for (auto &argument : arguments)
    argument_pointers.push_back(argument.data());
  argument_pointers.push_back(nullptr);
  for (auto &entry : environment)
    environment_pointers.push_back(entry.data());
  environment_pointers.push_back(nullptr);

  execve(canonical_real.c_str(), argument_pointers.data(),
         environment_pointers.data());
  fail(std::string("execve failed: ") + std::strerror(errno));
}
