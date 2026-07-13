#include "emlx_native_image.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

enum class Mode { unchanged, mutate, replace };

struct BarrierContext {
  Mode mode;
  std::string original_path;
};

[[noreturn]] void fail(const std::string &message) {
  std::fprintf(stderr, "%s\n", message.c_str());
  std::exit(2);
}

void write_all(int fd, const char *data, size_t size) {
  while (size > 0) {
    const ssize_t written = write(fd, data, size);
    if (written <= 0)
      fail(std::string("write failed: ") + std::strerror(errno));
    data += written;
    size -= static_cast<size_t>(written);
  }
}

void barrier(const char *path, void *opaque) {
  auto &context = *static_cast<BarrierContext *>(opaque);
  if (context.mode == Mode::unchanged)
    return;

  if (context.mode == Mode::mutate) {
    const int fd = open(path, O_WRONLY | O_APPEND);
    if (fd < 0)
      fail(std::string("open for mutation failed: ") + std::strerror(errno));
    write_all(fd, "x", 1);
    if (fsync(fd) != 0 || close(fd) != 0)
      fail(std::string("mutation flush failed: ") + std::strerror(errno));
    return;
  }

  context.original_path = std::string(path) + ".opened-original";
  unlink(context.original_path.c_str());
  if (rename(path, context.original_path.c_str()) != 0)
    fail(std::string("rename failed: ") + std::strerror(errno));
  const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0755);
  if (fd < 0)
    fail(std::string("replacement create failed: ") + std::strerror(errno));
  write_all(fd, "replacement", 11);
  if (fsync(fd) != 0 || close(fd) != 0)
    fail(std::string("replacement flush failed: ") + std::strerror(errno));
}

uint64_t inode_of(const std::string &path) {
  struct stat info {};
  if (stat(path.c_str(), &info) != 0)
    fail(std::string("stat failed: ") + std::strerror(errno));
  return static_cast<uint64_t>(info.st_ino);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3)
    fail("usage: native_image_fixture <unchanged|mutate|replace> <library>");

  Mode mode;
  if (std::strcmp(argv[1], "unchanged") == 0)
    mode = Mode::unchanged;
  else if (std::strcmp(argv[1], "mutate") == 0)
    mode = Mode::mutate;
  else if (std::strcmp(argv[1], "replace") == 0)
    mode = Mode::replace;
  else
    fail("unknown mode");

  void *handle = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    fail(std::string("dlopen failed: ") + dlerror());
  auto anchor = reinterpret_cast<EMLXMLXRuntimeAnchor>(dlsym(handle, "emlx_test_anchor"));
  if (!anchor)
    fail("anchor symbol is missing");

  BarrierContext context{mode, {}};
  emlx_native_image_set_test_barrier(barrier, &context);
  EMLXNativeImageIdentity identity;
  EMLXNativeImageError error;
  const bool ok = emlx_native_image_identity(anchor, identity, error);
  emlx_native_image_set_test_barrier(nullptr, nullptr);

  if (mode == Mode::mutate) {
    if (ok || error.code != EMLXNativeImageErrorCode::changed_while_hashing)
      fail("same-inode mutation was not rejected");
    std::puts("changed_while_hashing");
  } else if (!ok) {
    fail(std::string("identity failed: ") + emlx_native_image_error_name(error.code));
  } else if (mode == Mode::replace) {
    if (identity.inode != inode_of(context.original_path) ||
        identity.inode == inode_of(argv[2]))
      fail("identity did not remain pinned to the opened original inode");
    std::puts("opened_original");
  } else {
    if (identity.inode != inode_of(argv[2]))
      fail("unchanged identity has the wrong inode");
    std::puts("unchanged");
  }

  dlclose(handle);
  return 0;
}
