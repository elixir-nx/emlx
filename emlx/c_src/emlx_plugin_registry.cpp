#include "emlx_plugin_registry.hpp"
#include "nx_nif_utils.hpp"

#include <dlfcn.h>
#include <sstream>
#include <unordered_map>

namespace {

std::unordered_map<std::string, void *> g_plugin_handles;
std::unordered_map<std::string, void *> g_plugin_vtables;

} // namespace

const void *emlx_get_plugin(const std::string &name) {
  auto it = g_plugin_vtables.find(name);
  return it == g_plugin_vtables.end() ? nullptr : it->second;
}

ERL_NIF_TERM load_plugin(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  std::string name;
  std::string path;
  if (!nx::nif::get(env, argv[0], name)) {
    return nx::nif::error(env, "load_plugin expects a name (as the 1st argument)");
  }
  if (!nx::nif::get(env, argv[1], path)) {
    return nx::nif::error(env, "load_plugin expects a path string (as the 2nd argument)");
  }

  auto existing_handle = g_plugin_handles.find(name);
  if (existing_handle != g_plugin_handles.end()) {
    dlclose(existing_handle->second);
    g_plugin_handles.erase(existing_handle);
    g_plugin_vtables.erase(name);
  }

  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::ostringstream msg;
    msg << "Failed to load plugin \"" << name << "\" at " << path << ": " << dlerror();
    return nx::nif::error(env, msg.str().c_str());
  }

  using VTableFn = void *(*)();
  auto get_vtable = reinterpret_cast<VTableFn>(dlsym(handle, "emlx_plugin_vtable"));
  if (get_vtable == nullptr) {
    dlclose(handle);
    return nx::nif::error(env, "plugin is missing the emlx_plugin_vtable symbol");
  }

  void *vtable = get_vtable();
  if (vtable == nullptr) {
    dlclose(handle);
    return nx::nif::error(env, "plugin returned a null vtable");
  }

  g_plugin_handles[name] = handle;
  g_plugin_vtables[name] = vtable;
  return nx::nif::ok(env);
}
