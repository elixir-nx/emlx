#include "emlx_sha256.hpp"
#include "emlx_depfile.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "emlx_plugin_toolchain.hpp"

namespace fs = std::filesystem;

namespace {

struct Entry {
  std::string path;
  std::vector<uint8_t> content;
};

struct ManifestEntry {
  std::string kind;
  std::string path;
  std::vector<uint8_t> content;
};

void update_u64(EMLXSHA256 &hash, uint64_t value) {
  std::array<uint8_t, 8> bytes{};
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<uint8_t>(value >> (i * 8U));
  hash.update(bytes.data(), bytes.size());
}

std::vector<uint8_t> read_file(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read " + path.string());
  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  if (end < 0)
    throw std::runtime_error("cannot determine file size for " + path.string());
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  input.seekg(0, std::ios::beg);
  if (!bytes.empty() &&
      !input.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
    throw std::runtime_error("cannot read complete file " + path.string());
  return bytes;
}

std::string hash_file(const fs::path &path) {
  const int raw_fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (raw_fd < 0)
    throw std::runtime_error("cannot open " + path.string() + ": " +
                             std::strerror(errno));
  std::array<uint8_t, 32> digest{};
  std::string error;
  const bool ok = emlx_sha256_file_descriptor(raw_fd, digest, error);
  close(raw_fd);
  if (!ok)
    throw std::runtime_error("cannot hash " + path.string() + ": " + error);
  return emlx_sha256_hex(digest);
}

std::string hash_headers(const fs::path &include_root) {
  const fs::path canonical_root = fs::canonical(include_root);
  const fs::path mlx_root = canonical_root / "mlx";
  if (!fs::is_directory(mlx_root))
    throw std::runtime_error("MLX public include root has no mlx directory");

  std::vector<Entry> entries;
  std::set<std::string> canonical_files;
  for (const auto &directory_entry : fs::recursive_directory_iterator(mlx_root)) {
    const auto status = directory_entry.symlink_status();
    if (fs::is_directory(status))
      continue;
    if (!fs::is_regular_file(status) && !fs::is_symlink(status))
      throw std::runtime_error("MLX header tree contains a nonregular entry");
    const fs::path canonical_file = fs::canonical(directory_entry.path());
    const auto relative_canonical = canonical_file.lexically_relative(canonical_root);
    if (relative_canonical.empty() || relative_canonical.native().starts_with(".."))
      throw std::runtime_error("MLX header symlink escapes the include root");
    if (!canonical_files.insert(canonical_file.string()).second)
      throw std::runtime_error("MLX header tree contains duplicate content aliases");
    const fs::path logical = directory_entry.path().lexically_relative(canonical_root);
    if (logical.empty() || logical.native().starts_with(".."))
      throw std::runtime_error("invalid MLX header logical path");
    entries.push_back({logical.generic_string(), read_file(canonical_file)});
  }

  std::sort(entries.begin(), entries.end(), [](const Entry &left, const Entry &right) {
    return std::lexicographical_compare(
        left.path.begin(), left.path.end(), right.path.begin(), right.path.end(),
        [](unsigned char a, unsigned char b) { return a < b; });
  });

  EMLXSHA256 hash;
  static constexpr char domain[] = "EMLX_MLX_HEADERS_MANIFEST_V1\n";
  hash.update(reinterpret_cast<const uint8_t *>(domain), sizeof(domain) - 1);
  update_u64(hash, entries.size());
  for (const auto &entry : entries) {
    update_u64(hash, entry.path.size());
    hash.update(reinterpret_cast<const uint8_t *>(entry.path.data()),
                entry.path.size());
    update_u64(hash, entry.content.size());
    hash.update(entry.content.data(), entry.content.size());
  }
  return emlx_sha256_hex(hash.final());
}

std::string quote(const std::string &value) {
  std::string output = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"')
      output.push_back('\\');
    output.push_back(character);
  }
  output.push_back('"');
  return output;
}

void write_header(const fs::path &output, const std::string &mlx_version,
                  const std::string &variant, const std::string &target,
                  const std::string &mlx_build_id,
                  const std::string &mlx_headers_build_id) {
  std::ostringstream generated;
  generated << "#pragma once\n"
            << "inline constexpr char EMLX_EXPECTED_MLX_VERSION[] = "
            << quote(mlx_version) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_MLX_VARIANT[] = "
            << quote(variant.empty() ? "default" : variant) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_TARGET_TRIPLE[] = "
            << quote(target) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_MLX_BUILD_ID[] = "
            << quote(mlx_build_id) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_MLX_HEADERS_BUILD_ID[] = "
            << quote(mlx_headers_build_id) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_COMPILER_FAMILY[] = "
            << quote(EMLX_ACTUAL_COMPILER_FAMILY) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_COMPILER_MAJOR[] = "
            << quote(EMLX_ACTUAL_COMPILER_MAJOR) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_ARCHITECTURE_ABI[] = "
            << quote(EMLX_ACTUAL_ARCHITECTURE_ABI) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_CXX_LANGUAGE_MODE[] = "
            << quote(EMLX_ACTUAL_CXX_LANGUAGE_MODE) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_CXX_STDLIB_FAMILY[] = "
            << quote(EMLX_ACTUAL_CXX_STDLIB_FAMILY) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_CXX_STDLIB_ABI[] = "
            << quote(EMLX_ACTUAL_CXX_STDLIB_ABI) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_EXCEPTIONS_MODE[] = "
            << quote(EMLX_ACTUAL_EXCEPTIONS_MODE) << ";\n"
            << "inline constexpr char EMLX_EXPECTED_RTTI_MODE[] = "
            << quote(EMLX_ACTUAL_RTTI_MODE) << ";\n";
  const std::string content = generated.str();
  if (fs::exists(output)) {
    const auto current = read_file(output);
    if (current.size() == content.size() &&
        std::equal(current.begin(), current.end(), content.begin()))
      return;
  }

  fs::create_directories(output.parent_path());
  const fs::path temporary = output.string() + ".tmp." + std::to_string(getpid());
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  if (!stream)
    throw std::runtime_error("cannot create compatibility header");
  stream.write(content.data(), content.size());
  stream.close();
  if (!stream)
    throw std::runtime_error("cannot write compatibility header");
  fs::rename(temporary, output);
}

void atomic_write(const fs::path &output, const std::string &content) {
  if (fs::exists(output)) {
    const auto current = read_file(output);
    if (current.size() == content.size() &&
        std::equal(current.begin(), current.end(), content.begin()))
      return;
  }
  fs::create_directories(output.parent_path());
  const fs::path temporary = output.string() + ".tmp." + std::to_string(getpid());
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  if (!stream)
    throw std::runtime_error("cannot create generated identity artifact");
  stream.write(content.data(), static_cast<std::streamsize>(content.size()));
  stream.close();
  if (!stream)
    throw std::runtime_error("cannot write generated identity artifact");
  fs::rename(temporary, output);
}

void write_literal_header(const fs::path &output, const std::string &symbol,
                          const std::string &value) {
  std::array<uint8_t, 32> decoded{};
  if (!emlx_sha256_parse_hex(value, decoded))
    throw std::runtime_error("generated identity must be 64 lowercase hexadecimal bytes");
  const std::string content =
      "#pragma once\ninline constexpr char " + symbol + "[65] =\n    \"" +
      value + "\";\n";
  atomic_write(output, content);
}

std::string read_literal(const fs::path &path, const std::string &symbol) {
  const auto bytes = read_file(path);
  const std::string source(bytes.begin(), bytes.end());
  const size_t start = source.find(symbol);
  if (start == std::string::npos)
    throw std::runtime_error("generated compatibility identity is missing");
  const size_t assignment = source.find('=', start + symbol.size());
  if (assignment == std::string::npos)
    throw std::runtime_error("generated compatibility identity is malformed");
  size_t quote_start = assignment + 1;
  while (quote_start < source.size() &&
         std::isspace(static_cast<unsigned char>(source[quote_start])))
    ++quote_start;
  if (quote_start >= source.size() || source[quote_start] != '"')
    throw std::runtime_error("generated compatibility identity is malformed");
  const size_t value_start = quote_start + 1;
  const size_t end = source.find('"', value_start);
  if (end == std::string::npos)
    throw std::runtime_error("generated compatibility identity is malformed");
  const std::string value = source.substr(value_start, end - value_start);
  std::array<uint8_t, 32> decoded{};
  if (!emlx_sha256_parse_hex(value, decoded))
    throw std::runtime_error("generated compatibility identity is invalid");
  return value;
}

bool within(const fs::path &child, const fs::path &root) {
  const auto relative = child.lexically_relative(root);
  if (relative.empty())
    return child == root;
  const auto text = relative.generic_string();
  return text != ".." && !text.starts_with("../");
}

bool has_mlx_component(const fs::path &path) {
  return std::any_of(path.begin(), path.end(), [](const fs::path &part) {
    return part == "mlx";
  });
}

std::string byte_sorted_relative(const fs::path &path, const fs::path &root,
                                 const char *prefix) {
  const auto relative = path.lexically_relative(root);
  if (relative.empty() || relative.generic_string().starts_with("../"))
    throw std::runtime_error("manifest dependency escapes its logical root");
  return std::string(prefix) + "/" + relative.generic_string();
}

std::vector<fs::path> authoritative_dependencies(const fs::path &depfile,
                                                 const std::string &source) {
  const auto rules = emlx_parse_make_depfile(depfile);
  std::vector<fs::path> dependencies;
  size_t matching = 0;
  for (const auto &rule : rules) {
    if (std::find(rule.targets.begin(), rule.targets.end(), source) ==
        rule.targets.end())
      continue;
    ++matching;
    dependencies.reserve(dependencies.size() + rule.dependencies.size());
    for (const auto &dependency : rule.dependencies) {
      std::error_code error;
      const auto canonical = fs::canonical(dependency, error);
      if (error || !fs::is_regular_file(canonical))
        throw std::runtime_error("plugin depfile contains a missing dependency");
      dependencies.push_back(canonical);
    }
  }
  if (matching != 1)
    throw std::runtime_error(
        "plugin depfile must have exactly one authoritative source rule");
  return dependencies;
}

void add_manifest_entry(std::vector<ManifestEntry> &entries,
                        std::set<std::pair<std::string, std::string>> &seen,
                        std::string kind, std::string path,
                        std::vector<uint8_t> content) {
  if (!seen.emplace(kind, path).second)
    return;
  entries.push_back(
      {std::move(kind), std::move(path), std::move(content)});
}

std::string plugin_manifest_id(
    const fs::path &axon_root, const fs::path &emlx_root,
    const fs::path &mlx_root, const fs::path &scan_header,
    const fs::path &actual_mlx_header, const fs::path &compat_header,
    const std::string &mlx_headers_id,
    const std::vector<fs::path> &explicit_policy_inputs,
    const std::vector<std::string> &flags,
    const std::vector<std::pair<std::string, fs::path>> &source_depfiles) {
  const fs::path axon = fs::canonical(axon_root);
  const fs::path emlx = fs::canonical(emlx_root);
  const fs::path mlx = fs::canonical(mlx_root);
  const fs::path scan = fs::canonical(scan_header);
  const fs::path actual_headers = fs::canonical(actual_mlx_header);
  const fs::path compat = fs::canonical(compat_header);
  const fs::path emlx_public_headers = compat.parent_path();
  std::vector<ManifestEntry> entries;
  std::set<std::pair<std::string, std::string>> seen;
  bool saw_scan = false;
  bool saw_actual_headers = false;
  bool saw_compat = false;
  bool saw_mlx = false;

  for (const auto &[source, depfile] : source_depfiles) {
    for (const auto &dependency : authoritative_dependencies(depfile, source)) {
      if (dependency == scan) {
        saw_scan = true;
        continue;
      }
      if (dependency == actual_headers) {
        saw_actual_headers = true;
        continue;
      }
      if (dependency == compat) {
        saw_compat = true;
        add_manifest_entry(entries, seen, "FILE", "generated:emlx_compat",
                           read_file(dependency));
      } else if (within(dependency, emlx_public_headers)) {
        add_manifest_entry(
            entries, seen, "FILE",
            byte_sorted_relative(dependency, emlx_public_headers,
                                 "emlx-public"),
            read_file(dependency));
      } else if (within(dependency, mlx)) {
        saw_mlx = true;
        add_manifest_entry(entries, seen, "FILE",
                           byte_sorted_relative(dependency, mlx, "mlx"),
                           read_file(dependency));
      } else if (within(dependency, axon)) {
        add_manifest_entry(entries, seen, "FILE",
                           byte_sorted_relative(dependency, axon, "emlx_axon"),
                           read_file(dependency));
      } else if (within(dependency, emlx)) {
        add_manifest_entry(entries, seen, "FILE",
                           byte_sorted_relative(dependency, emlx, "emlx"),
                           read_file(dependency));
      } else if (has_mlx_component(dependency)) {
        throw std::runtime_error(
            "plugin resolved an MLX dependency outside the selected include root");
      }
    }
  }
  if (!saw_scan || !saw_actual_headers || !saw_compat || !saw_mlx) {
    std::ostringstream missing;
    missing << "plugin scan closure is missing:";
    if (!saw_scan)
      missing << " scan_build_id";
    if (!saw_actual_headers)
      missing << " mlx_headers_build_id";
    if (!saw_compat)
      missing << " emlx_compat";
    if (!saw_mlx)
      missing << " mlx_headers";
    throw std::runtime_error(missing.str());
  }

  add_manifest_entry(entries, seen, "VALUE", "mlx_headers_build_id",
                     std::vector<uint8_t>(mlx_headers_id.begin(),
                                          mlx_headers_id.end()));
  for (const auto &input : explicit_policy_inputs) {
    const fs::path canonical = fs::canonical(input);
    std::string logical;
    if (within(canonical, axon))
      logical = byte_sorted_relative(canonical, axon, "emlx_axon");
    else if (within(canonical, emlx))
      logical = byte_sorted_relative(canonical, emlx, "emlx");
    else
      throw std::runtime_error("explicit plugin manifest input has no logical root");
    add_manifest_entry(entries, seen, "POLICY", logical, read_file(canonical));
  }
  for (size_t index = 0; index < flags.size(); ++index) {
    add_manifest_entry(entries, seen, "FLAG", std::to_string(index),
                       std::vector<uint8_t>(flags[index].begin(), flags[index].end()));
  }

  std::sort(entries.begin(), entries.end(), [](const auto &left, const auto &right) {
    return std::tie(left.kind, left.path) < std::tie(right.kind, right.path);
  });
  EMLXSHA256 hash;
  static constexpr char domain[] = "EMLX_QWEN3_PLUGIN_BUILD_MANIFEST_V1\n";
  hash.update(reinterpret_cast<const uint8_t *>(domain), sizeof(domain) - 1);
  update_u64(hash, entries.size());
  for (const auto &entry : entries) {
    update_u64(hash, entry.kind.size());
    hash.update(reinterpret_cast<const uint8_t *>(entry.kind.data()),
                entry.kind.size());
    update_u64(hash, entry.path.size());
    hash.update(reinterpret_cast<const uint8_t *>(entry.path.data()),
                entry.path.size());
    update_u64(hash, entry.content.size());
    hash.update(entry.content.data(), entry.content.size());
  }
  return emlx_sha256_hex(hash.final());
}

void verify_plugin_closure(
    const fs::path &mlx_root, const fs::path &scan_header,
    const fs::path &final_header,
    const std::vector<std::tuple<std::string, fs::path, fs::path>> &depfiles) {
  const fs::path mlx = fs::canonical(mlx_root);
  const fs::path scan = fs::canonical(scan_header);
  const fs::path final = fs::canonical(final_header);
  for (const auto &[source, scan_depfile, final_depfile] : depfiles) {
    const auto scan_dependencies = authoritative_dependencies(scan_depfile, source);
    const auto final_dependencies = authoritative_dependencies(final_depfile, source);
    auto normalize = [&](const std::vector<fs::path> &dependencies,
                         const fs::path &required, const fs::path &forbidden) {
      std::set<fs::path> result;
      bool saw_required = false;
      for (const auto &dependency : dependencies) {
        if (dependency == required) {
          saw_required = true;
          continue;
        }
        if (dependency == forbidden)
          throw std::runtime_error("plugin depfile selected the wrong build-ID header");
        if (within(dependency, mlx)) {
          // The selected root is accepted.
        } else if (has_mlx_component(dependency)) {
          throw std::runtime_error(
              "plugin resolved an MLX dependency outside the selected include root");
        }
        result.insert(dependency);
      }
      if (!saw_required)
        throw std::runtime_error("plugin depfile did not select its build-ID header");
      return result;
    };
    if (normalize(scan_dependencies, scan, final) !=
        normalize(final_dependencies, final, scan))
      throw std::runtime_error("plugin scan and final dependency closures differ");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "sha256") {
      std::cout << hash_file(argv[2]) << "\n";
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "mlx-headers") {
      std::cout << hash_headers(argv[2]) << "\n";
      return 0;
    }
    if (argc >= 7 && std::string(argv[1]) == "verify-host-deps") {
      if (((argc - 5) % 2) != 0)
        throw std::runtime_error("verify-host-deps requires source/depfile pairs");
      std::vector<std::pair<std::string, fs::path>> pairs;
      for (int index = 5; index < argc; index += 2)
        pairs.emplace_back(argv[index], argv[index + 1]);
      emlx_verify_host_depfiles(argv[2], argv[3], argv[4], pairs);
      return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "write-id-header") {
      write_literal_header(argv[2], argv[3], argv[4]);
      return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "write-mlx-header-id") {
      const std::string actual = hash_headers(argv[2]);
      if (actual != read_literal(argv[4], "EMLX_EXPECTED_MLX_HEADERS_BUILD_ID"))
        throw std::runtime_error(
            "plugin MLX public-header identity does not match EMLX");
      write_literal_header(argv[3], "EMLX_QWEN3_MLX_HEADERS_BUILD_ID", actual);
      return 0;
    }
    if (argc >= 13 && std::string(argv[1]) == "plugin-build-id") {
      const fs::path axon_root = argv[2];
      const fs::path emlx_root = argv[3];
      const fs::path mlx_root = argv[4];
      const fs::path scan_header = argv[5];
      const fs::path actual_mlx_header = argv[6];
      const fs::path compat_header = argv[7];
      const std::string mlx_headers_id = read_literal(
          actual_mlx_header, "EMLX_QWEN3_MLX_HEADERS_BUILD_ID");
      const fs::path output_header = argv[8];
      const fs::path output_text = argv[9];
      std::vector<fs::path> policy_inputs;
      std::vector<std::string> flags;
      std::vector<std::pair<std::string, fs::path>> depfiles;
      int index = 10;
      while (index < argc) {
        const std::string option = argv[index++];
        if (option == "--policy" && index < argc) {
          policy_inputs.emplace_back(argv[index++]);
        } else if (option == "--flag" && index < argc) {
          flags.emplace_back(argv[index++]);
        } else if (option == "--dep" && index + 1 < argc) {
          const std::string source = argv[index++];
          depfiles.emplace_back(source, argv[index++]);
        } else {
          throw std::runtime_error("invalid plugin-build-id argument");
        }
      }
      if (depfiles.empty())
        throw std::runtime_error("plugin build has no translation units");
      const std::string build_id = plugin_manifest_id(
          axon_root, emlx_root, mlx_root, scan_header, actual_mlx_header,
          compat_header, mlx_headers_id, policy_inputs, flags, depfiles);
      write_literal_header(output_header, "EMLX_QWEN3_PLUGIN_BUILD_ID", build_id);
      atomic_write(output_text, build_id + "\n");
      std::cout << build_id << "\n";
      return 0;
    }
    if (argc >= 8 && std::string(argv[1]) == "verify-plugin-deps") {
      if (((argc - 5) % 3) != 0)
        throw std::runtime_error(
            "verify-plugin-deps requires source/scan/final triples");
      std::vector<std::tuple<std::string, fs::path, fs::path>> depfiles;
      for (int index = 5; index < argc; index += 3)
        depfiles.emplace_back(argv[index], argv[index + 1], argv[index + 2]);
      verify_plugin_closure(argv[2], argv[3], argv[4], depfiles);
      return 0;
    }
    if (argc != 7) {
      std::cerr << "usage: emlx_build_identity sha256 <file> | mlx-headers "
                   "<include-root> | <libmlx> <include-root> <mlx-version> "
                   "<variant> <target> <output> | verify-host-deps "
                   "<mlx-root> <staged-header> <published-header> "
                   "<source> <depfile>... | write-id-header <output> <symbol> "
                   "<sha256> | write-mlx-header-id <mlx-root> <output> "
                   "<expected-sha256> | plugin-build-id ... | "
                   "verify-plugin-deps ...\n";
      return 2;
    }
    const std::string library_hash = hash_file(argv[1]);
    const std::string header_hash = hash_headers(argv[2]);
    write_header(argv[6], argv[3], argv[4], argv[5], library_hash, header_hash);
    std::cout << library_hash << " " << header_hash << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "emlx_build_identity: " << error.what() << "\n";
    return 1;
  }
}
