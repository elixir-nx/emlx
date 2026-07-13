#include "emlx_depfile.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr size_t kDepfileMaxBytes = 64U * 1024U * 1024U;
constexpr size_t kDepfileMaxRules = 1U << 20;
constexpr size_t kDepfileMaxTokens = 1U << 22;

std::string read_bounded(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read depfile");
  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  if (end < 0 || static_cast<uint64_t>(end) > kDepfileMaxBytes)
    throw std::runtime_error("depfile exceeds the size limit");
  std::string bytes(static_cast<size_t>(end), '\0');
  input.seekg(0, std::ios::beg);
  if (!bytes.empty() && !input.read(bytes.data(), bytes.size()))
    throw std::runtime_error("cannot read complete depfile");
  if (bytes.find('\0') != std::string::npos)
    throw std::runtime_error("depfile contains a NUL byte");
  return bytes;
}

std::string remove_continuations(const std::string &input) {
  std::string output;
  output.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] == '\\' && i + 1 < input.size() && input[i + 1] == '\n') {
      output.push_back(' ');
      ++i;
      continue;
    }
    if (input[i] == '\\' && i + 2 < input.size() && input[i + 1] == '\r' &&
        input[i + 2] == '\n') {
      output.push_back(' ');
      i += 2;
      continue;
    }
    output.push_back(input[i]);
  }
  return output;
}

size_t find_unescaped_colon(const std::string &line) {
  bool escaped = false;
  for (size_t i = 0; i < line.size(); ++i) {
    if (escaped) {
      escaped = false;
    } else if (line[i] == '\\') {
      escaped = true;
    } else if (line[i] == ':') {
      return i;
    }
  }
  if (escaped)
    throw std::runtime_error("depfile rule ends with an escape");
  return std::string::npos;
}

std::vector<std::string> tokenize_make(const std::string &value) {
  std::vector<std::string> tokens;
  std::string current;
  bool escaped = false;
  for (unsigned char byte : value) {
    if (escaped) {
      current.push_back(static_cast<char>(byte));
      escaped = false;
    } else if (byte == '\\') {
      escaped = true;
    } else if (std::isspace(byte)) {
      if (!current.empty()) {
        tokens.push_back(std::move(current));
        current.clear();
      }
    } else {
      current.push_back(static_cast<char>(byte));
    }
    if (tokens.size() > kDepfileMaxTokens)
      throw std::runtime_error("depfile contains too many tokens");
  }
  if (escaped)
    throw std::runtime_error("depfile token ends with an escape");
  if (!current.empty())
    tokens.push_back(std::move(current));
  return tokens;
}

bool within(const fs::path &child, const fs::path &root) {
  const auto relative = child.lexically_relative(root);
  if (relative.empty())
    return child == root;
  const auto text = relative.generic_string();
  return text != ".." && !text.starts_with("../");
}

bool names_mlx_namespace(const fs::path &path) {
  for (const auto &component : path) {
    if (component == "mlx")
      return true;
  }
  return false;
}

fs::path canonical_existing(const fs::path &path, const char *kind) {
  std::error_code error;
  const fs::path canonical = fs::canonical(path, error);
  if (error)
    throw std::runtime_error(std::string(kind) + " does not resolve: " +
                             path.string());
  if (!fs::is_regular_file(canonical, error) || error)
    throw std::runtime_error(std::string(kind) + " is not a regular file");
  return canonical;
}

void atomic_copy_if_changed(const fs::path &source, const fs::path &output) {
  std::ifstream input(source, std::ios::binary);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input.good() && !input.eof())
    throw std::runtime_error("cannot read staged compatibility header");
  const std::string content = buffer.str();

  if (fs::exists(output)) {
    std::ifstream current(output, std::ios::binary);
    std::ostringstream current_buffer;
    current_buffer << current.rdbuf();
    if (current && current_buffer.str() == content)
      return;
  }

  fs::create_directories(output.parent_path());
  const fs::path temporary =
      output.string() + ".tmp." + std::to_string(static_cast<long long>(getpid()));
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream)
      throw std::runtime_error("cannot create published compatibility header");
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.close();
    if (!stream)
      throw std::runtime_error("cannot write published compatibility header");
  }
  std::error_code error;
  fs::rename(temporary, output, error);
  if (error) {
    fs::remove(temporary);
    throw std::runtime_error("cannot publish compatibility header atomically");
  }
}

} // namespace

std::vector<EMLXDepfileRule> emlx_parse_make_depfile(const fs::path &path) {
  const std::string input = remove_continuations(read_bounded(path));
  std::vector<EMLXDepfileRule> rules;
  std::istringstream lines(input);
  std::string line;
  while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (std::all_of(line.begin(), line.end(), [](unsigned char byte) {
          return std::isspace(byte);
        }))
      continue;
    const size_t colon = find_unescaped_colon(line);
    if (colon == std::string::npos)
      throw std::runtime_error("depfile rule has no unescaped colon");
    auto targets = tokenize_make(line.substr(0, colon));
    auto dependencies = tokenize_make(line.substr(colon + 1));
    if (targets.empty())
      throw std::runtime_error("depfile rule has no target");
    rules.push_back({std::move(targets), std::move(dependencies)});
    if (rules.size() > kDepfileMaxRules)
      throw std::runtime_error("depfile contains too many rules");
  }
  if (rules.empty())
    throw std::runtime_error("depfile contains no rules");
  return rules;
}

void emlx_verify_host_depfiles(
    const fs::path &mlx_include_root, const fs::path &staged_compat_header,
    const fs::path &published_compat_header,
    const std::vector<std::pair<std::string, fs::path>> &source_depfiles) {
  if (source_depfiles.empty())
    throw std::runtime_error("host source inventory is empty");

  const fs::path canonical_mlx_root = fs::canonical(mlx_include_root);
  const fs::path canonical_staged =
      canonical_existing(staged_compat_header, "staged compatibility header");
  std::set<std::string> sources;
  std::set<fs::path> depfiles;
  bool saw_mlx = false;

  for (const auto &[source_name, depfile] : source_depfiles) {
    if (source_name.empty() || !sources.insert(source_name).second)
      throw std::runtime_error("host source inventory contains a duplicate source");
    const fs::path canonical_depfile = canonical_existing(depfile, "depfile");
    if (!depfiles.insert(canonical_depfile).second)
      throw std::runtime_error("host source inventory contains a duplicate depfile");

    const auto rules = emlx_parse_make_depfile(canonical_depfile);
    size_t source_rules = 0;
    bool source_dependency = false;
    bool staged_dependency = false;
    for (const auto &rule : rules) {
      if (std::find(rule.targets.begin(), rule.targets.end(), source_name) !=
          rule.targets.end()) {
        ++source_rules;
        for (const auto &dependency : rule.dependencies) {
          const fs::path canonical_dependency =
              canonical_existing(dependency, "depfile dependency");
          if (canonical_dependency == canonical_staged)
            staged_dependency = true;
          if (fs::equivalent(canonical_dependency, fs::path(source_name)))
            source_dependency = true;
          if (within(canonical_dependency, canonical_mlx_root)) {
            saw_mlx = true;
          } else if (names_mlx_namespace(fs::path(dependency))) {
            throw std::runtime_error(
                "compiler resolved an MLX dependency outside the selected include root");
          }
        }
      }
    }
    if (source_rules != 1)
      throw std::runtime_error("depfile must contain exactly one authoritative source rule");
    if (!source_dependency)
      throw std::runtime_error("depfile does not contain its source dependency");
    if (!staged_dependency)
      throw std::runtime_error("depfile did not consume the staged compatibility header");
  }

  if (!saw_mlx)
    throw std::runtime_error("host dependency closure contains no selected MLX header");
  atomic_copy_if_changed(canonical_staged, published_compat_header);
}
