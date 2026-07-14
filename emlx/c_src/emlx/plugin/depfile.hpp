#pragma once

#include <filesystem>
#include <string>
#include <vector>

struct EMLXDepfileRule {
  std::vector<std::string> targets;
  std::vector<std::string> dependencies;
};

std::vector<EMLXDepfileRule>
emlx_parse_make_depfile(const std::filesystem::path &path);

void emlx_verify_host_depfiles(
    const std::filesystem::path &mlx_include_root,
    const std::filesystem::path &staged_compat_header,
    const std::filesystem::path &published_compat_header,
    const std::vector<std::pair<std::string, std::filesystem::path>>
        &source_depfiles);
