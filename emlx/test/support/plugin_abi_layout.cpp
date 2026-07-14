#include "emlx/plugin/abi.hpp"

#include <cstdint>
#include <iostream>

int main() {
  constexpr uint64_t expected = 0x072ddc26bff8c5c1ULL;
  constexpr uint64_t offsets[] = {0, 8, 24};
  const auto actual = emlx_plugin_layout_record(
      14695981039346656037ULL, 7, 40, 8, offsets, 3);

  if (actual != expected ||
      actual != emlx_plugin_layout_conformance_vector_v1())
    return 1;

  std::cout << std::hex << actual << " " << EMLX_PLUGIN_LAYOUT_ABI_HASH_V1
            << "\n";
  return 0;
}
