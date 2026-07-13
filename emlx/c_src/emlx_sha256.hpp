#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

class EMLXSHA256 {
public:
  EMLXSHA256();
  void update(const uint8_t *data, size_t size);
  std::array<uint8_t, 32> final();

private:
  void transform(const uint8_t block[64]);

  std::array<uint32_t, 8> state_;
  std::array<uint8_t, 64> buffer_{};
  uint64_t total_size_ = 0;
  size_t buffer_size_ = 0;
  bool finalized_ = false;
};

std::array<uint8_t, 32> emlx_sha256_bytes(const uint8_t *data, size_t size);
bool emlx_sha256_file_descriptor(int fd, std::array<uint8_t, 32> &digest,
                                 std::string &error);
std::string emlx_sha256_hex(const std::array<uint8_t, 32> &digest);
bool emlx_sha256_parse_hex(const std::string &hex,
                           std::array<uint8_t, 32> &digest);
