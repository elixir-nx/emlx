#include "emlx_sha256.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

namespace {

constexpr uint32_t k[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr uint32_t rotate_right(uint32_t value, uint32_t count) {
  return (value >> count) | (value << (32U - count));
}

} // namespace

EMLXSHA256::EMLXSHA256()
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void EMLXSHA256::transform(const uint8_t block[64]) {
  uint32_t words[64];
  for (size_t i = 0; i < 16; ++i) {
    words[i] = (static_cast<uint32_t>(block[i * 4]) << 24U) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16U) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8U) |
               static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (size_t i = 16; i < 64; ++i) {
    const uint32_t s0 = rotate_right(words[i - 15], 7) ^
                        rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3U);
    const uint32_t s1 = rotate_right(words[i - 2], 17) ^
                        rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10U);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (size_t i = 0; i < 64; ++i) {
    const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                          rotate_right(e, 25);
    const uint32_t choice = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + sum1 + choice + k[i] + words[i];
    const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                          rotate_right(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void EMLXSHA256::update(const uint8_t *data, size_t size) {
  if (finalized_)
    throw std::logic_error("SHA-256 context is already finalized");
  if (size > UINT64_MAX - total_size_)
    throw std::overflow_error("SHA-256 input length overflow");
  total_size_ += size;
  while (size > 0) {
    const size_t count = std::min(size, buffer_.size() - buffer_size_);
    std::memcpy(buffer_.data() + buffer_size_, data, count);
    buffer_size_ += count;
    data += count;
    size -= count;
    if (buffer_size_ == buffer_.size()) {
      transform(buffer_.data());
      buffer_size_ = 0;
    }
  }
}

std::array<uint8_t, 32> EMLXSHA256::final() {
  if (finalized_)
    throw std::logic_error("SHA-256 context is already finalized");
  const uint64_t bit_size = total_size_ * 8U;
  buffer_[buffer_size_++] = 0x80U;
  if (buffer_size_ > 56) {
    std::fill(buffer_.begin() + buffer_size_, buffer_.end(), 0);
    transform(buffer_.data());
    buffer_size_ = 0;
  }
  std::fill(buffer_.begin() + buffer_size_, buffer_.begin() + 56, 0);
  for (size_t i = 0; i < 8; ++i)
    buffer_[63 - i] = static_cast<uint8_t>(bit_size >> (i * 8U));
  transform(buffer_.data());
  finalized_ = true;

  std::array<uint8_t, 32> result{};
  for (size_t i = 0; i < state_.size(); ++i) {
    result[i * 4] = static_cast<uint8_t>(state_[i] >> 24U);
    result[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16U);
    result[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8U);
    result[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
  }
  return result;
}

std::array<uint8_t, 32> emlx_sha256_bytes(const uint8_t *data, size_t size) {
  EMLXSHA256 hash;
  hash.update(data, size);
  return hash.final();
}

bool emlx_sha256_file_descriptor(int fd, std::array<uint8_t, 32> &digest,
                                 std::string &error) {
  EMLXSHA256 hash;
  std::array<uint8_t, 64 * 1024> buffer{};
  if (lseek(fd, 0, SEEK_SET) < 0) {
    error = std::strerror(errno);
    return false;
  }
  while (true) {
    const ssize_t count = read(fd, buffer.data(), buffer.size());
    if (count == 0)
      break;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      error = std::strerror(errno);
      return false;
    }
    hash.update(buffer.data(), static_cast<size_t>(count));
  }
  digest = hash.final();
  return true;
}

std::string emlx_sha256_hex(const std::array<uint8_t, 32> &digest) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result(64, '0');
  for (size_t i = 0; i < digest.size(); ++i) {
    result[i * 2] = hex[digest[i] >> 4U];
    result[i * 2 + 1] = hex[digest[i] & 0x0fU];
  }
  return result;
}

bool emlx_sha256_parse_hex(const std::string &hex,
                           std::array<uint8_t, 32> &digest) {
  if (hex.size() != 64)
    return false;
  auto nibble = [](char value) -> int {
    if (value >= '0' && value <= '9')
      return value - '0';
    if (value >= 'a' && value <= 'f')
      return value - 'a' + 10;
    return -1;
  };
  for (size_t i = 0; i < digest.size(); ++i) {
    const int high = nibble(hex[i * 2]);
    const int low = nibble(hex[i * 2 + 1]);
    if (high < 0 || low < 0)
      return false;
    digest[i] = static_cast<uint8_t>((high << 4) | low);
  }
  return true;
}
