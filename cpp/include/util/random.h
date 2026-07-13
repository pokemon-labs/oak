#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>
#include <type_traits>

class mt19937 {
private:
  std::mt19937::result_type seed;
  std::mt19937 engine;
  std::uniform_real_distribution<double> uniform_;
  std::uniform_int_distribution<uint64_t> uniform_64_;

public:
  mt19937() = default;
  mt19937(std::mt19937::result_type seed)
      : seed(seed), engine(std::mt19937{seed}) {}

  std::mt19937::result_type get_seed() const noexcept { return seed; }

  std::mt19937::result_type random_seed() noexcept {
    return uniform_64_(engine);
  }

  // Uniform random in (0, 1)
  double uniform() noexcept { return uniform_(engine); }

  // Random integer in [0, n)
  int random_int(int n) noexcept {
    assert(n != 0);
    return uniform_64_(engine) % n;
  }

  uint64_t uniform_64() noexcept { return uniform_64_(engine); }

  template <typename Container>
  uint32_t sample_pdf(const Container &input) noexcept {
    double p = uniform();
    for (uint32_t i = 0; i < input.size(); ++i) {
      p -= static_cast<double>(input[i]);
      if (p <= 0) {
        return i;
      }
    }
    return 0;
  }

  template <template <typename...> typename Vector, typename T>
    requires(T::get_d())
  uint32_t sample_pdf(const Vector<T> &input) noexcept {
    double p = uniform();
    for (uint32_t i = 0; i < input.size(); ++i) {
      p -= input[i].get_d();
      if (p <= 0) {
        return i;
      }
    }
    return 0;
  }

  void discard(size_t n) { engine.discard(n); }
};

class fast_prng {
  uint8_t *state_ptr;

  static inline uint32_t rotl32(uint32_t x, int k) noexcept {
    return (x << k) | (x >> (32 - k));
  }

  // Access the two 32-bit words stored in state_ptr
  inline uint32_t &s0() noexcept {
    return *reinterpret_cast<uint32_t *>(state_ptr);
  }
  inline uint32_t &s1() noexcept {
    return *reinterpret_cast<uint32_t *>(state_ptr + 4);
  }

  uint32_t next32() noexcept {
    uint32_t result = rotl32(s0() + s1(), 9) + s0();

    s1() ^= s0();
    s0() = rotl32(s0(), 13) ^ s1() ^ (s1() << 5);
    s1() = rotl32(s1(), 28);

    return result;
  }

  uint64_t next64() noexcept {
    return (static_cast<uint64_t>(next32()) << 32) | next32();
  }

public:
  explicit fast_prng(uint8_t *external_state) : state_ptr(external_state) {}

  static void seed(uint8_t *buffer, uint64_t seed) {
    std::seed_seq seq{static_cast<uint32_t>(seed),
                      static_cast<uint32_t>(seed >> 32)};
    uint32_t seeds[2];
    seq.generate(seeds, seeds + 2);
    std::memcpy(buffer, seeds, 8);
  }

  uint64_t uniform_64() noexcept { return next64(); }

  double uniform() noexcept {
    // Like std::uniform_real_distribution, top 53 bits of next64() in [0, 1)
    return (next64() >> 11) * (1.0 / (1ull << 53));
  }

  int random_int(int n) noexcept { return static_cast<int>(next32() % n); }

  uint64_t random_seed() noexcept { return next64(); }

  void discard(size_t n) noexcept {
    for (size_t i = 0; i < n; ++i)
      next32();
  }

  template <typename Container>
  int sample_pdf(const Container &input) noexcept {
    while (true) {
      double p = uniform();
      for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        p -= static_cast<double>(input[i]);
        if (p <= 0.0) {
          return i;
        }
      }
      // std::cerr << "sample_pdf retried\n";
    }
  }
};
