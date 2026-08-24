#pragma once

#include <libpkmn/pkmn.h>

#include <istream>

// std::tuple<uint16_t, uint16_t, uint16_t> VERSION = {1, 1, 0};

namespace Train::Battle {

template <typename in_type, typename out_type>
constexpr out_type compress_probs(in_type x) {
  if constexpr (std::is_integral_v<out_type>) {
    if constexpr (std::is_signed_v<out_type>) {
      static_assert(!std::is_same_v<out_type, out_type>,
                    "Signed integral types not supported to store probs.");
      return {};
    } else {
      return x * std::numeric_limits<out_type>::max();
    }
  } else {
    return static_cast<out_type>(x);
  }
}

template <typename in_type, typename out_type>
constexpr out_type uncompress_probs(in_type x) {
  if constexpr (std::is_integral_v<in_type>) {
    return x / static_cast<out_type>(std::numeric_limits<in_type>::max());
  } else {
    return static_cast<out_type>(x);
  }
}

template <typename policy_type = uint16_t, typename value_type = uint16_t,
          typename offset_type = uint32_t>
struct CompressedFramesImpl {

  using Offset = offset_type;
  using FrameCount = uint16_t;

  struct Update {

    struct Side {
      uint8_t k;
      pkmn_choice choice;
      std::array<policy_type, 9> empirical;
      std::array<policy_type, 9> nash;

      Side() = default;
      Side(const auto &side, const auto choice) : k{side.k}, choice{choice} {
        for (auto i = 0; i < k; ++i) {
          empirical[i] = compress_probs<double, policy_type>(side.empirical[i]);
          nash[i] = compress_probs<double, policy_type>(side.nash[i]);
        }
      }
    };

    // we store
    using MN = uint8_t;
    using Iter = uint32_t;

    Iter iterations;
    value_type empirical_value;
    value_type nash_value;

    Side p1;
    Side p2;

    static constexpr size_t n_bytes_static(auto m, auto n) {
      // The leading byte is the combination m/n
      // Max: 1 + 2 + 4 + 4 + 72 = 83
      return sizeof(MN) + 2 * sizeof(pkmn_choice) + sizeof(Iter) +
             2 * sizeof(value_type) + 2 * (m + n) * sizeof(policy_type);
    }

    Update() = default;
    Update(const auto &search_output, pkmn_choice c1, pkmn_choice c2)
        : p1{search_output.p1, c1}, p2{search_output.p2, c2},
          iterations{static_cast<uint32_t>(search_output.iterations)},
          empirical_value{compress_probs<double, value_type>(
              search_output.empirical_value)},
          nash_value{
              compress_probs<double, value_type>(search_output.nash_value)} {}

    size_t n_bytes() const { return n_bytes_static(p1.k, p2.k); }

    bool write(char *const buffer) const {
      auto index = 0;
      const auto write = [buffer, &index](const auto &data, auto s) {
        if constexpr (requires { data.data(); }) {
          std::memcpy(buffer + index,
                      reinterpret_cast<const char *>(data.data()), s);
        } else {
          std::memcpy(buffer + index, reinterpret_cast<const char *>(&data), s);
        }
        index += s;
      };
      MN mn = (p1.k - 1) | ((p2.k - 1) << 4);
      write(mn, 1);
      write(p1.choice, 1);
      write(p2.choice, 1);
      write(iterations, sizeof(Iter));
      write(empirical_value, sizeof(value_type));
      write(nash_value, sizeof(value_type));
      write(p1.empirical, p1.k * sizeof(policy_type));
      write(p1.nash, p1.k * sizeof(policy_type));
      write(p2.empirical, p2.k * sizeof(policy_type));
      write(p2.nash, p2.k * sizeof(policy_type));
      assert(index == n_bytes());
      return true;
    }

    bool read(const char *buffer) {
      auto index = 0;
      const auto read = [buffer, &index](auto &data, auto s) {
        if constexpr (requires { data.data(); }) {
          std::memcpy(reinterpret_cast<char *>(data.data()), buffer + index, s);
        } else {
          std::memcpy(reinterpret_cast<char *>(&data), buffer + index, s);
        }
        index += s;
      };
      MN mn;
      read(mn, 1);
      p1.k = (mn % 16) + 1;
      p2.k = (mn / 16) + 1;
      read(p1.choice, 1);
      read(p2.choice, 1);
      read(iterations, sizeof(Iter));
      read(empirical_value, sizeof(value_type));
      read(nash_value, sizeof(value_type));
      read(p1.empirical, p1.k * sizeof(policy_type));
      read(p1.nash, p1.k * sizeof(policy_type));
      read(p2.empirical, p2.k * sizeof(policy_type));
      read(p2.nash, p2.k * sizeof(policy_type));
      assert(index == n_bytes());
      return true;
    }

    void write_to_tensor(uint8_t *k, uint8_t *choice, Iter *iter,
                         float *empirical_policies, float *nash_policies,
                         float *ev, float *nv) const {
      k[0] = p1.k;
      k[1] = p2.k;
      choice[0] = p1.choice;
      choice[1] = p2.choice;
      iter[0] = iterations;
      std::fill_n(empirical_policies, 18, 0);
      std::fill_n(nash_policies, 18, 0);
      for (auto i = 0; i < p1.k; ++i) {
        empirical_policies[0 + i] =
            uncompress_probs<policy_type, float>(p1.empirical[i]);
        nash_policies[0 + i] = uncompress_probs<policy_type, float>(p1.nash[i]);
      }
      for (auto i = 0; i < p2.k; ++i) {
        empirical_policies[9 + i] =
            uncompress_probs<policy_type, float>(p2.empirical[i]);
        nash_policies[9 + i] = uncompress_probs<policy_type, float>(p2.nash[i]);
      }
      ev[0] = uncompress_probs<value_type, float>(empirical_value);
      nv[0] = uncompress_probs<value_type, float>(nash_value);
    }
  };

  pkmn_gen1_battle battle;
  pkmn_result result;
  std::vector<Update> updates;

  CompressedFramesImpl() = default;
  CompressedFramesImpl(const pkmn_gen1_battle &battle) : battle{battle} {}

  uint32_t n_bytes() const {
    uint32_t n = sizeof(Offset) + sizeof(FrameCount) +
                 sizeof(pkmn_gen1_battle) + sizeof(pkmn_result);
    for (const auto &update : updates) {
      n += update.n_bytes();
    }
    return n;
  }

  bool write(char *buffer) const {
    auto index = 0;
    *reinterpret_cast<Offset *>(buffer + index) = n_bytes();
    index += sizeof(Offset);
    *reinterpret_cast<FrameCount *>(buffer + index) = updates.size();
    index += sizeof(FrameCount);
    std::memcpy(buffer + index, battle.bytes, sizeof(pkmn_gen1_battle));
    index += sizeof(pkmn_gen1_battle);
    buffer[index] = static_cast<char>(result);
    index += 1;
    for (const auto &update : updates) {
      const auto n = update.n_bytes();
      update.write(buffer + index);
      index += n;
    }
    assert(index == n_bytes());
    return true;
  }

  bool read(const char *buffer) {
    uint32_t index = 0;
    const Offset buffer_length =
        *reinterpret_cast<const Offset *>(buffer + index);
    index += sizeof(Offset);
    const FrameCount n_updates =
        *reinterpret_cast<const FrameCount *>(buffer + index);
    index += sizeof(FrameCount);
    std::memcpy(battle.bytes, buffer + index, sizeof(pkmn_gen1_battle));
    index += sizeof(pkmn_gen1_battle);
    result = buffer[index];
    index += 1;
    while (index < buffer_length) {
      updates.emplace_back();
      auto &update = updates.back();
      update.read(buffer + index);
      index += update.n_bytes();
    }
    assert(updates.size() == n_updates);
    assert(index == buffer_length);
    return true;
  }
};

using CompressedFrames = CompressedFramesImpl<>;

} // namespace Train::Battle
