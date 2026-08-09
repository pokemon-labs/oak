#pragma once

#include <libpkmn/data.h>
#include <nn/battle/cache.h>
#include <nn/battle/network.h>
#include <search/bandit/exp3.h>
#include <search/bandit/pexp3.h>
#include <search/bandit/pucb.h>
#include <search/bandit/ucb.h>
#include <search/bandit/ucb1.h>
#include <search/mcts.h>
#include <util/file-lock.h>
#include <util/strings.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <variant>

namespace Py::Search {

struct Eval;

struct Eval {
  using Variant = std::variant<MCTS::MonteCarlo, PokeEngine::Eval,
                               std::shared_ptr<NN::Battle::NetworkBase>>;
  Variant data;
  template <class T, class... Args>
  Eval(std::in_place_type_t<T>, Args &&...args)
      : data(std::in_place_type<T>, std::forward<Args>(args)...) {}

  bool is_network() const noexcept {
    return std::holds_alternative<std::shared_ptr<NN::Battle::NetworkBase>>(
        data);
  }
  Network network() {
    Network network;
    network.data = std::get<std::shared_ptr<NN::Battle::NetworkBase>>(data);
    return network;
  }
};

class Network : public Eval {
public:
  using NetworkPtr = std::shared_ptr<NN::Battle::NetworkBase>;
  Network(int act = 1) : Eval(std::in_place_type<NetworkPtr>) {
    switch (static_cast<NN::Activation>(act)) {
    case NN::Activation::relu: {
      this->data = std::make_shared<NN::Battle::Network>();
      return;
    }
    case NN::Activation::clamp: {
      this->data = std::make_shared<NN::Battle::NetworkClamped>();
      return;
    }
    default: {
      throw std::runtime_error{"Invalid activation."};
    }
    }
  }
  auto &get() { return std::get<NetworkPtr>(this->data); }
  const auto &get() const { return std::get<NetworkPtr>(this->data); }

  void zero_initialize() { data = std::make_shared<NN::Battle::Network>(); }

  bool read_parameters(const std::string &path) {
    auto [file, fd] = FileLock::try_open_file(path);
    FileLock::FdGuard guard{fd};
    struct Header {
      uint8_t bytes[8];
    };
    const auto read_params = [&](auto &network) -> bool {
      network->pokemon_net.read_parameters(file);
      network->active_net.read_parameters(file);
      network->moves_net.read_parameters(file);
      network->main_net_float()->read_parameters(file);
      this->data = network;
      return true;
    };
    Header header{};
    static_assert(sizeof(header) == 8);
    file.read(reinterpret_cast<char *>(&header), 8);
    using NN::Activation;
    const auto activation = static_cast<Activation>(header.bytes[0] + 1);
    if (activation == Activation::clamp) {
      auto network = std::make_shared<NN::Battle::NetworkClamped>();
      return read_params(network);
    } else if (activation == Activation::relu) {
      auto network = std::make_shared<NN::Battle::Network>();
      return read_params(network);
    } else {
      throw std::runtime_error{"Agent: could not parse header at: " + path};
      return false;
    }
  }

  bool quantize() {
    auto network = get();
    if (auto net = std::dynamic_pointer_cast<NN::Battle::Network>(network)) {
      throw std::runtime_error{
          "Attempting to quantize a network with non clamped activations."};
      return false;
    } else if (auto net = std::dynamic_pointer_cast<NN::Battle::NetworkClamped>(
                   network)) {
      const auto [in, h, v, p] = network->shape();
      auto *m = network->main_net_float();
      auto q = NN::Battle::visit_quantized_network(in, h, v, p, [m](auto &net) {
        net.main_net.try_copy_parameters(*m);
      });
      q->pokemon_net = network->pokemon_net;
      q->active_net = network->active_net;
      q->moves_net = network->moves_net;
      if (!q) {
        throw std::runtime_error{"Network with that shape is not quantizable."};
        return false;
      }
      this->data = q;
      return true;
    } else if (network->main_net_float() == nullptr) {
      return true;
    } else {
      assert(false);
    }
    return false;
  }
  bool is_quantized() const { return false; }
};

class MonteCarlo : public Eval {
public:
  MonteCarlo() : Eval{std::in_place_type<MCTS::MonteCarlo>} {}
  auto &get() { return std::get<::MCTS::MonteCarlo>(this->data); }
  const auto &get() const { return std::get<::MCTS::MonteCarlo>(this->data); }
  bool &forbid_switches() { return get().forbid_switches; }
  const bool &forbid_switches() const { return get().forbid_switches; }
  bool &forbid_status() { return get().forbid_status; }
  const bool &forbid_status() const { return get().forbid_status; }
};

class PokeEngine : public Eval {
public:
  PokeEngine() : Eval{std::in_place_type<::PokeEngine::Eval>} {}
  auto &get() { return std::get<::PokeEngine::Eval>(this->data); }
  const auto &get() const { return std::get<::PokeEngine::Eval>(this->data); }
};

struct Heap {
  template <typename... T>
  using NodeVariantT = std::variant<std::monostate, MCTS::Node<T>...>;

  template <typename... T>
  using TableVariantT = std::variant<std::monostate, MCTS::Table<T>...>;

  using NodeVariant =
      NodeVariantT<Exp3::JointBandit, PExp3::JointBandit, UCB::JointBandit,
                   PUCB::JointBandit, UCB1::JointBandit>;
  using TableVariant =
      TableVariantT<Exp3::JointBandit, PExp3::JointBandit, UCB::JointBandit,
                    PUCB::JointBandit, UCB1::JointBandit>;
  using Variant = std::variant<NodeVariant, TableVariant>;
  Variant data;

  template <class T, class... Args>
  Heap(std::in_place_type_t<T>, Args &&...args)
      : data(std::in_place_type<T>, std::forward<Args>(args)...) {}

  void reset() {
    std::visit([](auto &v) { v = std::monostate{}; }, data);
  }
};

class Node : public Heap {
public:
  Node() : Heap{std::in_place_type<Heap::NodeVariant>} {}

  bool update(uint8_t i, uint8_t j, const MCTS::Obs &obs) {
    const auto lambda = [&](auto &node) {
      using T = std::remove_cvref_t<decltype(node)>;
      if constexpr (std::is_same_v<T, std::monostate>) {
        return false;
      } else {
        if (auto child = node.children.find({i, j, obs});
            child == node.children.end()) {
          node = {};
          return false;
        } else {
          std::swap(node, child->second);
          return true;
        }
      }
    };
    return std::visit(lambda, std::get<Heap::NodeVariant>(this->data));
  }
};

class Table : public Heap {
public:
  Table() : Heap{std::in_place_type<Heap::TableVariant>} {}
};

struct BanditParams {
  template <typename... T> using VariantT = std::variant<typename T::Params...>;
  using Variant = VariantT<Exp3::Bandit, PExp3::Bandit, UCB::Bandit,
                           PUCB::Bandit, UCB1::Bandit>;
  Variant data;
  BanditParams() = default;
  template <class T, class... Args>
  BanditParams(std::in_place_type_t<T>, Args &&...args)
      : data(std::in_place_type<T>, std::forward<Args>(args)...) {}
  virtual ~BanditParams() = default;
};

class Exp3 : public BanditParams {
public:
  Exp3(float lr, float exploration)
      : BanditParams{std::in_place_type<::Exp3::Bandit::Params>, lr,
                     exploration} {}
};
class PExp3 : public BanditParams {
public:
  PExp3(float lr, float exploration)
      : BanditParams{std::in_place_type<::PExp3::Bandit::Params>, lr,
                     exploration} {}
};
class UCB : public BanditParams {
public:
  UCB(float c) : BanditParams{std::in_place_type<::UCB::Bandit::Params>, c} {}
};
class PUCB : public BanditParams {
public:
  PUCB(float c) : BanditParams{std::in_place_type<::PUCB::Bandit::Params>, c} {}
};
class UCB1 : public BanditParams {
public:
  UCB1(float c) : BanditParams{std::in_place_type<::UCB1::Bandit::Params>, c} {}
};
struct MatrixUCB : public BanditParams {
  uint32_t delay;
  uint32_t interval;
  uint32_t minimum;
  float c;
  MatrixUCB(const BanditParams &params, uint32_t delay, uint32_t interval,
            uint32_t minimum, float c)
      : delay{delay}, interval{interval}, minimum{minimum}, c{c} {
    this->data = params.data;
  }
};

// Budget

struct Budget {
  using Variant = std::variant<size_t, std::chrono::milliseconds, bool *>;
  Variant data;
  template <class T, class... Args>
  Budget(std::in_place_type_t<T>, Args &&...args)
      : data(std::in_place_type<T>, std::forward<Args>(args)...) {}
};

class Flag : public Budget {
  Flag() : Budget{std::in_place_type<bool *>} {}
  bool &value() { return *std::get<bool *>(this->data); }
};

class Iterations : public Budget {
public:
  Iterations(size_t i) : Budget{std::in_place_type<size_t>, i} {}
};

class SideCache {
public:
  template <typename T> using Cache = NN::Battle::SideCache<T>;
  template <typename... T> using VariantT = std::variant<Cache<T>...>;
  using Variant = VariantT<float, uint8_t>;
  Variant data;

  bool is_quantized() const { return std::get_if<Cache<uint8_t>>(&data); }
  void quantize(std::shared_ptr<NN::Battle::NetworkBase> network) {
    if (!this->is_quantized()) {
      const auto &net = network.get();
      auto quantized = NN::Battle::quantize_cache(
          std::get<Cache<float>>(data), net->pokemon_out_dim(),
          net->active_out_dim(), net->moves_out_dim());
      data = std::move(quantized);
    }
  }
  void precompute(std::shared_ptr<NN::Battle::NetworkBase> network,
                  const ::PKMN::Side &side, uint8_t index) {
    const auto precompute = [&](auto &net) {
      using Network = typename std::remove_cvref_t<decltype(net)>;
      constexpr auto activation = Network::act;
      if (std::holds_alternative<NN::Battle::SideCache<float>>(this->data)) {
        auto &c = std::get<NN::Battle::SideCache<float>>(this->data);
        c.precompute<activation>(net, side, index);
      } else {
        auto &c = std::get<NN::Battle::SideCache<uint8_t>>(this->data);
        c.precompute<activation>(net, side, index);
      }
    };
    NN::Battle::visit_network(network, precompute);
  }
};

namespace Parse {

Eval eval(const std::string &s, bool quantize) {
  if (s == "mc" || s == "montecarlo") {
    return MonteCarlo{};
  }
  if (s == "fp") {
    return PokeEngine{};
  }
  Network network{};
  if (!network.read_parameters(s)) {
    throw std::runtime_error{"Parse::eval: could not read parameters at: " + s};
  }
  if (quantize) {
    network.quantize();
  }
  return network;
}

BanditParams bandit(const std::string &s) {
  const auto bandit_split = ::Parse::split(s, '-');
  if (bandit_split.size() < 2) {
    throw std::runtime_error{"Could not parse bandit string: " + s};
  }

  const auto &name = bandit_split[0];
  const float c_or_lr = std::stof(bandit_split[1]);

  if (name == "ucb") {
    return UCB{c_or_lr};
  } else if (name == "ucb1") {
    return UCB1{c_or_lr};
  } else if (name == "pucb") {
    return PUCB{c_or_lr};
  }
  float exploration = .05f;
  if (bandit_split.size() >= 3) {
    exploration = std::stof(bandit_split[2]);
  }
  if (name == "exp3") {
    return Exp3{c_or_lr, exploration};
  } else if (name == "pexp3") {
    return PExp3{c_or_lr, exploration};
  } else {
    throw std::runtime_error{"Could not parse bandit string: " + name};
  }
}

Heap heap(bool use_table) {
  if (use_table) {
    return Table{};
  } else {
    return Node{};
  }
}

Budget budget(const std::string &s) {
  const auto pos = s.find_first_not_of("0123456789");
  const size_t number = std::stoull(s.substr(0, pos));
  const std::string unit = (pos == std::string::npos) ? "" : s.substr(pos);
  if (unit.empty()) {
    return Iterations{number};
  } else if (unit == "ms" || unit == "millisec" || unit == "milliseconds") {
    return Budget{std::in_place_type<std::chrono::milliseconds>,
                  std::chrono::milliseconds{number}};
  } else if (unit == "s" || unit == "sec" || unit == "seconds") {
    return Budget{std::in_place_type<std::chrono::milliseconds>,
                  std::chrono::milliseconds{number * 1000}};
  } else {
    throw std::runtime_error{"Invalid search duration specification: " + s};
  }
}
} // namespace Parse

} // namespace Py::Search