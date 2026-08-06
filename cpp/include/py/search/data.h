#pragma once

#include <nn/battle/cache.h>
#include <nn/battle/network.h>
#include <search/bandit/exp3.h>
#include <search/bandit/pexp3.h>
#include <search/bandit/pucb.h>
#include <search/bandit/ucb.h>
#include <search/bandit/ucb1.h>
#include <search/mcts.h>
#include <util/file-lock.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <variant>

namespace Py::Search {

struct Eval {
  using Variant = std::variant<MCTS::MonteCarlo, PokeEngine::Eval,
                               std::shared_ptr<NN::Battle::NetworkBase>>;
  Variant data;
  template <class T, class... Args>
  Eval(std::in_place_type_t<T>, Args &&...args)
      : data(std::in_place_type<T>, std::forward<Args>(args)...) {}
};

class Network : public Eval {
public:
  using NetworkPtr = std::shared_ptr<NN::Battle::NetworkBase>;
  Network()
      : Eval(std::in_place_type<NetworkPtr>,
             std::make_shared<NN::Battle::Network>()) {}
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

  bool quantize() { return false; }
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
  Exp3(float lr, float exploration)
      : BanditParams{std::in_place_type<::Exp3::Bandit::Params>, lr,
                     exploration} {}
};
class PExp3 : public BanditParams {
  PExp3(float lr, float exploration)
      : BanditParams{std::in_place_type<::PExp3::Bandit::Params>, lr,
                     exploration} {}
};
class UCB : public BanditParams {
public:
  UCB(float c) : BanditParams{std::in_place_type<::UCB::Bandit::Params>, c} {}
};
class PUCB : public BanditParams {
  PUCB(float c) : BanditParams{std::in_place_type<::PUCB::Bandit::Params>, c} {}
};
class UCB1 : public BanditParams {
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
  void quantize(const Network &network) {
    if (!this->is_quantized()) {
      const auto &net = network.get();
      auto quantized = NN::Battle::quantize_cache(
          std::get<Cache<float>>(data), net->pokemon_out_dim(),
          net->active_out_dim(), net->moves_out_dim());
      data = std::move(quantized);
    }
  }
};

} // namespace Py::Search