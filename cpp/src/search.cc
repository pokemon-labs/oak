#include <util/search.h>

#include <search/mcts.h>
#include <util/strings.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace RuntimeSearch {

bool Heap::empty() const noexcept {
  return std::holds_alternative<std::monostate>(data);
}

template <typename T>
auto both(Heap &heap) -> std::pair<MCTS::Node<T> *, MCTS::Table<T> *> {
  return {std::get_if<MCTS::Node<T>>(&heap.data),
          std::get_if<MCTS::Table<T>>(&heap.data)};
}

bool Heap::update(uint8_t i, uint8_t j, const MCTS::Obs &obs) {
  const auto lambda = [&](auto &node) {
    using T = std::remove_cvref_t<decltype(node)>;
    if constexpr (std::is_same_v<T, std::monostate>) {
      return false;
    } else {
      if constexpr (TypeTraits::is_node<T>) {
        if (!node.stats.is_init()) {
          return false;
        }
        auto child = node.children.find({i, j, obs});
        if (child == node.children.end()) {
          node = {};
          return false;
        } else {
          std::swap(node, child->second);
          return true;
        }
      } else {
        static_assert(TypeTraits::is_table<T>);
        return true;
      }
    }
  };
  return std::visit(lambda, data);
}

std::string Heap::type() const noexcept {
  return std::visit(
      [](const auto &value) { return std::string(typeid(value).name()); },
      data);
}

// Agent

void Agent::initialize_network(const pkmn_gen1_battle &b) {
  struct FdGuard {
    int fd;
    ~FdGuard() {
      flock(fd, LOCK_UN);
      close(fd);
    }
  };

  const auto try_open_file = [this]() {
    constexpr auto tries = 3;
    for (auto i = 0; i < tries; ++i) {
      int fd = open(eval.c_str(), O_RDONLY);
      if (fd == -1) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      if (flock(fd, LOCK_SH) == -1) {
        close(fd);
        throw std::runtime_error{"Agent: could not lock file: " + eval};
      }
      std::string fd_path = "/proc/self/fd/" + std::to_string(fd);
      std::fstream file{fd_path};
      if (file) {
        return std::make_pair(std::move(file), fd);
      }
      flock(fd, LOCK_UN);
      close(fd);
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    throw std::runtime_error{"Agent: could not open file at: " + eval};
  };

  auto [file, fd] = try_open_file();
  struct Header {
    uint8_t bytes[8];
  };
  FdGuard guard{fd};

  const auto read_parameters_and_maybe_quantize = [&](auto &network) {
    if (!network->read_parameters(file)) {
      throw std::runtime_error{"Agent: could not read parameters at: " + eval};
    }
    network->fill_cache(b);
    if (discrete) {
      const auto [id, hd, vd, pd] = network->main_net.shape();
      auto q_network_ptr = NN::Battle::visit_quantized_network(
          id, hd, vd, pd, [&network](auto &net) {
            // convert to quantized
            net.active_net = network->active_net;
            net.pokemon_net = network->pokemon_net;
            net.pokemon_out_dim = network->pokemon_out_dim;
            net.active_out_dim = network->active_out_dim;
            net.side_embedding_dim = network->side_embedding_dim;
            net.battle_embedding.resize(network->battle_embedding.size());
            net.battle_cache = network->battle_cache;
            net.main_net.try_copy_parameters(network->main_net);
          });
      network_ptr = std::move(q_network_ptr);
    } else {
      network_ptr = std::move(network);
    }
    assert(network_ptr);
  };

  Header header{};
  static_assert(sizeof(header) == 8);
  file.read(reinterpret_cast<char *>(&header), 8);
  using NN::Activation;
  const auto activation = static_cast<Activation>(header.bytes[0] + 1);
  if (activation == Activation::clamp) {
    auto network = std::make_unique<NN::Battle::NetworkClamped>();
    read_parameters_and_maybe_quantize(network);
    return;
  }
  if (discrete) {
    throw std::runtime_error{"Agent: .discrete was specified but the parsed "
                             "header does not encode clamped activations."};
  }
  if (activation == Activation::relu) {
    auto network = std::make_unique<NN::Battle::Network>();
    read_parameters_and_maybe_quantize(network);
    return;
  } else {
    throw std::runtime_error{"Agent: could not parse header at: " + eval};
  }
}

MCTS::Output run(mt19937 &device, const MCTS::Input &input, Heap &heap_variant,
                 Agent &agent, MCTS::Output output, bool *const flag) {

  const auto parse_eval_and_search = [&](const auto dur, const auto &params,
                                         auto &heap) {
    MCTS::Search s{};
    if (agent.is_monte_carlo()) {
      MCTS::MonteCarlo model{};
      return s.run(device, dur, params, heap, model, input, output);
    } else if (agent.is_foul_play()) {
      PokeEngine::Eval model{};
      return s.run(device, dur, params, heap, model, input, output);
    } else {
      if (!agent.network_ptr) {
        agent.initialize_network(input.battle);
      }
      if (auto network =
              dynamic_cast<NN::Battle::Network *>(agent.network_ptr.get())) {
        return s.run(device, dur, params, heap, *network, input, output);
      } else if (auto network = dynamic_cast<NN::Battle::NetworkClamped *>(
                     agent.network_ptr.get())) {
        return s.run(device, dur, params, heap, *network, input, output);
      }
      {
        const auto [id, hd, vd, pd] = agent.network_ptr->shape();
        auto q_network_ptr = NN::Battle::visit_quantized_network(
            id, hd, vd, pd,
            [&](auto &net) {
              output = s.run(device, dur, params, heap, net, input, output);
            },
            std::move(agent.network_ptr));
        if (q_network_ptr) {
          agent.network_ptr = std::move(q_network_ptr);
        }
        return output;
      }
    }
  };

  const auto parse_heap_and_search = [&](const auto dur, const auto &params,
                                         const auto &both) {
    const auto [node_ptr, table_ptr] = both;
    using Node = std::remove_cvref_t<decltype(*node_ptr)>;
    using Table = std::remove_cvref_t<decltype(*table_ptr)>;
    auto &heap = heap_variant.data;
    if (agent.table) {
      if (heap_variant.empty()) {
        heap = Table{};
        return parse_eval_and_search(dur, params, std::get<Table>(heap));
      } else if (!table_ptr) {
        throw std::runtime_error{"RuntimeSearch: Bad Heap access. Expecting " +
                                 std::string{typeid(Table).name()}};
      }
      return parse_eval_and_search(dur, params, std::get<Table>(heap));
    } else {
      if (heap_variant.empty()) {
        heap = Node{};
        return parse_eval_and_search(dur, params, std::get<Node>(heap));
      } else if (!node_ptr) {
        throw std::runtime_error{"RuntimeSearch: Bad Heap access. Expecting " +
                                 std::string{typeid(Node).name()}};
      }
      return parse_eval_and_search(dur, params, std::get<Node>(heap));
    }
  };

  const auto parse_matrix_ucb_and_search = [&](auto dur, auto &bandit_params,
                                               const auto &both) {
    const auto &matrix_ucb = agent.matrix_ucb;
    if (!matrix_ucb.empty()) {
      const auto matrix_ucb_split = Parse::split(agent.matrix_ucb, '-');
      if (matrix_ucb_split.size() != 4) {
        throw std::runtime_error{"Could not parse MatrixUCB name: " +
                                 agent.matrix_ucb};
      }
      MCTS::MatrixUCBParams<std::remove_cvref_t<decltype(bandit_params)>>
          matrix_ucb_params{bandit_params};
      matrix_ucb_params.delay = std::stoull(matrix_ucb_split[0]);
      matrix_ucb_params.interval = std::stoull(matrix_ucb_split[1]);
      matrix_ucb_params.minimum = std::stoull(matrix_ucb_split[2]);
      matrix_ucb_params.c = std::stof(matrix_ucb_split[3]);
      return parse_heap_and_search(dur, matrix_ucb_params, both);
    } else {
      return parse_heap_and_search(dur, bandit_params, both);
    }
  };

  const auto parse_bandit_and_search = [&](auto dur) {
    const auto bandit_split = Parse::split(agent.bandit, '-');
    if (bandit_split.size() < 2) {
      throw std::runtime_error("Could not parse bandit string: " +
                               agent.bandit);
    }

    const auto check_for_priors = [&agent]() {
      if (agent.is_monte_carlo() || agent.is_foul_play()) {
        throw std::runtime_error{"Contextual bandit specified with eval that "
                                 "does not produce policy priors."};
      }
    };

    const auto &name = bandit_split[0];
    const float f1 = std::stof(bandit_split[1]);
    if (name == "ucb") {
      UCB::Bandit::Params params{.c = f1};
      return parse_matrix_ucb_and_search(dur, params,
                                         both<UCB::JointBandit>(heap_variant));
    } else if (name == "ucb1") {
      UCB1::Bandit::Params params{.c = f1};
      return parse_matrix_ucb_and_search(dur, params,
                                         both<UCB1::JointBandit>(heap_variant));
    } else if (name == "pucb") {
      check_for_priors();
      PUCB::Bandit::Params params{.c = f1};
      return parse_matrix_ucb_and_search(dur, params,
                                         both<PUCB::JointBandit>(heap_variant));
    }

    float alpha = .05;
    if (bandit_split.size() >= 3) {
      alpha = std::stof(bandit_split[2]);
    }
    if (name == "exp3") {
      Exp3::Bandit::Params params{.gamma = f1,
                                  .one_minus_gamma = (1 - f1),
                                  .alpha = alpha,
                                  .one_minus_alpha = (1 - alpha)};
      return parse_matrix_ucb_and_search(dur, params,
                                         both<Exp3::JointBandit>(heap_variant));
    } else if (name == "pexp3") {
      check_for_priors();
      PExp3::Bandit::Params params{.gamma = f1,
                                   .one_minus_gamma = (1 - f1),
                                   .alpha = alpha,
                                   .one_minus_alpha = (1 - alpha)};
      return parse_matrix_ucb_and_search(
          dur, params, both<PExp3::JointBandit>(heap_variant));
    } else {
      throw std::runtime_error("Could not parse bandit string: " + name);
    }
  };

  const auto parse_budget_and_search = [&]() {
    if (flag != nullptr) {
      return parse_bandit_and_search(flag);
    }
    const auto pos = agent.budget.find_first_not_of("0123456789");
    size_t number = std::stoll(agent.budget.substr(0, pos));
    std::string unit =
        (pos == std::string::npos) ? "" : agent.budget.substr(pos);
    if (unit.empty()) {
      return parse_bandit_and_search(number);
    } else if (unit == "ms" || unit == "millisec" || unit == "milliseconds") {
      return parse_bandit_and_search(std::chrono::milliseconds{number});
    } else if (unit == "s" || unit == "sec" || unit == "seconds") {
      return parse_bandit_and_search(std::chrono::seconds{number});
    } else {
      throw std::runtime_error("Invalid search duration specification: " +
                               agent.budget);
    }
  };

  return parse_budget_and_search();
}

} // namespace RuntimeSearch