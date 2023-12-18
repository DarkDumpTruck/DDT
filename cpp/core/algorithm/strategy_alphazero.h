#pragma once

#pragma warning(push, 0)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/script.h>
#include <torch/torch.h>
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "core/util/common.h"

namespace alphazero {

using ActionType = uint8_t;

/* NOISE_ALPHA_RATIO needs to be changed across different games.
   Related papers:
   [1] Domain Knowledge in Exploration Noise in AlphaZero.
       Eric W., George D., Aaron T., Abtin M.
   [2] https://medium.com/oracledevs/lessons-from-alpha-zero-part
       -6-hyperparameter-tuning-b1cfcbe4ca9a
*/
const float NOISE_ALPHA_RATIO = 10.83f;

const float CPUCT = 1.25;
const float FPU_REDUCTION = 0.25;

struct ValueType {
  // v should be the winrate/score for player 0.
  // in this project, we only consider two-player zero sum games. value(0) +
  // value(1) = 1.
  float v;
  ValueType() = default;
  explicit ValueType(float v_) : v(v_) {}
  explicit ValueType(int index, float v_) { set(index, v_); }
  explicit ValueType(float v0, float v1) { v = v0 / (v0 + v1); }
  float operator()(int index) const { return get(index); }
  float get(int index) const {
    assert(index == 0 || index == 1);
    return index == 0 ? v : 1 - v;
  }
  void set(int index, float v_) {
    assert(index == 0 || index == 1);
    v = index == 0 ? v_ : 1 - v_;
  }
};

struct Node {
  Node() = default;
  explicit Node(ActionType m) : move(m) {}

  float q = 0;
  float v = 0;
  float policy = 0;
  ActionType move = 0;
  int n = 0;
  bool player = 0;
  bool ended = false;
  ValueType value;
  std::vector<Node> children{};

  void add_children(const std::vector<ActionType>& valids) noexcept {
    for (ActionType w = 0; w < valids.size(); ++w) {
      if (valids[w]) {
        children.emplace_back(w);
      }
    }
    static auto rd = std::default_random_engine(0);
    std::shuffle(children.begin(), children.end(), rd);
  }
  size_t size() const noexcept { return children.size(); }
  void update_policy(const std::vector<float>& pi) noexcept {
    for (auto& c : children) {
      c.policy = pi[c.move];
    }
  }
  float uct(float sqrt_parent_n, float cpuct, float fpu_value) const noexcept {
    return (n == 0 ? fpu_value : q) + cpuct * policy * sqrt_parent_n / (n + 1);
  }
  Node* best_child(float cpuct, float fpu_reduction) noexcept {
    auto seen_policy = 0.0f;
    for (const auto& c : children) {
      if (c.n > 0) {
        seen_policy += c.policy;
      }
    }
    auto fpu_value = v - fpu_reduction * std::sqrt(seen_policy);
    auto sqrt_n = std::sqrt((float)n);
    auto best_i = 0;
    auto best_uct = children.at(0).uct(sqrt_n, cpuct, fpu_value);
    for (size_t i = 1; i < children.size(); ++i) {
      auto uct = children.at(i).uct(sqrt_n, cpuct, fpu_value);
      if (uct > best_uct) {
        best_uct = uct;
        best_i = i;
      }
    }
    return &children.at(best_i);
  }
};

template <class GameState>
class MCTS {
 public:
  MCTS(float cpuct, int num_moves, float epsilon = 0,
       float root_policy_temp = 1.4, float fpu_reduction = 0)
      : cpuct_(cpuct),
        num_moves_(num_moves),
        current_(&root_),
        epsilon_(epsilon),
        root_policy_temp_(root_policy_temp),
        fpu_reduction_(fpu_reduction) {}

  void update_root(const GameState& gs, ActionType move) {
    depth_ = 0;
    if (root_.children.empty()) {
      root_.add_children(gs.Valid_moves());
    }
    auto x = std::find_if(root_.children.begin(), root_.children.end(),
                          [move](const Node& n) { return n.move == move; });
    assert(x != root_.children.end());
    root_ = *x;
  }

  std::unique_ptr<GameState> find_leaf(const GameState& gs) {
    current_ = &root_;
    auto leaf = gs.Copy();

    while (current_->n > 0 && !current_->ended) {
      path_.push_back(current_);
      auto fpu_reduction = fpu_reduction_;
      // root fpu is half-ed.
      if (current_ == &root_) {
        fpu_reduction /= 2;
      }
      // fpu of failing node is half-ed.
      if (current_->n > 0 && current_->v < 0.2) {
        fpu_reduction /= 2;
      }
      current_ = current_->best_child(cpuct_, fpu_reduction);

#ifdef DEBUG_MCTS_PATH
      std::cout << GameState::action_to_string(current_->move, current->player)
                << " ";
#endif

      leaf->Move(current_->move);
    }

#ifdef DEBUG_MCTS_PATH
    if (leaf->End()) {
      std::cout << "Winner: " << (int)leaf->Winner();
    }
    std::cout << std::endl;
#endif

    if (current_->n == 0) {
      current_->player = leaf->Current_player();
      if ((bool)(current_->ended = leaf->End())) {
        current_->value = ValueType(leaf->Current_player(),
                                    leaf->Winner() == leaf->Current_player());
      }
      current_->add_children(leaf->Valid_moves());
    }
    return leaf;
  }

  void process_result(const std::vector<float>& pi, ValueType value,
                      bool root_noise_enabled = false) {
    process_result(&pi[0], pi.size(), value, root_noise_enabled);
  }

  void process_result(const float* pi, size_t sp, ValueType value,
                      bool root_noise_enabled = false) {
    if (current_->ended) {
      value = current_->value;
    } else {
      // Rescale pi based on valid moves.
      std::vector<float> scaled(sp, 0);
      float sum = 0;
      for (auto& c : current_->children) {
        sum += pi[c.move];
      }
      if (sum > 1e-7f) {
        for (auto& c : current_->children) {
          scaled[c.move] = pi[c.move] / sum;
        }
      }
      if (current_ == &root_) {
        sum = 0;
        for (auto& c : current_->children) {
          scaled[c.move] = std::pow(scaled[c.move], 1.0 / root_policy_temp_);
          sum += scaled[c.move];
        }
        if (sum > 1e-7f) {
          for (auto& c : current_->children) {
            scaled[c.move] = scaled[c.move] / sum;
          }
        }
        current_->update_policy(scaled);
        if (root_noise_enabled) {
          add_root_noise();
        }
      } else {
        current_->update_policy(scaled);
      }
    }

    while (!path_.empty()) {
      // TODO: update parent->values when all subtree have been visited.
      auto* parent = path_.back();
      path_.pop_back();
      auto v = value(parent->player);
      current_->q = (current_->q * current_->n + v) / (current_->n + 1);
      if (current_->n == 0) {
        current_->v = value(current_->player);
      }
      ++current_->n;
      current_ = parent;
    }
    ++depth_;
    ++root_.n;
  }

  void add_root_noise() {
    auto dist =
        std::gamma_distribution<float>{NOISE_ALPHA_RATIO / root_.size(), 1.0};
    static auto re = std::default_random_engine(0);
    std::vector<float> noise(num_moves_, 0);
    float sum = 0;
    for (auto& c : root_.children) {
      noise[c.move] = dist(re);
      sum += noise[c.move];
    }
    for (auto& c : root_.children) {
      c.policy = c.policy * (1 - epsilon_) + epsilon_ * noise[c.move] / sum;
    }
  }

  float root_value() const {
    float q = 0;
    for (const auto& c : root_.children) {
      if (c.n > 0 && c.q > q) {
        q = c.q;
      }
    }
    return q;
  }

  std::vector<Node>& root_children() noexcept { return root_.children; }

  std::vector<int> counts() const noexcept {
    std::vector<int> result(num_moves_, 0);
    for (const auto& c : root_.children) {
      result[c.move] = c.n;
    }
    return result;
  }

  std::map<int, int> counts_map() const noexcept {
    std::map<int, int> result;
    for (const auto& c : root_.children) {
      result[c.n] = c.move;
    }
    return result;
  }

  std::vector<float> probs(float temp) const noexcept {
    auto counts = this->counts();
    std::vector<float> probs(num_moves_, 0);

    if (temp < 1e-7f) {
      auto best_moves = std::vector<int>{0};
      auto best_count = counts[0];
      for (auto m = 1; m < num_moves_; ++m) {
        if (counts[m] > best_count) {
          best_count = counts[m];
          best_moves = {m};
        } else if (counts[m] == best_count) {
          best_moves.push_back(m);
        }
      }

      for (auto m : best_moves) {
        probs[m] = 1.0 / best_moves.size();
      }
      return probs;
    }

    float sum = 0;
    for (int i = 0; i < num_moves_; i++) {
      sum += counts[i];
    }
    for (int i = 0; i < num_moves_; i++) {
      probs[i] = counts[i] / sum;
    }
    sum = 0;
    for (int i = 0; i < num_moves_; i++) {
      sum += std::pow(probs[i], 1. / temp);
    }
    for (int i = 0; i < num_moves_; i++) {
      probs[i] = probs[i] / sum;
    }
    return probs;
  }

  int depth() const noexcept { return depth_; }

  static ActionType pick_move(const std::vector<float>& p) {
    std::uniform_real_distribution<float> dist{0.0F, 1.0F};
    static auto re = std::default_random_engine(0);
    auto choice = dist(re);
    auto sum = 0.0F;
    for (size_t m = 0; m < p.size(); ++m) {
      sum += p[m];
      if (sum > choice) {
        return m;
      }
    }
    // Due to floating point error we didn't pick a move.
    // Pick the last valid move.
    for (size_t m = p.size() - 1; m >= 0; --m) {
      if (p[m] > 0) {
        return m;
      }
    }
    assert(false);
    return 0;
  }

 private:
  float cpuct_;
  int num_moves_;

  int depth_ = 0;

 public:
  Node root_ = Node{};

 private:
  Node* current_;
  std::vector<Node*> path_{};
  float epsilon_;
  float root_policy_temp_;
  float fpu_reduction_;
};

template <class GameState, int SpecThreadCount>
class Algorithm {
 public:
  Algorithm(std::string model_path_) : device("cpu"), model_path(model_path_) {}

  struct Context {
    Context(std::unique_ptr<GameState> game_)
        : game(std::move(game_)),
          mcts(
              /*cpuct=*/CPUCT,
              /*num_moves=*/game->Num_actions(),
              /*epsilon=*/0,
              /*root_policy_temp=*/1.4,
              /*fpu_reduction=*/FPU_REDUCTION) {
      for (int i = 0; i < SpecThreadCount; ++i) {
        specs[i] = std::make_unique<MCTS<GameState>>(
            /*cpuct=*/CPUCT,
            /*num_moves=*/game->Num_actions(),
            /*epsilon=*/0,
            /*root_policy_temp=*/1.4,
            /*fpu_reduction=*/FPU_REDUCTION);
      }
    }

    template <size_t N>
    void evaluate(const std::array<std::unique_ptr<GameState>, N>& games,
                  at::Tensor& vs, at::Tensor& pis) {
      auto input = torch::zeros({(int)N, base->d1, base->d2, base->d3});
      for (int i = 0; i < (int)N; ++i) {
        float* buffer =
            input.data_ptr<float>() + base->d1 * base->d2 * base->d3 * i;
        games[i]->Canonicalize(buffer);
      }

      std::vector<torch::jit::IValue> inputs = {input.to(base->device)};
      auto outputs = base->model.forward(inputs).toTuple();
      vs = torch::exp(outputs->elements()[0].toTensor()).cpu();
      pis = torch::exp(outputs->elements()[1].toTensor()).cpu();
    }

    void evaluate(const GameState& game, ValueType& v, at::Tensor& pi) {
      auto input = torch::zeros({1, base->d1, base->d2, base->d3});
      game.Canonicalize(input.data_ptr<float>());

      std::vector<torch::jit::IValue> inputs = {input.to(base->device)};
      auto outputs = base->model.forward(inputs).toTuple();
      auto tensor_v = torch::exp(outputs->elements()[0].toTensor()).cpu();
      pi = torch::exp(outputs->elements()[1].toTensor()).cpu();

      float* current_tensor_v = tensor_v.data_ptr<float>();
      v.set(current_tensor_v[0], current_tensor_v[1]);
    }

    void step(int iterations) {
      std::unique_lock lock(base->model_mutex);

      // initalize spec trees with most p-value moves.
      if (SpecThreadCount > 0) {
        ValueType v;
        at::Tensor tensor_pi;
        mcts.find_leaf(*game);
        evaluate(*game, v, tensor_pi);
        const float* pi = tensor_pi.data_ptr<float>();

        auto& children = mcts.root_children();
        std::vector<int> idx(children.size());
        for (int i = 0; i < (int)idx.size(); i++) idx[i] = children[i].move;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return pi[a] > pi[b]; });
        children.erase(
            std::remove_if(
                children.begin(), children.end(),
                [&](const alphazero::Node& node) {
                  return std::find(idx.begin(), idx.begin() + SpecThreadCount,
                                   node.move) != idx.begin() + SpecThreadCount;
                }),
            children.end());
        mcts.process_result(pi, base->sp, v);
        for (int i = 0; i < SpecThreadCount; i++) {
          specs[i]->root_children().emplace_back(idx[i]);
          specs[i]->root_.player = game->Current_player();
          specs[i]->process_result(pi, base->sp, v);
        }
      }

      int iter = 0;
      for (; iter < iterations; iter++) {
        std::array<std::unique_ptr<GameState>, SpecThreadCount + 1> leaves;
        for (int i = 0; i < SpecThreadCount; i++) {
          leaves[i] = specs[i]->find_leaf(*game);
        }
        leaves[SpecThreadCount] = mcts.find_leaf(*game);

        {
          at::Tensor vs, pis;
          evaluate(leaves, vs, pis);

          for (int i = 0; i < SpecThreadCount; i++) {
            const float* pi = pis.data_ptr<float>() + base->sp * i;
            const float* v = vs.data_ptr<float>() + base->sv * i;
            specs[i]->process_result(pi, base->sp, ValueType(v[0], v[1]));
          }
          const float* pi = pis.data_ptr<float>() + base->sp * SpecThreadCount;
          const float* v = vs.data_ptr<float>() + base->sv * SpecThreadCount;
          mcts.process_result(pi, base->sp, ValueType(v[0], v[1]));
        }

#ifdef ALPHAZERO_SHOW_ACTION_CNT
        // TODO: maybe print per time
        if (iter % 64 == 0) {
          for (int i = 0; i < SpecThreadCount + 1; i++) printf("\33[F");
          for (int i = 0; i < SpecThreadCount; i++) {
            auto counts = specs[i]->counts_map();
            auto root_value = specs[i]->root_value();
            printf("\nAction: [%s], Winrate:  %.4f",
                   GameState::action_to_string(counts.begin()->second,
                                               game->Current_player())
                       .c_str(),
                   root_value);
          }
          int show_count = ALPHAZERO_SHOW_ACTION_CNT;
          auto counts = mcts.counts_map();
          printf("\nAction: ");
          for (auto iter = counts.rbegin(); show_count && iter != counts.rend();
               iter++, show_count--) {
            printf("[%s]: %d , ",
                   GameState::action_to_string(iter->second,
                                               game->Current_player())
                       .c_str(),
                   iter->first);
          }
          auto root_value = mcts.root_value();
          printf("Winrate: %.4f", root_value);
        }
#endif
      }
    }

    ActionType best_move() {
      float best_value = 0;
      ActionType best_action = 0;
      for (const auto& c : mcts.root_.children) {
        if (c.n > 0 && c.q > best_value) {
          best_value = c.q;
          best_action = c.move;
        }
      }
      for (auto& spec : specs) {
        for (const auto& c : spec->root_.children) {
          if (c.n > 0 && c.q > best_value) {
            best_value = c.q;
            best_action = c.move;
          }
        }
      }
      return best_action;
    }

    ActionType select_move() {
      // TODO: select move based on probability.
      return best_move();
    }

    at::Tensor get_policy(float temperature) {
      auto probs = mcts.probs(temperature);
      at::Tensor tensor = torch::from_blob(probs.data(), {probs.size()});
      return tensor;
    }

    at::Tensor get_canonicalized() {
      auto input = torch::zeros({1, base->d1, base->d2, base->d3});
      game->Canonicalize(input.data_ptr<float>());
      return input;
    }

    std::unique_ptr<GameState> game;
    Algorithm* base;
    MCTS<GameState> mcts;
    std::array<std::unique_ptr<MCTS<GameState>>, SpecThreadCount> specs;
  };

  std::unique_ptr<Context> compute(const GameState& game) {
    auto context = std::make_unique<Context>(game.Copy());
    context->base = this;
    return context;
  }

  void init(const std::array<int, 3>& dimentions, int size_v, int size_pi) {
    if (torch::cuda::is_available()) {
      std::cout << "Using CUDA." << std::endl;
      device = torch::Device("cuda:0");
      options = options.device(device);
    } else {
      std::cout << "Using CPU." << std::endl;
    }

    model = torch::jit::load(model_path, device);
    if (model.is_training()) {
      std::cout << "Warning: Model is in training mode. Calling eval()."
                << std::endl;
      model.eval();
    }

    d1 = dimentions[0];
    d2 = dimentions[1];
    d3 = dimentions[2];
    sv = size_v;
    sp = size_pi;

    // undocumented API that looks like it can be used to optimize the model
    // torch::jit::optimize_for_inference(model);

    // warm up the model
    std::cout << "Warming up." << std::endl;
    std::vector<torch::jit::IValue> inputs = {
        torch::ones({1, d1, d2, d3}, options)};
    auto _ = model.forward(inputs).toTuple();
    if (_->elements().size() > 0) {
      std::cout << "Warm up ok." << std::endl;
    }
  }

 protected:
  torch::Device device;
  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat);

  int d1, d2, d3;
  int sv, sp;

  std::mutex model_mutex;
  std::string model_path;
  torch::jit::script::Module model;
};

}  // namespace alphazero
