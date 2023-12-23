// #define ALPHAZERO_SHOW_ACTION_CNT 6

#include "core/algorithm/strategy_alphazero.h"
#include "core/examples/game_quoridor.h"
#include "core/util/processor_count.h"

constexpr bool DEBUG_SHOW_ACTIONS_PER_TURN = false;
constexpr bool DEBUG_SHOW_GAMEBOARD = false;

constexpr int TORCH_NUM_THREADS = 2;
constexpr int TORCH_NUM_INTEROP_THREADS = 2;

constexpr int ALPHAZERO_NUM_PLAYOUT = 400;
constexpr int ALPHAZERO_NUM_PLAYOUT_CAP = 200;
constexpr int ALPHAZERO_CAP_TURN = 10;
constexpr int ALPHAZERO_MAX_TURN = 100;
constexpr float ALPHAZERO_TEMPERATURE_START = 1.0f;
constexpr float ALPHAZERO_TEMPERATURE_END = 0.1f;
constexpr float ALPHAZERO_TEMPERATURE_LAMBDA = -0.07f;

using Game = Quoridor::GameState;
using Algorithm = alphazero::Algorithm<Game, 0>;

void init_torch() {
  torch::set_num_threads(TORCH_NUM_THREADS);
  torch::set_num_interop_threads(TORCH_NUM_INTEROP_THREADS);
}

int main(int argc, const char** argv) {
  init_torch();
  Algorithm algorithm("testdata/quoridor_baseline.pt");
  algorithm.init(Quoridor::CANONICAL_SHAPE, Quoridor::NUM_PLAYERS + 1,
                  Quoridor::NUM_ACTIONS);

  while (true) {
    Game game;
    float temperature = ALPHAZERO_TEMPERATURE_START;
    int turn;
    std::vector<std::unique_ptr<Algorithm::Context>> contexts;

    for (turn = 0; turn < ALPHAZERO_MAX_TURN && !game.End(); turn++) {
      auto start_ts = high_resolution_clock::now();
      bool capped = turn >= ALPHAZERO_CAP_TURN;

      temperature = std::exp(ALPHAZERO_TEMPERATURE_LAMBDA * turn) *
                        (temperature - ALPHAZERO_TEMPERATURE_END) +
                    ALPHAZERO_TEMPERATURE_END;

      auto context = algorithm.compute(game);
      context->step(capped ? ALPHAZERO_NUM_PLAYOUT_CAP : ALPHAZERO_NUM_PLAYOUT,
                    !capped);
      auto action = context->select_move(temperature);
      game.Move(action);

      contexts.emplace_back(std::move(context));

      auto duration =
          duration_cast<milliseconds>(high_resolution_clock::now() - start_ts)
              .count();
      if constexpr (DEBUG_SHOW_ACTIONS_PER_TURN) {
        std::cout << "Turn " << turn << ", cost=" << duration << "ms"
                  << ", action=" << Quoridor::action_to_string(action)
                  << std::endl;
      } else {
        std::cout.put('.').flush();
      }
      if constexpr (DEBUG_SHOW_GAMEBOARD) {
        std::cout << game.ToString() << std::endl;
      }
    }

    if (turn == ALPHAZERO_MAX_TURN) {
      std::cout << "\nGame is too long, skipped." << std::endl;
      std::cout << game.ToString() << std::endl;
      continue;
    } else {
      std::cout << "\nGame finished in " << turn << " turns. Winner is " << (game.Winner() ? 'X' : 'O') << std::endl;
      std::cout << game.ToString() << std::endl;
    }

    int n = contexts.size();
    at::Tensor canonical = torch::empty(
        {n, Quoridor::CANONICAL_SHAPE[0], Quoridor::CANONICAL_SHAPE[1],
         Quoridor::CANONICAL_SHAPE[2]},
        torch::kFloat);
    at::Tensor policy = torch::empty({n, Quoridor::NUM_ACTIONS}, torch::kFloat);
    at::Tensor values = torch::zeros({n, 3}, torch::kFloat);
    for (int i = 0; i < n; i++) {
      auto& context = contexts[i];
      context->game->Canonicalize(canonical[i].data_ptr<float>());
      context->mcts.set_probs(policy[i].data_ptr<float>(), 1.0f);
      values[i][game.Winner()] = 1.0f;
    }

    for (int saveIndex = 0;; saveIndex++) {
      auto c_path = std::format("output/c_{:04d}.pt", saveIndex);
      auto p_path = std::format("output/p_{:04d}.pt", saveIndex);
      auto v_path = std::format("output/v_{:04d}.pt", saveIndex);
      if (std::filesystem::exists(c_path) || std::filesystem::exists(p_path) ||
          std::filesystem::exists(v_path)) {
        continue;
      }
      if (!std::filesystem::exists("output")) {
        std::filesystem::create_directory("output");
      }
      torch::save(canonical, c_path);
      torch::save(policy, p_path);
      torch::save(values, v_path);
      break;
    }
  }

  return 0;
}
