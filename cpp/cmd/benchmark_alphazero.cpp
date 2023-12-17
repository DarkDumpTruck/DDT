#include "core/algorithm/strategy_alphazero.h"
#include "core/examples/game_quoridor.h"
#include "core/util/processor_count.h"

constexpr int NumParallel = 4;

int main(int argc, const char** argv) {
  int NumIterations = 20000;
  if (argc == 3 && strcmp(argv[1], "-i") == 0) {
    NumIterations = std::atoi(argv[2]);
  }
  torch::set_num_threads(ProcessorCnt);
  alphazero::Algorithm<Quoridor::GameState, NumParallel - 1> algorithm(
      "testdata/quoridor_baseline.pt");
  Quoridor::GameState game;
  algorithm.init(Quoridor::CANONICAL_SHAPE, Quoridor::NUM_PLAYERS + 1,
                 Quoridor::NUM_ACTIONS);
  auto context = algorithm.compute(game);

  auto start = std::chrono::high_resolution_clock::now();
  context->step(/*iterations=*/NumIterations);
  auto best_move = context->best_move();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "\nTime: " << duration << "ms\nIteration: " << NumIterations << "\nBest move: " << Quoridor::action_to_string(best_move)
            << std::endl;
  return 0;
}
