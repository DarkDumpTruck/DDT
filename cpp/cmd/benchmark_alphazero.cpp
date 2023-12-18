#include "core/algorithm/strategy_alphazero.h"
#include "core/examples/game_quoridor.h"
#include "core/util/processor_count.h"

constexpr int NumParallel = 4;
constexpr int NumIterations = 10000;

int main() {
  torch::set_num_threads(ProcessorCnt);
  alphazero::Algorithm<Quoridor::GameState, NumParallel> algorithm(
      "testdata/quoridor_baseline.pt");
  Quoridor::GameState game;
  algorithm.init(Quoridor::CANONICAL_SHAPE, Quoridor::NUM_PLAYERS + 1,
                 Quoridor::NUM_ACTIONS);
  auto context = algorithm.compute(game);

  auto start = std::chrono::high_resolution_clock::now();
  context->step(/*iterations=*/NumIterations);
  context->best_move();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "\nTime: " << duration << "ms\nIteration: " << NumIterations
            << std::endl;
  return 0;
}
