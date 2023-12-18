#include "core/algorithm/strategy_alphazero.h"
#include "core/examples/game_quoridor.h"
#include "core/util/processor_count.h"

int main(int argc, const char** argv) {
  torch::set_num_threads(2);
  alphazero::Algorithm<Quoridor::GameState, 0> algorithm(
      "testdata/quoridor_baseline.pt");
  Quoridor::GameState game;
  algorithm.init(Quoridor::CANONICAL_SHAPE, Quoridor::NUM_PLAYERS + 1,
                 Quoridor::NUM_ACTIONS);
  auto context = algorithm.compute(game);

  return 0;
}
