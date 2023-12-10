#include "core/algorithm/strategy_alphazero.h"
#include "core/examples/game_quoridor.h"
#include "gtest/gtest.h"

/* Tests Node::update_policy() and Node::best_child().
 * 
 * When a node is first created, calling best_child() should return 
 * the child with max p-value which is valid.
 */
TEST(AlphaZeroTest, MctsNodeSimpleTest) {
  Quoridor::GameState game;
  alphazero::Node root;
  auto valids = game.Valid_moves();
  root.add_children(valids);

  std::vector<float> pi(Quoridor::NUM_ACTIONS, 0.f);
  EXPECT_TRUE(valids[Quoridor::MOVE_UP]);
  pi[Quoridor::MOVE_UP] = 0.1f;
  EXPECT_TRUE(valids[Quoridor::MOVE_DOWN]);
  pi[Quoridor::MOVE_DOWN] = 0.2f;
  EXPECT_TRUE(valids[Quoridor::MOVE_RIGHT]);
  pi[Quoridor::MOVE_RIGHT] = 0.3f;
  EXPECT_FALSE(valids[Quoridor::MOVE_LEFT]);
  pi[Quoridor::MOVE_LEFT] = 0.4f;
  root.n = 1;
  root.update_policy(pi);

  for (auto& c : root.children) {
    if (c.move == Quoridor::MOVE_UP) {
      EXPECT_FLOAT_EQ(0.1f, c.policy);
    }
  }

  auto* best = root.best_child(2.0, 0);
  // best move should be the valid move with max pi.
  EXPECT_EQ(best->move, Quoridor::MOVE_RIGHT);
}

/* Test MCTS::find_leaf() and MCTS::process_result().
 * 
 * This testcase constructs a special situation: the network output
 * always ask both player to go right, and the value is always 0.5.
 * In this case, the MCTS should have three stages:
 * 1. Every player goes to the right until reach game end.
 *    The q-value should be 0.5 for every node in this stage.
 * 2. The winner will stick on the "going right" policy, while the
 *    loser will try other moves. The q-value for the winner should
 *    be larger than 0.5, while the loser should be smaller than 0.5.
 * 3. The loser's q-value will continue to decrease, until he finds 
 *    a way to beat the "going right" policy, after that he will win
 *    the game for a long time.
 */
TEST(AlphaZeroTest, MctsSimpleTest) {
  Quoridor::GameState game;
  alphazero::MCTS<Quoridor::GameState> mcts(
      /*cpuct=*/2, /*num_actions=*/game.Num_actions());
  while (mcts.depth() < Quoridor::WIDTH * 2 - 5) {
    auto leaf = mcts.find_leaf(game);
    alphazero::ValueType value(0, 0.5f);
    std::vector<float> pi(game.Num_actions(), 0);
    pi[Quoridor::MOVE_RIGHT] = 1;
    mcts.process_result(pi, value);
  }
  for (auto& c : mcts.root_.children) {
    if (c.move == Quoridor::MOVE_RIGHT) {
      std::cout << c.q << std::endl;
      EXPECT_FLOAT_EQ(c.q, 0.5f);

      for (auto& cc : c.children) {
        if (cc.move == Quoridor::MOVE_RIGHT) {
          std::cout << cc.q << std::endl;
          EXPECT_FLOAT_EQ(cc.q, 0.5f);
          break;
        }
      }
    } else {
      EXPECT_TRUE(c.n == 0);
    }
  }
  while (mcts.depth() < 64) {
    auto leaf = mcts.find_leaf(game);
    alphazero::ValueType value(0, 0.5f);
    std::vector<float> pi(game.Num_actions(), 0);
    pi[Quoridor::MOVE_RIGHT] = 1;
    mcts.process_result(pi, value);
  }
  for (auto& c : mcts.root_.children) {
    if (c.move == Quoridor::MOVE_RIGHT) {
      std::cout << c.q << std::endl;
      EXPECT_EQ(0 < c.q && c.q < 0.5f, Quoridor::WIDTH % 2 == 1);
      EXPECT_TRUE(c.n == 63);

      for (auto& cc : c.children) {
        if (cc.move == Quoridor::MOVE_RIGHT) {
          std::cout << cc.q << std::endl;
          EXPECT_EQ(0.5f < cc.q && cc.q < 1.f, Quoridor::WIDTH % 2 == 1);
          break;
        }
      }
    } else {
      EXPECT_TRUE(c.n == 0);
    }
  }
  EXPECT_EQ(alphazero::MCTS<Quoridor::GameState>::pick_move(mcts.probs(0)),
            Quoridor::MOVE_RIGHT);
}


/* Test MCTS::find_leaf() and MCTS::process_result().
 * 
 * This testcase constructs another situation: the network output
 * ask both player to go up or down with equal policy, with a value
 * of 0.7.
 * In this case, the MCTS should never reach the end of the game,
 * and the q-value should be 0.7 for every node.
 * 
 * TODO: check that the MCTS do not reach the end of the game.
 */
TEST(AlphaZeroTest, MctsSimpleTest2) {
  Quoridor::GameState game;
  alphazero::MCTS<Quoridor::GameState> mcts(
      /*cpuct=*/2, /*num_actions=*/game.Num_actions());
  while (mcts.depth() < 300) {
    auto leaf = mcts.find_leaf(game);
    alphazero::ValueType value(0, 0.7f);
    std::vector<float> pi(game.Num_actions(), 0);
    pi[Quoridor::MOVE_UP] = 1;
    pi[Quoridor::MOVE_DOWN] = 1;
    mcts.process_result(pi, value);
  }
  for (auto& c : mcts.root_.children) {
    if (c.move == Quoridor::MOVE_DOWN || c.move == Quoridor::MOVE_UP) {
      EXPECT_NEAR(c.q, 0.7f, 1e-5f);
      EXPECT_TRUE(c.n > 0);
    } else {
      EXPECT_TRUE(c.n == 0);
    }
  }
}

TEST(AlphaZeroTest, AlphazeroSimpleTest) {
  alphazero::Algorithm<Quoridor::GameState, 3> algorithm("testdata/quoridor_baseline.pt");
  Quoridor::GameState game;
  
  algorithm.init(Quoridor::CANONICAL_SHAPE, Quoridor::NUM_PLAYERS + 1, Quoridor::NUM_ACTIONS);
  auto context = algorithm.compute(game);
  context->step(/*iterations=*/1000);
  auto action = context->best_move();

  EXPECT_EQ(action, Quoridor::MOVE_RIGHT);
}

TEST(AlphaZeroTest, AlphazeroSimpleTest2) {
  alphazero::Algorithm<Quoridor::GameState, 3> algorithm("testdata/quoridor_baseline.pt");
  Quoridor::GameState game;
  game.Move(Quoridor::MOVE_RIGHT);

  algorithm.init(Quoridor::CANONICAL_SHAPE, Quoridor::NUM_PLAYERS + 1, Quoridor::NUM_ACTIONS);
  auto context = algorithm.compute(game);
  context->step(/*iterations=*/1000);
  auto action = context->best_move();

  EXPECT_EQ(action, Quoridor::MOVE_RIGHT);
}
